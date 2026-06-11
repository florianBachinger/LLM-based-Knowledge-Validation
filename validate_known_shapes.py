import json
import os
import re
import glob
import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.getenv("OLLAMA_BASE_URL")
MODELS = "mathstral:7b,ministral-3:8b,gemma4:e2b,qwen3.5:4b,glm4:9b".split(",")
MODELS = [m.strip() for m in MODELS]

NUM_REPETITIONS = 1
MAX_RETRIES = 2        # additional attempts on parse failure (total max)
TEMPERATURE = 0.05
TIMEOUT_SECONDS = 500
MAX_WORKERS = 6


SHAPES_DIR = "shapes_to_validate"
RESULTS_DIR = "results"

# Load system prompt
with open("_system_prompt.txt", "r") as f:
    system_prompt = f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _escape_control_chars_in_json_strings(s: str) -> str:
    """Escape literal control characters (newlines, tabs, etc.) that appear
    inside JSON string values, leaving structural whitespace untouched."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_string:
            if ch == '\\' and i + 1 < len(s):
                # Escaped char inside string – pass through both
                result.append(ch)
                i += 1
                result.append(s[i])
            elif ch == '"':
                in_string = False
                result.append(ch)
            elif ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            if ch == '"':
                in_string = True
            result.append(ch)
        i += 1
    return ''.join(result)


def parse_response(content: str):
    """Try to extract and validate the expected JSON structure from model output.

    Returns the parsed dict on success, or None on failure.
    Expected schema: {"result": "valid"|"invalid", "explanation": "<string>"}
    """
    # 1. Try direct parse
    candidates = [content]

    # 2. Try extracting from markdown code fences  ```json ... ```  or  ``` ... ```
    code_blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    candidates.extend(code_blocks)

    # 3. Try extracting first top-level { ... } blob
    brace_match = re.search(r"\{[\s\S]*\}", content)
    if brace_match:
        candidates.append(brace_match.group())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            # Fix invalid JSON escapes (e.g. LaTeX \geq, \theta) produced
            # by models that embed LaTeX in JSON strings.  Process each
            # backslash-sequence and double the backslash only when the
            # following character is not a valid JSON escape starter.
            def _fix_escape(m):
                seq = m.group(0)
                if seq[1] in '"\\/bfnrtu':
                    return seq          # valid JSON escape – keep as-is
                return '\\' + seq       # invalid escape – add backslash
            try:
                fixed = re.sub(r'\\[\s\S]', _fix_escape, candidate)
                parsed = json.loads(fixed)
            except (json.JSONDecodeError, TypeError):
                # Try escaping literal control chars inside JSON strings
                try:
                    fixed_ctrl = _escape_control_chars_in_json_strings(candidate)
                    parsed = json.loads(fixed_ctrl)
                except (json.JSONDecodeError, TypeError):
                    # Combine both fixes: control chars + invalid escapes
                    try:
                        fixed_both = re.sub(r'\\[\s\S]', _fix_escape, fixed_ctrl)
                        parsed = json.loads(fixed_both)
                    except (json.JSONDecodeError, TypeError):
                        continue

        result_val = parsed.get("result")
        explanation_val = parsed.get("explanation")

        if result_val in ("valid", "invalid") and isinstance(explanation_val, str):
            return parsed

    return None


def query_model(model: str, system_prompt: str, user_prompt: str, temperature: float):
    """Send a single stateless chat completion request.

    Each call transmits the full messages array — no session/conversation state
    is carried over between calls.  The Ollama /v1/chat/completions endpoint is
    inherently stateless, so every request is a fresh context window.

    Returns (raw_content: str, parsed: dict | None).
    On request errors, raw_content contains the error description.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()

        data = resp.json()
        raw_response = json.dumps(data, indent=2, ensure_ascii=False)

        choices = data.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "")

        parsed = parse_response(content)
        if parsed is None:
            reasoning = message.get("reasoning", "")
            if reasoning:
                parsed = parse_response(reasoning)

        return raw_response, parsed

    except requests.exceptions.RequestException as e:
        return str(e), None


def save_log(log_path: str, content: str):
    """Write raw model output to a log file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)


def discover_shapes(shapes_dir: str):
    """Walk shapes_to_validate/ and return a sorted list of
    (equation_name, shape_id, filepath) tuples.
    """
    shape_pattern = re.compile(r"^shape_(\d+)\.md$")
    entries = []

    for equation_name in sorted(os.listdir(shapes_dir)):
        equation_path = os.path.join(shapes_dir, equation_name)
        if not os.path.isdir(equation_path):
            continue

        for fname in sorted(os.listdir(equation_path)):
            m = shape_pattern.match(fname)
            if m:
                shape_id = int(m.group(1))
                filepath = os.path.join(equation_path, fname)
                entries.append((equation_name, shape_id, filepath))

    return entries


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def load_completed(csv_path: str):
    """Load already-completed result keys from the CSV file.

    Returns a set of (model, equation_name, shape_id, repetition_id) tuples.
    """
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return set()
    return set(
        (row["model"], row["equation_name"], int(row["shape_id"]), int(row["repetition_id"]))
        for _, row in df.iterrows()
    )


# Lock for thread-safe CSV writes
_csv_lock = threading.Lock()


def append_record(csv_path: str, record: dict):
    """Append a single result record to the CSV, creating the file if needed."""
    with _csv_lock:
        write_header = not os.path.exists(csv_path)
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode="a", header=write_header, index=False)


def evaluate_task(task: dict, csv_path: str):
    """Evaluate a single (model, shape, repetition) task with retries.

    Returns the result record dict.
    """
    model = task["model"]
    equation_name = task["equation_name"]
    shape_id = task["shape_id"]
    rep = task["rep"]
    user_prompt = task["user_prompt"]
    log_dir = task["log_dir"]


    raw_output = ""
    parsed = None
    error_message = None
    success = False
    log_path = None
    attempts = 0

    while attempts <= MAX_RETRIES:
        raw_output, parsed = query_model(
            model, system_prompt, user_prompt, TEMPERATURE
        )

        log_filename = f"rep_{rep}_attempt_{attempts + 1}_output.log"
        log_path = os.path.join(log_dir, log_filename)
        save_log(log_path, raw_output)

        if parsed is not None:
            success = True
            break

        attempts += 1

    if not success:
        error_message = (
            f"Failed to parse valid JSON after {MAX_RETRIES + 1} attempts"
        )

    explanation = parsed["explanation"] if parsed else None
    if explanation:
        explanation = " ".join(explanation.split())

    record = {
        "model": model,
        "equation_name": equation_name,
        "shape_id": shape_id,
        "repetition_id": rep,
        "success": success,
        "result": parsed["result"] if parsed else None,
        "explanation": explanation,
        "error_message": error_message,
        "log_path": log_path,
        "think": False 
    }
    append_record(csv_path, record)
    return record


def main():
    shapes = discover_shapes(SHAPES_DIR)
    total_shapes = len(shapes)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    completed = load_completed(csv_path)

    # Build full task list, filtering out already-completed work
    tasks = []
    total_skipped = 0
    for model in MODELS:
        for equation_name, shape_id, filepath in shapes:
            with open(filepath, "r", encoding="utf-8") as f:
                user_prompt = f.read()
            log_dir = os.path.join(
                RESULTS_DIR, "logs", model, equation_name, f"shape_{shape_id}"
            )
            os.makedirs(log_dir, exist_ok=True)
            for rep in range(1, NUM_REPETITIONS + 1):
                key = (model, equation_name, shape_id, rep)
                if key in completed:
                    total_skipped += 1
                    continue
                tasks.append({
                    "model": model,
                    "equation_name": equation_name,
                    "shape_id": shape_id,
                    "rep": rep,
                    "user_prompt": user_prompt,
                    "log_dir": log_dir,
                })

    print(f"Discovered {total_shapes} shape file(s) across "
          f"{len(set(e[0] for e in shapes))} equation(s).")
    print(f"Models:      {', '.join(MODELS)}")
    print(f"Repetitions: {NUM_REPETITIONS}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Workers:     {MAX_WORKERS}")
    if total_skipped:
        print(f"Resuming:    {total_skipped} result(s) already in CSV — will skip")
    print(f"Tasks to run: {len(tasks)}")
    print("=" * 70)

    total_new = 0
    total_success = 0
    total_fail = 0
    start_time = datetime.datetime.now()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(evaluate_task, task, csv_path): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            label = f"{task['model']}  {task['equation_name']}/shape_{task['shape_id']}  rep={task['rep']}"
            try:
                record = future.result()
                total_new += 1
                if record["success"]:
                    total_success += 1
                    print(f"  . {label}", flush=True)
                else:
                    total_fail += 1
                    print(f"  x {label}  — {record['error_message']}", flush=True)
            except Exception as exc:
                total_fail += 1
                print(f"  ! {label}  — exception: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = datetime.datetime.now() - start_time

    print("\n" + "=" * 70)
    print(f"Experiment complete in {elapsed}")
    print(f"New queries: {total_new}  |  Skipped (already done): {total_skipped}")
    print(f"Successes:   {total_success}  |  Failures: {total_fail}")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        successes = df["success"].sum() if not df.empty else 0
        print(f"Total rows in CSV: {len(df)}  |  Successes: {successes}  |  "
              f"Failures: {len(df) - successes}")
    print(f"Results saved to: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
