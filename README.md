# Language Model-based Domain Knowledge Validation

The main idea is the validation and explanation of domain knowledge in the form of shape
constraints for a function. Given a candidate constraint (e.g. monotonicity or curvature of a
physical equation), a small language or math model is asked whether the constraint is physically
plausible and to explain its reasoning.

![](<Documentation/experimental setup.drawio.png>)

## Example

From friction-experiment data a symbolic regression algorithm might suggest the constraints
$\frac{\partial \mu_\textit{dyn}}{\partial T} \leq 0$ and
$\frac{\partial^2 \mu_\textit{dyn}}{\partial T^2} \geq 0$.

A model should recognise that overheating brakes reduces braking power
($\frac{\partial \mu_\textit{dyn}}{\partial T} \leq 0$) and that the friction system reaches
thermal equilibrium, making the function convex over temperature
($\frac{\partial^2 \mu_\textit{dyn}}{\partial T^2} \geq 0$).

Because it is unclear whether small language/math models can reliably provide physically
plausible explanations, the experimental setup evaluates several models against a ground-truth
benchmark of known-valid shape constraints.

## Pipeline

1. **Shape files** — each constraint is encoded as a markdown prompt in `shapes_to_validate/<equation>/<shape_N>.md`.
2. **Validation** (`validate_known_shapes.py`) — each shape is sent to one or more Ollama models; the model must respond with `{"result": "valid"|"invalid", "explanation": "..."}`.
3. **Results** — responses are appended to `results/experiment_results.csv`; raw model outputs are stored in `results/logs/`.
4. **Analysis** (`analyze_results.py`) — reads the CSV and produces six summary figures in `figures/`.

## Scripts

### `validate_known_shapes.py`

Runs the main experiment. Key behaviours:
- Discovers all `shape_*.md` files under `shapes_to_validate/` automatically.
- Queries each configured Ollama model via the `/v1/chat/completions` endpoint.
- Retries up to `MAX_RETRIES` times when the model response cannot be parsed as valid JSON.
- Writes results incrementally to the CSV so the run can be resumed after interruption.
- Executes tasks in parallel using a thread pool.

**Current configuration** (top of file):

| Parameter | Value | Description |
|---|---|---|
| `MODELS` | `mathstral:7b, ministral-3:8b, gemma4:e2b, qwen3.5:9b, glm4:9b` | Models to evaluate |
| `NUM_REPETITIONS` | `1` | Repeated queries per shape per model |
| `MAX_RETRIES` | `2` | Extra parse-retry attempts per query |
| `TEMPERATURE` | `0.05` | Low temperature for near-deterministic output |
| `TIMEOUT_SECONDS` | `500` | Per-request HTTP timeout |
| `MAX_WORKERS` | `6` | Parallel worker threads |

The Ollama base URL is read from the `OLLAMA_BASE_URL` environment variable.

### `analyze_results.py`

Reads `results/experiment_results.csv` and saves the following figures to `figures/`:

| File | Content |
|---|---|
| `01_parse_success_rate` | Fraction of queries that returned parseable JSON per model |
| `02_accuracy_per_model` | Correct "valid" call rate per model (all ground-truth shapes are valid) |
| `03_decision_distribution` | Stacked bar of valid vs. invalid decisions per model |
| `03_accuracy_by_order` | Accuracy broken down by constraint order (0th / 1st / 2nd derivative) |
| `04_accuracy_heatmap` | Model × equation accuracy heatmap |
| `05_accuracy_heatmap_selected_equations` | Model × equation accuracy heatmap for the four equations investigated by our domain expert. |

## Models Used in Experiments

Information retrieved on 11 May 2026. The **Ollama Last Updated** date reflects when the Ollama library entry was last refreshed — it may represent a new quantisation, a minor model revision, or an entirely new model generation rather than just a metadata change. Compare it with **Initial Release** to identify version gaps that could affect reproducibility.

| Ollama Tag | Developer | Parameters | Disk (Ollama) | Context Length | Initial Release | Ollama Last Updated | License | Links |
|---|---|---|---|---|---|---|---|---|
| `mathstral:7b` | Mistral AI | 7 B | 4.1 GB | 32K | Jul 2024 | ~May 2025 | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [Model](https://huggingface.co/mistralai/Mathstral-7B-v0.1) · [Ollama](https://ollama.com/library/mathstral) |
| `ministral-3:8b` | Mistral AI | ~8 B | 6.0 GB | 256K | Jan 2026² | Jan 2026 | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [Model](https://huggingface.co/mistralai) · [Ollama](https://ollama.com/library/ministral-3) |
| `gemma4:e2b` | Google DeepMind | 2.3 B eff.¹ | 7.2 GB | 128K | April 2026 | April 2026 | [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license) | [Model](https://huggingface.co/google/gemma-4-E2B) · [Ollama](https://ollama.com/library/gemma4) |
| `glm4:9b` | Zhipu AI (Z.ai) | 9 B | 5.5 GB | 128K | Jun 2024 | ~May 2025 | [GLM-4 License](https://huggingface.co/zai-org/glm-4-9b/blob/main/LICENSE) | [Model](https://huggingface.co/zai-org/glm-4-9b) · [Ollama](https://ollama.com/library/glm4) |

> ¹ **gemma4:e2b** — "E2B" means *effective 2 billion* parameters. The model uses Per-Layer Embeddings (PLE), so the total stored weight is ≈ 5.1 B parameters, which explains the 7.2 GB disk footprint despite the 2.3 B effective parameter count.
>
> ² **ministral-3** on Ollama is a new model family (Apache 2.0, vision support, 3 B / 8 B / 14 B variants, 256 K context), **distinct** from the original Ministral-3B / Ministral-8B models released in October 2024 under the Mistral Research License.
>
> Disk sizes reflect Ollama's default quantisation (Q4\_K\_M or equivalent). "Ollama Last Updated" values are approximated from the "Updated X ago" label on the respective Ollama library page as observed on 11 May 2026.

---

## Setup

### Prerequisites
- Docker Desktop

### Devcontainer
Open this folder in VS Code and select **Reopen in Container**.

### Running the experiment
```bash
python validate_known_shapes.py
```

### Running the analysis
```bash
python analyze_results.py
```

## Future Research

- **Agentic feedback loop** — provide a feedback mechanism so the model can reevaluate its own
  reasoning (e.g. catching cases where it calls a shape invalid due to negative values while the
  variable domain prohibits negative inputs). Possible directions: multi-turn prompts, tool use,
  or skills for mathematical interpretation.
- **LLM-critic** — a separate, larger LLM that re-evaluates the small model's explanation and
  verdict as a validation benchmark.
- **Edge device evaluation** — run the experiment on consumer-grade or CPU-only hardware to
  characterise latency and feasibility for on-premise use.
