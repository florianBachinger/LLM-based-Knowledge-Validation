import re

HARD_EQNS = {
    'I.6.2a', 'I.6.2', 'I.6.2b', 'I.9.18',
    'I.15.3t', 'I.15.3x', 'I.29.16', 'I.30.3',
    'I.32.17', 'I.34.14', 'I.37.4', 'I.39.22',
    'I.40.1', 'I.41.16', 'I.44.4', 'I.50.26',
    'II.6.15a', 'II.6.15b', 'II.11.17', 'II.11.20',
    'II.11.27', 'II.11.28', 'II.13.23', 'II.13.34',
    'II.24.17', 'II.35.18', 'II.35.21', 'II.36.38',
    'III.4.33', 'III.9.52', 'III.10.19', 'III.21.20',
    'Bonus1.0', 'Bonus2.0', 'Bonus3.0', 'Bonus4.0',
    'Bonus5.0', 'Bonus6.0', 'Bonus7.0', 'Bonus9.0',
    'Bonus10.0', 'Bonus11.0', 'Bonus12.0', 'Bonus13.0',
    'Bonus14.0', 'Bonus15.0', 'Bonus16.0', 'Bonus17.0',
    'Bonus19.0', 'Bonus20.0'
}

filepath = r'c:\scr\GIT\2026-KnowledgeValidation\documentation\equations\1-Constraint-Information.tex'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

output_lines = []
eq_name_pattern = re.compile(r'^(I{1,3}\.\d+[\w.]*|Bonus\d+\.\d+)\s*&')
row_end_pattern = re.compile(r'^(.*)(\\\\)(\s*)(\n?)$')

marked = []

for line in lines:
    # Update column spec: add 'l' at end for new Hard column
    if r'\begin{longtable}' in line:
        line = line.replace('{llllp{4cm}p{4cm}p{4cm}}', '{llllp{4cm}p{4cm}p{4cm}l}')
        output_lines.append(line)
        continue

    m = row_end_pattern.match(line)
    if m:
        before = m.group(1)
        slashes = m.group(2)
        ws = m.group(3)
        nl = m.group(4)

        # First header row
        if 'Equation' in before and 'Taxonomy' in before:
            line = before + '& SRSD Hard ' + slashes + ws + nl
        # Second header row
        elif 'Domain / Subdomain' in before or 'Subdomain / Specific' in before:
            line = before + '& ' + slashes + ws + nl
        else:
            eq_m = eq_name_pattern.match(before.lstrip())
            if eq_m:
                eq_name = eq_m.group(1)
                if eq_name in HARD_EQNS:
                    marked.append(eq_name)
                    line = before + '& \\checkmark ' + slashes + ws + nl
                else:
                    line = before + '& ' + slashes + ws + nl
            else:
                # continuation row
                line = before + '& ' + slashes + ws + nl

    output_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(output_lines)

print(f"Done! Marked {len(marked)} equations as SRSD Hard:")
for e in sorted(marked):
    print(f"  {e}")
print(f"\nTotal: {len(marked)}")
