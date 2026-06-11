import pandas as pd
from pathlib import Path

from SCRBenchmark import Benchmark
import SCRBenchmark.SRSDFeynman as srsdf

RESULTS_PATH = Path("results/experiment_results.csv")
EQUATION_INFO_PATH = Path("documentation/equations/Matsubara2022_equation_information.csv")
OUTPUT_PATH = Path("results/experiment_results_merged.csv")

results = pd.read_csv(RESULTS_PATH)
results = results.sort_values(["equation_name", "shape_id"])
equation_info = pd.read_csv(EQUATION_INFO_PATH, sep=";")

# Normalise key columns for a case-insensitive join
equation_info = equation_info.rename(columns={"Equation_Name": "equation_name"})

merged = equation_info.merge(results, on="equation_name", how="left")

for id,row in merged.iterrows():
  bench = Benchmark(srsdf.AllEquations[row['equation_name']], initialize_constraint_checking_datasets=False)
  valid_shapes = bench.get_constraints()
  shapes = [ shape for shape in valid_shapes if shape['id'] == row['shape_id']]
  if(len(shapes) > 0):
    shape = shapes[0]
    merged.loc[id,'var_display_name'] = shape['var_display_name']
    merged.loc[id,'order_derivative'] = shape['order_derivative']

  else:
    print(f"Problem for {row['equation_name']}")

merged.to_csv(OUTPUT_PATH, index=False)
print(f"Saved merged file to {OUTPUT_PATH} ({len(merged)} rows, {len(merged.columns)} columns)")
unmatched = merged[merged["Equation"].isna()]["equation_name"].unique()
if len(unmatched):
    print(f"Warning: {len(unmatched)} equation(s) had no match in equation info: {list(unmatched)}")
