import pandas as pd
import numpy as np
from SCRBenchmark import Benchmark
import SCRBenchmark.SRSDFeynman as srsdf

import json
import os
import requests

# load configuration parameters
BASE_URL = os.getenv("OLLAMA_BASE_URL") 
MODELS = os.getenv("OLLAMA_MODELS").split(",") 
MODELS = [m.strip() for m in MODELS] 

# load prompt templates
with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()
with open("user_prompt.txt", "r") as f:
    user_prompt_template = f.read()

# load information about equation
equation_info = pd.read_csv("_Udrescu2020_equation_information.csv",index_col= False)

columns = ['EquationKey','EquationChapter', 'Equation', 'Equation-Description','Taxonomy-Domain','Taxonomy-Subdomain','Taxonomy-Specific','Variable_Key', 'Variable_Name', 'Variable_LaTeX','Variables_Description_LaTeX', 'Variables_Domain', 'Variables_Domain_LaTeX' ]
rows = []


def add_row(variable_data, equation_key,variable_key, variable_name, variable_latex, variable_description_latex, variables_domain, variables_domain_latex):
  rows.append({
      'EquationKey': variable_data['EquationKey'],
      'EquationChapter': variable_data['EquationChapter'],
      'Equation': variable_data['Equation'],
      'Equation-Description': variable_data['Equation-Description'],
      'Taxonomy-Domain': variable_data['Taxonomy-Domain'],
      'Taxonomy-Subdomain': variable_data['Taxonomy-Subdomain'],
      'Taxonomy-Specific': variable_data['Taxonomy-Specific'],
      'Variable_Key': variable_key,
      'Variable_Name': variable_name,
      'Variable_LaTeX': variable_latex,
      'Variables_Description_LaTeX': variable_description_latex,
      'Variables_Domain': variables_domain,
      'Variables_Domain_LaTeX': variables_domain_latex,
  })

for equation_dictionary_entry in list(srsdf.AllEquations.items()): # limit to first 2 for testing
  equation_key, equation_value = equation_dictionary_entry

  print(f"Processing equation: {equation_key}")
  
  bench = Benchmark(equation_value, initialize_constraint_checking_datasets=False)
  equation_level_info = equation_info[equation_info['EquationChapter'] == equation_key]
  if len(equation_level_info) < 1:
    raise ValueError(f"Expected at least 1 row for equation '{equation_key}', found 0")
  equation_level_info = equation_level_info.iloc[0]


  if equation_key == 'FeynmanICh26Eq2': # skip this variable as it has a different name in the equation information file

    add_row(
      equation_level_info,
      equation_key,
      "x0",
      "theta1",
      "$\\theta_1$",
      "$\\theta_1$: angle of incidence",
      "\\theta_1 in [0.0, 2 \\times pi]",
      "$\\theta_1 in [0.0, 2 \\times pi]$",
    )

    add_row(
      equation_level_info,
      equation_key,
      "x1",
      "theta2",
      "$\\theta_2$",
      "$\\theta_2$: angle of refraction",
      "\\theta_2 in [0.0, 2 \\times pi]",
      "$\\theta_2 in [0.0, 2 \\times pi]$",
    )

    continue
  if equation_key == 'FeynmanICh30Eq5': # skip this variable as it has a different name in the equation information file
    
    add_row(
      equation_level_info,
      equation_key,
      "x0",
      "lambda",
      "$\\lambda$",
      "$\\lambda$: wavelength",
      "\\lambda in [1.0e-11, 1.0e-9]",
      "$\\lambda in [1.0e-11, 1.0e-9]$",
    )

    add_row(
      equation_level_info,
      equation_key,
      "x1",
      "n",
      "$n$",
      "$n$: refractive index",
      "\\n in [1.0, 1.0e2]",
      "$\\n in [1.0, 1.0e2]$",
    )
    
    add_row(
      equation_level_info,
      equation_key,
      "x2",
      "theta",
      "$\\theta$",
      "$\\theta$: half-angle of the light cone",
      "\\theta in [0.0, 2 \\times \\pi]",
      "$\\theta in [0.0, 2 \\times \\pi]$",
    )

    continue

  for (variable_key,variable_name,objective) in zip(bench.equation.get_vars(),bench.equation.get_var_names(),bench.equation.sampling_objs):
    print(f"  Processing variable: {variable_name} in equation: {equation_key}")

    range = ""
    
    def format_value(val):
      """Replace multiples of pi with symbolic representation"""
      pi_multiples = [
        (2 * np.pi, "2.0 \\times \\pi"),
        (-2 * np.pi, "-2.0 \\times \\pi"),
        (np.pi, "\\pi"),
        (-np.pi, "-\\pi"),
      ]
      for pi_val, pi_str in pi_multiples:
        if abs(val - pi_val) < 1e-10:
          return pi_str
      return str(val)
    
    min_str = format_value(objective.min_value)
    max_str = format_value(objective.max_value)
    
    if objective.uses_negative and objective.uses_positive:
      range = f"[-{max_str}, -{min_str}] \\cup [{min_str}, {max_str}]"
    elif objective.uses_negative:
      range = f"[{max_str}, {min_str}]"
    elif objective.uses_positive:
      range = f"[{min_str}, {max_str}]"

    variable_info = equation_info[
      (equation_info['Variable'] == variable_name.replace('_',''))
      & (equation_info['EquationChapter'] == equation_key)
    ]
    if len(variable_info) != 1:
      raise ValueError(f"Expected 1 row for variable '{variable_name}' in equation '{equation_key}', found {len(variable_info)}")
    variable_info = variable_info.iloc[0]

    domain = f"{variable_info['Variable_LaTeX'].replace('$','')} \\in {range}"
    add_row(
        variable_info,
        equation_key,
        variable_key,
        variable_name,
        variable_info['Variable_LaTeX'],
        variable_info['Variables_Description_LaTeX'],
        domain,
        f"${domain}$",
    )

res = pd.DataFrame(rows, columns=columns)
print(res.head())
res.to_csv("_Matsubara2022_equation_information.csv", index=False)