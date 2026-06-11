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
with open("_system_prompt.txt", "r") as f:
    system_prompt = f.read()
with open("_user_prompt_template.txt", "r") as f:
    user_prompt_template = f.read()

# load information about equation
equation_infos = pd.read_csv("_Matsubara2022_equation_information.csv")


res = dict()

def extract_variable_descriptions(equation_infos, equation_key):
    chapter_matches = equation_infos[(equation_infos['EquationChapter']== equation_key)]
    if (len(chapter_matches) == 0):
      raise ValueError(f"No Match for chapter '{equation_key}'")
    return "\n".join([desc for desc in chapter_matches['Variables_Description_LaTeX'] ])


def extract_variable_latex(equation_infos, equation_key, variable_key):
    variable_match = equation_infos[(equation_infos['EquationChapter']== equation_key) & (equation_infos['Variable_Key']== variable_key)]
    if (len(variable_match) != 1):
      raise ValueError(f"No Match for unique combination chapter '{equation_key}' and variable '{variable_key}'")
    return variable_match['Variable_LaTeX'].iloc[0].replace("$","")

def extract_full_variable_domain(equation_infos, equation_key):
    chapter_matches = equation_infos[equation_infos['EquationChapter']== equation_key]
    if (len(chapter_matches) == 0):
      raise ValueError(f"No Match for chapter '{equation_key}'")
    return " \\wedge ".join([row['Variables_Domain'] 
                      for _, row in chapter_matches.iterrows()])

def extract_shape_comparator(descriptor):
    if descriptor == "zero":
      comparator = "= 0"
    elif descriptor == "positive":
      comparator = "\\geq 0"
    elif descriptor == "negative":
      comparator = "\\leq 0"
    else:
      raise ValueError(f"Unknown descriptor: {descriptor}")
    return comparator

for equation_dictionary_entry in list(srsdf.AllEquations.items()): 
  equation_key, equation_value = equation_dictionary_entry
  
  print(equation_key)
  bench = Benchmark(equation_value, initialize_constraint_checking_datasets=False)
  valid_shapes = bench.get_constraints()

  var_pairs = zip(bench.equation.get_vars(),bench.equation.get_var_names())


  for shape in valid_shapes:
    # Format constraint into LaTeX notation
    order = shape['order_derivative']

    # Build sample space condition
    if order != 0:
      variable_latex = extract_variable_latex(equation_infos, equation_key, shape['var_name'])

    comparator = extract_shape_comparator(shape['descriptor'])
    space_conditions = " \\wedge ".join([ f"{variable_latex} \\in [{variable_domain['low']},{variable_domain['high']}]" 
                                         for variable_latex, variable_domain 
                                         in zip([extract_variable_latex(equation_infos, equation_key, vd["name"]) 
                                                 for vd in shape['sample_space']], shape['sample_space'])])
    
    # Build derivative notation
    if order == 0:
      shape_property_latex = f"{space_conditions} \\implies f {comparator}"
    elif order == 1:
      shape_property_latex = f"{space_conditions} \\implies \\frac{{\\partial f}}{{\\partial {variable_latex}}} {comparator}"
    else:
      shape_property_latex = f"{space_conditions} \\implies \\frac{{\\partial^{order} f}}{{\\partial {variable_latex}^{order}}} {comparator}"
    
    application_taxonomy = equation_infos[equation_infos['EquationChapter']== equation_key].iloc[0]['Taxonomy-Domain']
    variable_domains = extract_full_variable_domain(equation_infos, equation_key)

    variable_descriptions = extract_variable_descriptions(equation_infos, equation_key) 

    system_prompt = system_prompt
    user_prompt = user_prompt_template.format(application_taxonomy=application_taxonomy,
                                              shape_property_latex = shape_property_latex,
                                              variable_domains = variable_domains,
                                              variable_descriptions = variable_descriptions)
    
    # Create directory for the equation if it doesn't exist
    equation_dir = os.path.join("shapes_to_validate", equation_key)
    os.makedirs(equation_dir, exist_ok=True)

    # Determine constraint file name (numbered)
    constraint_filename = f"shape_{shape['id']}.md"
    constraint_path = os.path.join(equation_dir, constraint_filename)

    # Save user prompt to file
    with open(constraint_path, "w") as f:
      f.write(user_prompt)
    

