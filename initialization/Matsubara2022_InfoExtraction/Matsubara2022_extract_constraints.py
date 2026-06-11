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
equation_info = pd.read_csv("initialization/Udrescu2020_InfoExtraction/Udrescu2020_equation_information_org.csv",sep=";",index_col= False)


res = dict()

for equation_dictionary_entry in list(srsdf.AllEquations.items())[:2]: # limit to first 2 for testing
  equation_key, equation_value = equation_dictionary_entry

  print(f"Processing equation: {equation_key}")
  
  bench = Benchmark(equation_value, initialize_constraint_checking_datasets=False)
  valid_shapes = bench.get_constraints()

  for shape in valid_shapes:
    # Format constraint into LaTeX notation
    var_name = shape['var_name']
    descriptor = shape['descriptor']
    order = shape['order_derivative']
    sample_space = shape['sample_space']
    
    if descriptor == "constant":
      comparator = "= 0"
    elif descriptor == "positive":
      comparator = "\\geq 0"
    elif descriptor == "negative":
      comparator = "\\leq 0"
    else:
      raise ValueError(f"Unknown descriptor: {descriptor}")
    
    # Build sample space condition
    space_conditions = " \\wedge ".join([f"{{{s['name']}}} \\in [{s['low']},{s['high']}]" for s in sample_space])
    
    # Build derivative notation
    if order == 0:
      constraint_str = f"${space_conditions} \\implies f {comparator}$"
    else:
      constraint_str = f"${space_conditions} \\implies \\partial^{order}f/\\partial{{{var_name}}}^{order} {comparator}$"
    
    print(' '*2 + constraint_str)

