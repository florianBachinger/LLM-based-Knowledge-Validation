import pandas as pd
import numpy as np
from SCRBenchmark import Benchmark
import SCRBenchmark.SRSDFeynman as srsdf
import Udresco2020_Feynman.Functions as ff

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
equation_info['Variables_Description_LaTeX'] = equation_info['Variables']
equation_info['Variables_Domain_LaTeX'] = equation_info['Variable domains']

equation_info = equation_info.drop(columns=['Variable domains'])

equation_info['Variable'] = [ var.split(':')[0].replace('$','').replace('\\','').replace('_','').replace('}','').replace('{','') for var in equation_info['Variables']]
equation_info['Variable_LaTeX'] = [ var.split(':')[0] for var in equation_info['Variables']]

equation_info['Variables_Domain'] = [ var_dom.replace('$','') if '}' not in var_dom else f"{var.replace('$','')}{var_dom.split('}')[1].replace('$','')}" for (var_dom, var) in zip(equation_info['Variables_Domain_LaTeX'],equation_info['Variable_LaTeX'])]
equation_info['Variables_Domain_LaTeX'] = [ var_dom if '}' not in var_dom else f"${var.replace('$','')}{var_dom.split('}')[1]}" for (var_dom, var) in zip(equation_info['Variables_Domain_LaTeX'],equation_info['Variable_LaTeX'])]

equation_info['EquationKey'] = [ [ function["EquationName"]  for function in ff.FunctionsJson if ((('Bonus' in eq) and function["DescriptiveName"].startswith(eq)) or (function["DescriptiveName"].endswith(eq)))
                                                                                                  ][0] for eq in equation_info["Equation"] ]

equation_info['EquationChapter'] = [ f"Feynman{eq.split('.')[0]}Ch{eq.split('.')[1]}Eq{eq.split('.')[2]}" if len(eq.split('.')) > 2 else f"Feynman{eq.split('.')[0]}" for eq in equation_info["Equation_Full"]]

entries = ff.FunctionsJson

equation_info = equation_info[['EquationKey','EquationChapter', 'Equation', 'Equation-Description','Taxonomy-Domain','Taxonomy-Subdomain','Taxonomy-Specific', 'Variable', 'Variable_LaTeX','Variables_Description_LaTeX', 'Variables_Domain', 'Variables_Domain_LaTeX' ]]
equation_info.to_csv("_Udrescu2020_equation_information.csv", index=False)