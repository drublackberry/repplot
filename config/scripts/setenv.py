'''
Script that sets the environment of the current project
'''

import os
import json


def check_env(env):
    envs_str = os.popen("conda info --envs").read()
    ini_ind = envs_str.find("*")
    end_ind = envs_str[ini_ind:].find("\n")
    sub_str = envs_str[ini_ind:end_ind + ini_ind]
    retr_env = sub_str[-len(env):]
    return retr_env == env


# Load the environment variables
with open(os.path.join('config', 'project_vars.json'), 'r') as fp:
    VARS = json.load(fp)

# Check the environment
if VARS['CHECK_VIRTUALENV'].lower() == "yes":
    if check_env(VARS['PROJECT_NAME']):
        print("[OK] Project environment set")
    else:
        print("[ERROR] Set the project environment, run \"source activate " + VARS["PROJECT_NAME"] + "\"")
        exit()
