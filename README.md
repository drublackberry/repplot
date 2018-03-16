# RepPlot

Author: Andreu Mora (andreu.mora@gmail.com)

## What is it

Repplot is a library for plotting scientific reports in an interactive way. It is based on the pandas DataFrame data 
structures and runs over matplotlib (static) or Bokeh (interactive)

## Dependencies

* Needs python 3 to execute the installation script
* Install gcc (sudo apt-get install gcc)
* Needs pip , conda or virtualenv to install packages, or manage virtual environments
* Packages are istalled through script `config/requirements.txt`

## Workflow

If using virtual environment:

1. Ensure that `CHECK_VIRTUALENV` in `config/project_vars.json` is set to `"Yes"`
2. Adapt the root directory of the repository by setting the variable `PROJECT_DIR` in `config/project_vars.json`. 
3. Adapt the python path by setting the list `PYTHONPATH_LIST` in `config/project_vars.json`. It should normally contain `($PROJECT_ROOT)/src`.
4. Set the virtualenv or conda environment through `source name_of_enviroment`. Note that `name_of_environment` shall match the variable `PROJECT_NAME` in `config/project_vars.json`.
5. Run `source setenv.sh`. This will install and check all dependencies.

If not using virtual environment:

1. Ensure that `CHECK_VIRTUALENV` in `config/project_vars.json` is set to `"No"`
2. Adapt the root directory of the repository by setting the variable `PROJECT_DIR` in `config/project_vars.json`. 
3. Adapt the python path by setting the list `PYTHONPATH_LIST` in `config/project_vars.json`. It should normally contain `($PROJECT_ROOT)/src`.
4. Run `source setenv.sh`. This will install and check all dependencies.

**Cloud settings**

- Make sure that all the database profiles point to remote.
- Make sure that all the paths in config point to the correct folder

Project template created with [Wet Gremlin](https://github.com/drublackberry/wet-gremlin)
