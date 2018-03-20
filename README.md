# RepPlot

Author: Andreu Mora (andreu.mora@gmail.com)

## What is it

Repplot is a library for plotting scientific reports in an interactive way. It is based on the pandas DataFrame data 
structures and runs over matplotlib (static) or Bokeh (interactive).

It all started based on this [post](http://www.hoberlab.com/2017/11/16/visualizing-things-with-bokeh/).

You can find some interactive examples for [reports](http://www.hoberlab.com/wp-content/uploads/2017/11/random.html) and [scatter clouds](http://www.hoberlab.com/wp-content/uploads/2017/11/cluster.html)

![An example of browsable scatter cloud](http://www.hoberlab.com/wp-content/uploads/2017/11/bokeh_plot.png)

![An example of browsable time series](http://www.hoberlab.com/wp-content/uploads/2017/11/bokeh_plot-1.png)

The magic of repplot can be appreciated in the repo by running either:

* [The showroom Notebook](https://github.com/drublackberry/repplot/blob/master/notebook/showroom.ipynb)
* A Flask server included in `bin/run_plot_server` that allows to request different data and plot parameters.


### Running the flask server

Just type 

`> python bin/run_plot_server.py -p 5000`

Or any other port you wish to use, and then direct yourself to either:

* `HOST/delta_report` or customize the deltas with `HOST/delta_report?vs=ham`
* `HOST/error_report`
* `HOST/scatter` or customize the x, y, color and size of the scatter plot `HOST/scatter?s=ham`



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
