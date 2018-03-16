#!/bin/bash

# Call the script to set the environment
python config/scripts/setenv.py

# Update pip to ensure all requirements are fit
pip install -r config/requirements.txt

# Set PYTHONPATH to point to the codebase
NEW_PYTHONPATH=`python config/scripts/get_env_pythonpath.py`
export PYTHONPATH=$PYTHONPATH:$NEW_PYTHONPATH

# Set the PROJECT_ROOT
PROJECT_DIR=`python config/scripts/get_env_dir.py`
export PROJECT_ROOT=$PROJECT_DIR

# copy the credentials from AWS
if [ ! -d $HOME/.aws ]; then
    echo "Setting new credentials for S3"
    cp -r $PROJECT_ROOT/config/.aws $HOME/.aws
else
    echo "Using existing AWS credentials to access S3"
fi
