#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda."
    exit
fi

# Create the environment from the environment.yml file
echo "Creating the conda environment..."
conda env create --file environment.yml

# Initialize conda for the shell if not already done
eval "$(conda shell.bash hook)"

# Activate the environment
echo "Activating the conda environment..."
conda activate python-for-traders

