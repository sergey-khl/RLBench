#!/bin/bash

module load python/3.10
module load cuda
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

python -m venv .venv_rlbench
source .venv_rlbench/bin/activate
pip install -r requirements_cc.txt

tar -cavf venv_rlbench.tar .venv_rlbench

pip freeze
