#!/bin/bash

#systemd-run --user --scope -p CPUQuota=25% 

## Pour lancer mon code (code qui parallélise lui même)
#python3 $source"sdg4varselect/joint_model/multi_run.py"
export XLA_FLAGS="--xla_force_host_platform_device_count=<30>"
poetry run python3 new_reg_path.py

#jupyter nbconvert --to html --TagRemovePreprocessor.remove_cell_tags='{"hide_code"}' new_regularization_path.ipynb