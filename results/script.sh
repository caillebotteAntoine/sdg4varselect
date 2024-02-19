#!/bin/bash

systemd-run --user --scope -p CPUQuota=1000% 

seed=$((SLURM_ARRAY_TASK_ID))

poetry run python3  running_test_from_script.py $seed

#run_test.py
## Pour lancer mon code (code qui parallélise lui même)
#python3 $source"sdg4varselect/joint_model/multi_run.py"
#export XLA_FLAGS="--xla_force_host_platform_device_count=<30>"
#poetry run python3 run_test.py
#jupyter nbconvert --to html new_regularization_path.ipynb

