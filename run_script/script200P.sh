#!/bin/bash

systemd-run --user --scope -p CPUQuota=25% 

## Pour lancer mon code (code qui parallélise lui même)
#python3 $source"sdg4varselect/joint_model/multi_run.py"
poetry run python3 run_200_percentage.py



