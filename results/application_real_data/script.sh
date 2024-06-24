#!/bin/bash
# SBATCH 
# Export de toutes les variables d'environnement
#$ -V
# Working Directory
#$ -cwd
# Queue
#$ -q infinit.q
# Erreur
#$ -e ErrAndOut/err$TASK_ID.err
# Sortie 
#$ -o ErrAndOut/out$TASK_ID.out 
seed=$((SGE_TASK_ID))
poetry run python3 model.py $seed