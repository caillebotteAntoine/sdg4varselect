#!/bin/bash
# SBATCH 

# Export de toutes les variables d'environnement
#$ -V

# Working Directory
#$ -cwd

# Queue
#$ -q long.q

# Erreur
#$ -e ErrAndOut/err$TASK_ID.err

# Sortie 
#$ -o ErrAndOut/out$TASK_ID.out 



seed=$((SGE_TASK_ID))

poetry run python3 joint_model_200_500.py $seed
exit 0