#!/bin/bash
# SBATCH 
# Export de toutes les variables d'environnement
#$ -V
# Working Directory
#$ -cwd
# Queue
#$ -q long.q
# Erreur
#$ -e ErrAndOut/err.err
# Sortie 
#$ -o ErrAndOut/out.out
seed=$2
echo $1
echo $2
poetry run python3 model.py $seed $1 $2