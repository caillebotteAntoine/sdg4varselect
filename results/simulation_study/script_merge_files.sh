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
echo $1
poetry run python3 merge_file.py $1 $1 $2 $3 $4
exit 0