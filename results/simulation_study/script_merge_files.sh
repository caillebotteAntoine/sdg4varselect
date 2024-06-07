#!/bin/bash
# SBATCH 
# Export de toutes les variables d'environnement
#$ -V
# Working Directory
#$ -cwd
# Queue
#$ -q short.q
# Erreur
#$ -e merge_err.err
# Sortie 
#$ -o merge_out.out
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
poetry run python3 merge_file.py $1 $2 $3 $4 $5 $6
exit 0