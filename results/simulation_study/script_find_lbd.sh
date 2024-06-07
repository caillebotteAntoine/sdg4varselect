#!/bin/bash
# SBATCH 
# Export de toutes les variables d'environnement
#$ -V
# Working Directory
#$ -cwd
# Queue
#$ -q long.q
# Erreur
#$ -e err.err
# Sortie 
#$ -o out.out

poetry run python3 find_best_lbd_grid.py