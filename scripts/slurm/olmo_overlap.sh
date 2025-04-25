#!/bin/bash
#SBATCH --job-name=olmo_overlap
#SBATCH --output=log/olmo_overlap_%j.out
#SBATCH --error=log/olmo_overlap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB

apptainer exec container/ python scripts/olmo_overlap.py
