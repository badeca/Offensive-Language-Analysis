#!/bin/bash
#SBATCH --job-name=run_mistral1
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH -o run_mistral1.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gchf@cin.ufpe.br

# === Preparação ===
module load Python3.10
source $HOME/Offensive-Language-Analysis/mistral_tcc/bin/activate

# === Diagnóstico do ambiente ===
echo "Python path:"
which python
python --version

# Executa seu script Python
python -u run_mistral1.py
