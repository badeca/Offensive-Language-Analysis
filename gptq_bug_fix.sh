#!/bin/bash
#SBATCH --job-name=gptq_install
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH -o gptq_install.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gchf@cin.ufpe.br

# === Preparação ===
module load Python3.10
source $HOME/Offensive-Language-Analysis/mistral_tcc/bin/activate

# Executa seu script Python
pip install auto-gptq
