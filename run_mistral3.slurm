#!/bin/bash
#SBATCH --job-name=run_mistral3
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH -o run_mistral3.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gchf@cin.ufpe.br

# Ativa o ambiente virtual
source $HOME/mistral_tcc/bin/activate

# Executa seu script Python
python run_mistral3.py