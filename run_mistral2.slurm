#!/bin/bash
#SBATCH --job-name=run_mistral2
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH -o run_mistral2.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gchf@cin.ufpe.br

# Ativa o ambiente virtual
source $HOME/mistral_tcc/bin/activate

# Executa seu script Python
python run_mistral2.py