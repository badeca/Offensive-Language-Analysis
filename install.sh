#!/bin/bash
#SBATCH --job-name=install_mistral
#SBATCH --mem=24G
#SBATCH -c 32
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH -o install.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gchf@cin.ufpe.br

# === Preparação ===
module load Python3.10
python -m venv mistral_tcc
source $HOME/mistral_tcc/bin/activate

# === Diagnóstico do ambiente ===
echo "Python path:"
which python
python --version

# === Instalação das dependências ===
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers
pip install datasets
pip install accelerate
pip install safetensors
pip install auto-gptq

# (opcional, se precisar de compatibilidade com outras libs de quantização)
pip install bitsandbytes

# === Verificação final ===
echo "Pacotes instalados:"
pip list
