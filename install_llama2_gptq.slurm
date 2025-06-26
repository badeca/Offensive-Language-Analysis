#!/bin/bash
#SBATCH --job-name=install_llama2_gptq
#SBATCH --mem=24G
#SBATCH -c 32
#SBATCH -p short
#SBATCH --gpus=1
#SBATCH -o install_llama2_gptq.log
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=gchf@cin.ufpe.br

# === Preparação ===
module load Python3.10

# Cria o ambiente virtual (caso ainda não exista)
python -m venv $HOME/Offensive-Language-Analysis/llama2_tcc
source $HOME/Offensive-Language-Analysis/llama2_tcc/bin/activate

# === Diagnóstico do ambiente ===
echo "Python path:"
which python
python --version

# === Instalação do PyTorch com CUDA (ajuste se necessário) ===
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# === Instalação das dependências principais ===
pip install transformers accelerate safetensors datasets

# === Instalação da versão recomendada do auto-gptq via Git (com CUDA/ExLlama v2) ===
pip uninstall -y auto-gptq exllama exllamav2 || true
git clone --depth 1 https://github.com/PanQiWei/AutoGPTQ.git /tmp/AutoGPTQ
BUILD_CUDA_EXT=1 pip install /tmp/AutoGPTQ
rm -rf /tmp/AutoGPTQ

# === (opcional) Suporte extra ===
pip install bitsandbytes protobuf openpyxl

# === Verificação final ===
echo "Instalação concluída. Lista de pacotes:"
pip list
