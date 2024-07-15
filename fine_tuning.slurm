#!/bin/bash
#SBATCH --cpus-per-gpu 10  # important settings
#SBATCH --gpus=1
#SBATCH -e slurm/output/slurm-%j.err  # files for stderr / stdout
#SBATCH -o slurm/output/slurm-%j.out

# CONDA_BASE=$(conda info --base)  # for some reason necessary with conda
# source $CONDA_BASE/etc/profile.d/conda.sh
# conda activate mmdb

source venv/bin/activate

# nvidia-smi  # test if it works
# python -c "import torch; print(torch.cuda.device_count())"

python3 bachelorthesis/fine_tuning.py  # run your stuff

deactivate