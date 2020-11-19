#!/bin/bash
#SBATCH --job-name              ger-dico
#SBATCH --partition             gpu-normal
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --time                  4:00:00
#SBATCH --mem                   50G
#SBATCH --gres                  gpu:1
#SBATCH --output                ger-dico.%j.out
#SBATCH --error                 ger-dico.%j.err
#SBATCH --mail-type		ALL
#SBATCH --mail-user		[yc07420@umac.mo]

source /etc/profile
source /etc/profile.d/modules.sh

#Adding modules
# module add cuda/10.0.130

ulimit -s unlimited

#Your program starts here
nvidia-smi
CUDA_VISIBLE_DEVICES=0 \
MKL_THREADING_LAYER=GNU \
python save_trained_dictionary.py \
--src_lang en \
--tgt_lang fr \
--src_emb /data/home/yc07420/myWorkstation/referPackage/MUSE/myTrainedDic/debug/wmten2fr_nobpe_dict/vectors-en.txt \
--tgt_emb /data/home/yc07420/myWorkstation/referPackage/MUSE/myTrainedDic/debug/wmten2fr_nobpe_dict/vectors-fr.txt \
--emb_dim 300 \
--save_dico_path /data/home/yc07420/myWorkstation/referPackage/MUSE/myTrainedDic/debug/wmten2fr_nobpe_dict
