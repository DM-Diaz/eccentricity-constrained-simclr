#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

# SBATCH --exclude=mind-0-18,mind-0-20,mind-0-22,mind-0-24,mind-1-7,mind-1-9

# ,mind-1-11,mind-1-19

echo $SLURM_JOBID
echo $SLURM_NODELIST

# Initialize conda properly
# (has to be set to your own home folder)
source /home/mmhender/anaconda3/etc/profile.d/conda.sh

conda activate env2025

# go to folder where script is located
cd /lab_data/hendersonlab/projects/eccbias/code/

debug=0
# debug=1
subjects=(1 2 3 4 5 6 7 8)
# subjects=(1 2 3 4 5 6)
# subjects=(1)
# 
sample_batch_size=500

# training_types=('Baseline' 'FoveaGaze' 'PeriphNonTTM' 'PeriphTTM' 'pretrained-simclr' 'pretrained-imgnet')
# training_types=('pretrained-simclr2')
# training_types=('Fusion')
# training_types=('simclr-imgnet100')
training_types=('simclr-imgnet1k')

n_components=200

model_architecture='resnet18'
# model_architecture='resnet50-fusion'

for subject in "${subjects[@]}"
do

    for training_type in "${training_types[@]}"
    do
        
        echo $subject
        echo $training_type
        
        python3 -c 'import extract_simclr_eccbias; extract_simclr_eccbias.extract_NSD_features('${subject}',"'${training_type}'", '${debug}', '${sample_batch_size}',"'${model_architecture}'")'  

        python3 -c 'import extract_simclr_eccbias; extract_simclr_eccbias.pca_NSD_features('${subject}',"'${training_type}'", '${n_components}',"'${model_architecture}'")'  
        
    done

done
