#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00
#SBATCH --exclude=mind-0-18,mind-0-20,mind-0-22,mind-0-24,mind-1-7,mind-1-9,mind-1-11,mind-1-19,mind-1-15

echo $SLURM_JOBID
echo $SLURM_NODELIST

# Initialize conda properly
# (has to be set to your own home folder)
source /home/mmhender/anaconda3/etc/profile.d/conda.sh

conda activate env2025

# go to folder where script is located
cd /lab_data/hendersonlab/projects/eccbias/code/

debug=0
subjects=(1 2 3 4 5 6 7 8)
# subjects=(1)

# model_name1='resnet18-Baseline'
# model_name2='resnet18-pretrained-simclr'
model_name1='resnet18-PeriphNonTTM'
model_name2='resnet18-PeriphTTM'


model_layer="concat"

for subject in "${subjects[@]}"
do

    echo $subject
    python fit_eccbias_varpart_model_nsd.py --subject $subject --model_name1 $model_name1 --model_name2 $model_name2  --model_layer $model_layer --debug $debug

done
    
