#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00

#SBATCH --exclude=mind-0-18,mind-0-20,mind-0-22,mind-0-24,mind-1-7,mind-1-9,mind-1-11,mind-1-19

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

# model_names=('resnet18-Baseline' 'resnet18-FoveaGaze' 'resnet18-PeriphNonTTM' 'resnet18-PeriphTTM' 'resnet18-pretrained-simclr' 'resnet18-pretrained-imgnet')
# model_names=('resnet18-pretrained-simclr2')
# model_names=('resnet18-simclr-imgnet100')
model_names=('resnet18-simclr-imgnet1k')

model_layer="concat"

for subject in "${subjects[@]}"
do

    for model_name in "${model_names[@]}"
    do
        echo $subject
        echo $model_name
        
        python fit_eccbias_model_nsd.py --subject $subject --model_name $model_name --model_layer $model_layer --debug $debug

    done
done
    
