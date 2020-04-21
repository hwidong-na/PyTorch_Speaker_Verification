#!/bin/bash
#SBATCH --job-name=timit
#SBATCH --account=rrg-bengioy-ad_gpu     # Yoshua pays for your job
#SBATCH --cpus-per-task=10               # Ask for 10 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=1-00:00:00                # The job will run for 24 hours
#SBATCH --array=0-1%2                    # Run N jobs, M parallel
#SBATCH -o /scratch/nahwidon/slurm-%j.out# Write the log in $SCRATCH
#SBATCH -e /scratch/nahwidon/slurm-%j.err# Write the err in $SCRATCH

source $HOME/python3.8/bin/activate

echo "Running task $SLURM_ARRAY_TASK_ID"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
if [[ $SLURM_ARRAY_TASK_ID == 0 ]];then 
loss_type=autovc-v3-mix
elif [[ $SLURM_ARRAY_TASK_ID == 1 ]];then 
loss_type=autovc-v5-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 3 ]];then 
# loss_type=autovc-v8-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 4 ]];then 
# loss_type=autovc-v6-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 5 ]];then 
# loss_type=autovc-v5-mix # mix for reporter
# elif [[ $SLURM_ARRAY_TASK_ID == 6 ]];then 
# loss_type=autovc-v4
# elif [[ $SLURM_ARRAY_TASK_ID == 7 ]];then 
# loss_type=autovc-v4-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 1 ]];then 
# loss_type=autovc-v5-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 2 ]];then 
# loss_type=autovc-v4-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 3 ]];then 
# loss_type=autovc-v3-mix
# elif [[ $SLURM_ARRAY_TASK_ID == 4 ]];then 
# loss_type=autovc-v2
# elif [[ $SLURM_ARRAY_TASK_ID == 5 ]];then 
# loss_type=autovc
# elif [[ $SLURM_ARRAY_TASK_ID == 6 ]];then 
# loss_type=manhattan
# elif [[ $SLURM_ARRAY_TASK_ID == 7 ]];then 
# loss_type=euclidean
# elif [[ $SLURM_ARRAY_TASK_ID == 8 ]];then 
# loss_type=ge2e-softmax
fi
# using same dataset
script="$SCRATCH/slurm/train_speech_embedder.`date +%Y%m%d.%H%M`.sh"
echo cp $HOME/PyTorch_Speaker_Verification/train_speech_embedder.sh $script
cp $HOME/PyTorch_Speaker_Verification/train_speech_embedder.sh $script
echo $script 2 "$SCRATCH/timit" $loss_type
srun $script 2 "$SCRATCH/timit" $loss_type
