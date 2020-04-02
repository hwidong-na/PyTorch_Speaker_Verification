#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=10               # Ask for 10 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH --array=0-1%2                    # Run 2 jobs, 2 parallel
#SBATCH -o /scratch/nahwidon/slurm-%j.out# Write the log in $SCRATCH
#SBATCH -e /scratch/nahwidon/slurm-%j.err# Write the err in $SCRATCH

source $HOME/python3.8/bin/activate

echo "Running task $SLURM_ARRAY_TASK_ID"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
if [[ $SLURM_ARRAY_TASK_ID == 0 ]];then 
loss_type=autovc
elif [[ $SLURM_ARRAY_TASK_ID == 1 ]];then 
loss_type=euclidean
if [[ $SLURM_ARRAY_TASK_ID == 2 ]];then 
loss_type=softmax
elif [[ $SLURM_ARRAY_TASK_ID == 3 ]];then 
loss_type=contrast
fi
# using same dataset
echo $HOME/PyTorch_Speaker_Verification/train_speech_embedder.sh 1 $SCRATCH/nahwidon.6588234.0 $loss_type
$HOME/PyTorch_Speaker_Verification/train_speech_embedder.sh 1 $SCRATCH/nahwidon.6588234.0 $loss_type
