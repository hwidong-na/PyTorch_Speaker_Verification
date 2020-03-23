#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=10               # Ask for 10 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/nahwidon/slurm-%j.out# Write the log in $SCRATCH
#SBATCH -e /scratch/nahwidon/slurm-%j.err# Write the err in $SCRATCH

skip=0
if [[ $1 ]];then
skip=$1
fi

# Step 0. Prepare environment
source $HOME/python3.8/bin/activate
JOBID=`basename $SLURM_TMPDIR`
mkdir -p $SCRATCH/$JOBID

# Step 0. Prepare source
cd $SLURM_TMPDIR
echo "working dir: `pwd`"
echo "output dir: $SCRATCH/$JOBID"
rm -rf PyTorch_Speaker_Verification
git clone $HOME/PyTorch_Speaker_Verification
cd PyTorch_Speaker_Verification

# Step 0. Prepare data
unzip -n $SCRATCH/darpa-timit-acousticphonetic-continuous-speech.zip -d $SLURM_TMPDIR

# Step 1. Preprocess data
if [[ $skip < 1 ]]; then
echo "\
unprocessed_data: '$SLURM_TMPDIR/data/*/*/*/*.wav'
---
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    train_path_unprocessed: '$SLURM_TMPDIR/data/TRAIN/*/*/*.wav'
    test_path: '$SLURM_TMPDIR/test_tisv'
    test_path_unprocessed: '$SLURM_TMPDIR/data/TEST/*/*/*.wav'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
    tisv_frame: 180 #Max number of time steps in input after preprocess
    tisv_frame_min: 140 #Min number of time steps in input after preprocess
    tisv_frame_max: 180 #Max number of time steps in input after preprocess
    alpha: 0.5 #for mixing
" > config/config.yaml

python data_preprocess.py

cp -r $SLURM_TMPDIR/train_tisv $SCRATCH/$JOBID/
cp -r $SLURM_TMPDIR/test_tisv $SCRATCH/$JOBID/

fi

# Step 2. Train speech embedder
if [[ $skip < 2 ]]; then
echo "
training: !!bool "true"
device: "cuda"
---
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    train_path_unprocessed: '$SLURM_TMPDIR/data/TRAIN/*/*/*.wav'
    test_path: '$SLURM_TMPDIR/test_tisv'
    test_path_unprocessed: '$SLURM_TMPDIR/data/TEST/*/*/*.wav'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
    tisv_frame: 180 #Max number of time steps in input after preprocess
    tisv_frame_min: 140 #Min number of time steps in input after preprocess
    tisv_frame_max: 180 #Max number of time steps in input after preprocess
    alpha: 0.5 #for mixing
---   
model:
    hidden: 768 #Number of LSTM hidden layer units
    num_layer: 3 #Number of LSTM layers
    proj: 256 #Embedding size
    model_path: '$SLURM_TMPDIR/model.model' #Model path for testing, inference, or resuming training
---
train:
    N : 64 #Number of speakers in batch
    M : 10 #Number of utterances per speaker
    num_workers: 0 #number of workers for dataloader
    lr: 0.01 
    epochs: 950 #Max training speaker epoch 
    log_interval: 1 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/checkpoint/train.log'
    checkpoint_interval: 120 #Save model after x speaker epochs
    checkpoint_dir: '$SLURM_TMPDIR/checkpoint'
    restore: !!bool "false" #Resume training from previous model path
    loss_type: 'contrast'
" > config/config.yaml

python train_speech_embedder.py

cp -r $SLURM_TMPDIR/checkpoint $SCRATCH/$JOBID/
fi

# Step 3. Test speech embedder
if [[ $skip < 3 ]];then
echo "
training: !!bool "false"
device: "cuda"
---
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    train_path_unprocessed: '$SLURM_TMPDIR/data/TRAIN/*/*/*.wav'
    test_path: '$SLURM_TMPDIR/test_tisv'
    test_path_unprocessed: '$SLURM_TMPDIR/data/TEST/*/*/*.wav'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
    tisv_frame: 180 #Max number of time steps in input after preprocess
    tisv_frame_min: 140 #Min number of time steps in input after preprocess
    tisv_frame_max: 180 #Max number of time steps in input after preprocess
    alpha: 0.5 #for mixing
---   
model:
    hidden: 768 #Number of LSTM hidden layer units
    num_layer: 3 #Number of LSTM layers
    proj: 256 #Embedding size
    model_path: '$SLURM_TMPDIR/checkpoint/final_epoch_950.model'
---
test:
    N : 32 #Number of speakers in batch
    M : 10 #Number of utterances per speaker
    K : 1 #Number of support set per speaker
    num_workers: 0 #number of workers for data laoder
    epochs: 10 #testing speaker epochs
    log_interval: 1 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/checkpoint/test.log'
" > config/config.yaml

python train_speech_embedder.py

fi
