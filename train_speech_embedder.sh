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
if [[ $2 ]]; then
prevexp=$2
echo "previous expermiment: $prevexp"
fi
if [[ $3 ]]; then
loss_type=$3
fi
if [[ ! $loss_type ]]; then
loss_type='euclidean'
fi

mixing_alpha=0.5

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
if [[ -s $prevexp/data.tgz ]];then
    tar zxvf $prevexp/data.tgz -C $SLURM_TMPDIR
fi
if [[ $skip < 1 ]]; then
echo "\
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    train_N: 10000
    train_path_unprocessed: '$SLURM_TMPDIR/data/TRAIN/*/*/*.wav'
    test_N: 250
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
    mixing_alpha: $mixing_alpha #for mixing
    num_workers: 1 #number of workers/cpu for data_preprocessor
    log_file: '$SLURM_TMPDIR/data.log'
    wav_path: '$SLURM_TMPDIR/wav'

" > config/config.yaml

python data_preprocess.py
ret=$?
if [[ $ret -ne 0 ]];then
echo "terminate python data_preprocess.py"
exit $ret
fi

(cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/data.tgz train_tisv test_tisv)
mv config/config.yaml "$SLURM_TMPDIR/config.data.yaml"

fi

# Step 2. Train speech embedder
if [[ -s $prevexp/model.tgz ]];then
    tar zxvf $prevexp/model.tgz -C $SLURM_TMPDIR
fi
if [[ $skip < 2 ]]; then
echo "
training: !!bool "true"
device: "cuda"
---
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    train_path_unprocessed: '$SLURM_TMPDIR/data/TRAIN/*/*/*.wav'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
    tisv_frame: 180 #Max number of time steps in input after preprocess
    tisv_frame_min: 140 #Min number of time steps in input after preprocess
    tisv_frame_max: 180 #Max number of time steps in input after preprocess
---   
model:
    hidden: 768 #Number of LSTM hidden layer units
    num_layer: 3 #Number of LSTM layers
    proj: 256 #Embedding size
---
train:
    N : 64 #Number of speakers in batch
    M : 10 #Number of utterances per speaker
    num_workers: 0 #number of workers for dataloader
    lr: 0.01
    optimizer: SGD
    momentum: 0.9
    epochs: 950 #Max training speaker epoch 
    log_interval: 1 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/train.log'
    checkpoint_interval: 120 #Save model after x speaker epochs
    checkpoint_dir: '$SLURM_TMPDIR/checkpoint'
    restore: !!bool "false" #Resume training from previous model path
    loss_type: '$loss_type'
" > config/config.yaml

python train_speech_embedder.py
ret=$?
if [[ $ret -ne 0 ]];then
echo "terminate python train_speech_embedder.py"
exit $ret
fi

(cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/model.tgz checkpoint)
mv config/config.yaml "$SLURM_TMPDIR/config.train.yaml"
fi

# Step 3. Test speech embedder
if [[ $skip < 3 ]];then
for K in 10 5 1; do

echo "
training: !!bool "false"
device: "cuda"
---
data:
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
---   
model:
    hidden: 768 #Number of LSTM hidden layer units
    num_layer: 3 #Number of LSTM layers
    proj: 256 #Embedding size
    model_path: '$SLURM_TMPDIR/checkpoint/final_epoch_950.model'
---
train:
    loss_type: '$loss_type'
---
test:
    N : 128 #Number of speakers in batch
    M : $(($K + 10)) #Number of utterances per speaker
    K : $K #Number of support set per speaker
    num_workers: 0 #number of workers for data laoder
    epochs: 5 #testing speaker epochs
    log_interval: 1 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/test.K$K.log'
" > config/config.yaml

python train_speech_embedder.py
mv config/config.yaml "$SLURM_TMPDIR/config.test.K$K.yaml"
done

fi

(cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/log.tgz *.log) 
(cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/config.tgz *.yaml) 
