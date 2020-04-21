#!/bin/bash
#SBATCH --job-name=timit
#SBATCH --account=rrg-bengioy-ad_gpu     # Yoshua pays for your job
#SBATCH --cpus-per-task=10               # Ask for 10 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=12:00:00                # The job will run for 12 hours
#SBATCH -o /scratch/nahwidon/slurm-%j.out# Write the log in $SCRATCH
#SBATCH -e /scratch/nahwidon/slurm-%j.err# Write the err in $SCRATCH

echo "Steps:"
echo "0: Prepare environment"
echo "1: Prepare data"
echo "2: Process data"
echo "3: Train speech embedder"
echo "4: Test speech embedder"

skip=0
if [[ $1 ]];then
skip=$1
echo "skip $skip steps"

fi
if [[ $2 ]]; then
prevexp=$2
echo "previous expermiment: $prevexp"
fi

# GE2E configuration
if [[ $3 ]]; then
loss_type=$3
fi
if [[ ! $loss_type ]]; then
loss_type='autovc'
fi
echo "loss type: $loss_type"

if [[ $4 ]]; then
num_layer=$4
fi
if [[ ! $num_layer ]]; then
num_layer=3
fi
echo "num layer: $num_layer"

# AucoVC configuration
if [[ $5 ]]; then
num_conv=$5
fi
if [[ ! $num_conv ]]; then
num_conv=3
fi
echo "num conv: $num_conv"
if [[ $6 ]]; then
num_enc=$6
fi
if [[ ! $num_enc ]]; then
num_enc=2
fi
echo "num enc: $num_enc"
if [[ $7 ]]; then
num_dec=$7
fi
if [[ ! $num_dec ]]; then
num_dec=3
fi
echo "num dec: $num_dec"
if [[ $8 ]]; then
num_post=$8
fi
if [[ ! $num_post ]]; then
num_post=3
fi
echo "num post: $num_post"

# Step 0. Prepare environment
source $HOME/python3.8/bin/activate
JOBID=`basename $SLURM_TMPDIR`
mkdir -p $SCRATCH/$JOBID

cd $SLURM_TMPDIR
echo "working dir: `pwd`"
echo "output dir: $SCRATCH/$JOBID"
git clone $HOME/PyTorch_Speaker_Verification
if [ -z "$PS1" ]; then
    echo This shell is not interactive
else
    echo This shell is interactive
    cp $HOME/PyTorch_Speaker_Verification/*.py PyTorch_Speaker_Verification
fi
cd PyTorch_Speaker_Verification

# Step 1. Prepare data
if [[ $skip -lt 1 ]]; then
    unzip -n $SCRATCH/TIMIT.zip -d $SLURM_TMPDIR
fi

# Step 2. Preprocess data
if [[ -s $prevexp/data.tgz ]];then
    tar zxvf $prevexp/data.tgz -C $SLURM_TMPDIR
    ln -sf $prevexp/data.tgz $SCRATCH/$JOBID/
fi
if [[ $skip -lt 2 ]]; then
echo "\
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    train_path_unprocessed: '$SLURM_TMPDIR/TRAIN/*/*/*.wav'
    test_path: '$SLURM_TMPDIR/test_tisv'
    test_path_unprocessed: '$SLURM_TMPDIR/TEST/*/*/*.wav'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
    num_workers: 1 #number of workers/cpu for data_preprocessor
    log_file: '$SLURM_TMPDIR/data.log'

" > config/config.yaml

python data_preprocess.py
ret=$?
if [[ $ret -ne 0 ]];then
echo "terminate python data_preprocess.py"
exit $ret
fi

if [[ ! -s $SCRATCH/$JOBID/data.tgz ]]; then
    (cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/data.tgz train_tisv test_tisv) &
fi
mv config/config.yaml "$SLURM_TMPDIR/config.data.yaml"

fi

# Step 2. Train speech embedder
if [[ -s $prevexp/model.tgz ]];then
    tar xvf $prevexp/model.tgz -C $SLURM_TMPDIR
fi
model_path="$SLURM_TMPDIR/checkpoint/ckpt_epoch_570.pth"
if [[ -s $model_path ]]; then
    restore=true
else
    restore=false
fi
mix=false
style=false
norm=1
for i in `seq 0 1 1`; do
    suffix="-mix"
    if [[ $loss_type =~ ${suffix}$ ]]; then
        echo "############ mix: true"
        mix=true
        # truncate suffix
        loss_type=${loss_type%%${suffix}}
    fi 
    suffix="-style"
    if [[ $loss_type =~ ${suffix}$ ]]; then
        echo "############ style: true"
        style=true
        # truncate suffix
        loss_type=${loss_type%%${suffix}}
    fi 
done

if [[ $loss_type == "autovc" ]]; then
    N=16 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=0 
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v2" ]]; then
    N=16 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=5 # Number of hallucinated samples per speaker
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v3" ]]; then
    N=16 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=5 # Number of hallucinated samples per speaker
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v4" ]]; then
    N=10 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=4 
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v5" ]]; then
    N=10 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=4 
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v6" ]]; then
    N=10 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=3 
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v7" ]]; then
    N=10 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=4 
    pretrain_epochs=150
elif [[ $loss_type == "autovc-v8" ]]; then
    N=10 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=4 
    pretrain_epochs=150
elif [[ $loss_type == "manhattan" ]]; then
    N=64 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=0 
    pretrain_epochs=0
    loss_type="scaled-norm"
elif [[ $loss_type == "euclidean" ]]; then
    N=64 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=0 
    pretrain_epochs=0
    loss_type="scaled-norm"
    norm=2
else
    N=64 # Number of speakers in a batch
    M=5 # Number of utterances in a batch
    K=0 
    pretrain_epochs=0
fi
if [[ $skip -lt 3 ]]; then
echo "
training: !!bool "true"
device: "cuda"
---
data:
    train_path: '$SLURM_TMPDIR/train_tisv'
    test_path: '$SLURM_TMPDIR/test_tisv'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
---   
model:
    hidden: 768 #Number of LSTM hidden layer units
    num_layer: $num_layer #Number of LSTM layers
    proj: 256 #Embedding size
    model_path: '$model_path'
---
autovc:
    channel: 512 #Number of channels in Conv Layer units
    kernel: 5 #Kernel size in Conv Layer units
    num_conv: $num_conv #Number of Conv layers
    num_enc: $num_enc #Number of LSTM layers in encoder
    dim_neck: 32 # Bottlenect dimension
    num_dec: $num_dec #Number of LSTM layers in decoder
    dim_pre: 512 # Number of channels before LSTM layer in decoder
    num_post: $num_post #Number of LSTM layers in post
    proj: 1024 # Number of LSTM hidden layer units
    freq: 32     # Down sample frequency
    mix: !!bool "$mix" # Mix the original and generated embeddings
    style: !!bool "$style" # Mix the original and generated embeddings
    margin: 0.0
---
train:
    N : $N #Number of speakers in batch
    M : $M #Number of utterances per speaker
    K : $K #Number of transferred utterances per speaker
    num_workers: 0 #number of workers for dataloader
    lr: 0.0001
    optimizer: ADAM
    momentum: 0.9
    epochs: 950 #Max training speaker epoch 
    pretrain_epochs: $pretrain_epochs #Max training speaker epoch 
    log_interval: 5 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/train.log'
    checkpoint_interval: 30 #Save model after x speaker epochs
    checkpoint_dir: '$SLURM_TMPDIR/checkpoint'
    restore: !!bool "$restore" #Resume training from previous model path
    loss_type: '$loss_type'
    norm: $norm
    reports: 'v_intra:v_inter:dbi'
---
test:
    N : 10 #Number of speakers in batch
    M : 6 #Number of utterances per speaker
    K : 1 #Number of support set per speaker
    num_workers: 0 #number of workers for data laoder
    log_interval: 0 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/train.test.log'
" > config/config.yaml
cp config/config.yaml "$SLURM_TMPDIR/config.train.yaml"

#(while :; do sleep 3; echo "a"; done)
(
cd $SLURM_TMPDIR
while : 
do
    sleep 600 # store partial results every 10 min
    if [[ ! -s $SCRATCH/$JOBID/model.tgz ]];then
        tar cvf $SCRATCH/$JOBID/model.tgz checkpoint
    else
        tar uvf $SCRATCH/$JOBID/model.tgz checkpoint
    fi
    tar zcvf $SCRATCH/$JOBID/log.tgz *.log
    tar zcvf $SCRATCH/$JOBID/config.tgz *.yaml
done 
) &
pid=$!

python train_speech_embedder.py
ret=$?
if [[ $ret -ne 0 ]];then
    echo "terminate python train_speech_embedder.py"
    kill -9 $pid
    killall sleep
    exit $ret
fi

kill -9 $pid
(cd $SLURM_TMPDIR; 
if [[ ! -s $SCRATCH/$JOBID/model.tgz ]];then
    tar cvf $SCRATCH/$JOBID/model.tgz checkpoint
else
    tar uvf $SCRATCH/$JOBID/model.tgz checkpoint
fi
) &
pid=$!
fi

# Step 3. Test speech embedder
if [[ $skip < 4 ]];then
for testK in 5 1; do

echo "
training: !!bool "false"
device: "cuda"
---
data:
    test_path: '$SLURM_TMPDIR/test_tisv'
    data_preprocessed: !!bool "true" 
    sr: 16000
    nfft: 512 #For mel spectrogram preprocess
    window: 0.025 #(s)
    hop: 0.01 #(s)
    nmels: 40 #Number of mel energies
---   
model:
    hidden: 768 #Number of LSTM hidden layer units
    num_layer: $num_layer #Number of LSTM layers
    proj: 256 #Embedding size
    model_path: '$SLURM_TMPDIR/checkpoint/final_epoch_950.model'
    # model_path: '$model_path'
---
autovc:
    channel: 512 #Number of channels in Conv Layer units
    kernel: 5 #Kernel size in Conv Layer units
    num_conv: $num_conv #Number of Conv layers
    num_enc: $num_enc #Number of LSTM layers in encoder
    dim_neck: 32 # Bottlenect dimension
    num_dec: $num_dec #Number of LSTM layers in decoder
    dim_pre: 512 # Number of channels before LSTM layer in decoder
    num_post: $num_post #Number of LSTM layers in post
    proj: 1024 # Number of LSTM hidden layer units
    freq: 32     # Down sample frequency
---
train:
    loss_type: '$loss_type'
    K : 5 #Number of transferred utterances per speaker
    norm: $norm
    reports: 'v_intra:v_inter:dbi'
---
test:
    N : 10 #Number of speakers in batch
    M : $(($testK + $M)) #Number of utterances per speaker
    K : $testK #Number of support set per speaker
    num_workers: 0 #number of workers for data laoder
    epochs: 10 #testing speaker epochs
    log_interval: 5 #Epochs before printing progress
    log_file: '$SLURM_TMPDIR/test.K$testK.log'
" > config/config.yaml
cp config/config.yaml "$SLURM_TMPDIR/config.test.K$testK.yaml"

python train_speech_embedder.py
done

fi

(cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/log.tgz *.log) 
(cd $SLURM_TMPDIR; tar zcvf $SCRATCH/$JOBID/config.tgz *.yaml) 
if [[ $pid ]]; then
    wait $pid
fi
