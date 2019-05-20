#!/usr/bin/env bash

source env/bin/activate

DATA_DIR="/projects/tir3/users/pmichel1/data"
CKPT_DIR="/projects/tir3/users/pmichel1/checkpoints"

# Binarize the dataset:
TEXT=
fairseq-preprocess \
    --source-lang en \
    --target-lang fr \
    --trainpref $DATA_DIR/wmt15_en_fr/train \
    --validpref $DATA_DIR/wmt15_en_fr/valid \
    --testpref $DATA_DIR/wmt15_en_fr/test \
    --destdir $DATA_DIR/wmt15_en_fr/data-bin \
    --thresholdtgt 0 \
    --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1500 instead
mkdir -p $CKPT_DIR/jsalt_baseline_en_fr
fairseq-train \
    $DATA_DIR/wmt15_en_fr/data-bin \
    -a transformer \
    --source-lang en \
    --target-lang fr \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)'\
    --lr 0.0005 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07' \
    --min-lr '1e-09' \
    --label-smoothing 0.1 \
    --dropout 0.3 \
    --max-tokens 4000 \
    --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-epoch 30 \
    --keep-interval-updates 5 \
    --save-dir $CKPT_DIR/jsalt_baseline_en_fr

# Generate:
fairseq-generate data-bin/wmt17_en_de \
    $DATA_DIR/wmt15_en_fr/data-bin \
    --path $CKPT_DIR/jsalt_baseline_en_fr/checkpoint_best.pt \
    --beam 5 \
    --remove-bpe
