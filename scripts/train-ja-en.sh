#!/usr/bin/env bash

source env/bin/activate

DATA_DIR="/projects/tir3/users/pmichel1/data"
CKPT_DIR="/projects/tir3/users/pmichel1/checkpoints"

# Binarize the dataset:
TEXT=
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA_DIR/en-ja/train \
    --validpref $DATA_DIR/en-ja/valid \
    --testpref $DATA_DIR/en-ja/test \
    --destdir $DATA_DIR/en-ja/data-bin \
    --thresholdtgt 0 \
    --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1500 instead
mkdir -p $CKPT_DIR/jsalt_baseline_ja_en
fairseq-train \
    $DATA_DIR/en-ja/data-bin \
    --source-lang ja \
    --target-lang en \
    --arch transformer \
    --encoder-embed-dim 512
    --encoder-ffn-embed-dim 1024
    --encoder-attention-heads 4
    --encoder-layers 6
    --encoder-normalize-before True
    --decoder-embed-dim 512
    --decoder-ffn-embed-dim 1024
    --decoder-attention-heads 4
    --decoder-layers 6
    --decoder-normalize-before True
    --dropout 0.1
    --share-all-embeddings True
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
    --save-dir $CKPT_DIR/jsalt_baseline_ja_en

# Generate:
fairseq-generate data-bin/en-ja \
    $DATA_DIR/en-ja/data-bin \
    --path $CKPT_DIR/jsalt_baseline_ja_en/checkpoint_best.pt \
    --beam 5 \
    --remove-bpe
