#!/usr/bin/env bash
source scripts/globals.sh
source env/bin/activate


# Binarize the dataset:
TEXT=
fairseq-preprocess \
    --source-lang ja \
    --target-lang en \
    --trainpref $DATA_ROOT/en-ja/train \
    --validpref $DATA_ROOT/en-ja/valid \
    --testpref $DATA_ROOT/en-ja/test \
    --destdir $DATA_ROOT/en-ja/data-bin \
    --joined-dictionary \
    --thresholdtgt 0 \
    --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1500 instead
mkdir -p $CKPT_ROOT/jsalt_baseline_ja_en
fairseq-train \
    $DATA_ROOT/en-ja/data-bin \
    --source-lang ja \
    --target-lang en \
    --arch transformer \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 1024 \
    --encoder-attention-heads 4 \
    --encoder-layers 6 \
    --encoder-normalize-before \
    --decoder-embed-dim 512 \
    --decoder-ffn-embed-dim 1024 \
    --decoder-attention-heads 4 \
    --decoder-layers 6 \
    --decoder-normalize-before \
    --dropout 0.1 \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)'\
    --lr 0.0005 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07' \
    --min-lr '1e-09' \
    --label-smoothing 0.1 \
    --max-tokens 4000 \
    --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --max-epoch 30 \
    --keep-last-epochs 1 \
    --save-dir $CKPT_ROOT/jsalt_baseline_ja_en

# Generate:
fairseq-generate \
    $DATA_ROOT/en-ja/data-bin \
    --path $CKPT_ROOT/jsalt_baseline_ja_en/checkpoint_best.pt \
    --beam 5 \
    --remove-bpe
