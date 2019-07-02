#!/bin/bash

####
#Abdul Rafae Khan#
#CUNY#
#JSALT INFORMAL MT#
####

source env/bin/activate


BASE_PATH=/path/to/base/directory

SRC=ar
TRG=en
for EXP in {actual-1k,literal-1k,actual-2k,literal-2k};do
	DATA_ROOT=$BASE_PATH/data/processed/$SRC-$TRG/$EXP
	DATABIN_ROOT=$BASE_PATH/data-bin/$SRC-$TRG/$EXP
	CKPT_ROOT=$BASE_PATH/checkpoints/$SRC-$TRG/$EXP
	
	if [ ! -d "$DATABIN_ROOT" ]
	then
		mkdir -p $DATABIN_ROOT
		echo "Process "$SRC-$TRG.$EXP
		fairseq-preprocess \
		--source-lang $SRC \
		--target-lang $TRG \
		--trainpref $DATA_ROOT/train.$SRC-$TRG.bpe \
		--validpref $DATA_ROOT/dev.$SRC-$TRG.bpe \
		--testpref $DATA_ROOT/test.$SRC-$TRG.bpe \
		--destdir $DATABIN_ROOT
	fi
	
	mkdir -p $CKPT_ROOT
	echo "Train "$SRC-$TRG.$EXP
	fairseq-train $DATABIN_ROOT \
	-a transformer_iwslt_de_en \
	--optimizer adam \
	--lr 0.0005 \
	-s $SRC \
	-t $TRG \
	--label-smoothing 0.1 \
	--dropout 0.3 \
	--max-tokens 4000 \
	--min-lr '1e-09' \
	--lr-scheduler inverse_sqrt \
	--weight-decay 0.0001 \
	--criterion label_smoothed_cross_entropy \
	--max-update 50000 \
	--warmup-updates 4000 \
	--warmup-init-lr '1e-07' \
	--adam-betas '(0.9, 0.98)' \
	--save-dir $CKPT_ROOT 

	echo "Translate  "$SRC-$TRG.$EXP
	fairseq-generate $DATABIN_ROOT \
	--path $CKPT_ROOT/checkpoint_best.pt \
	--batch-size 32 \
	--beam 12 \
	--remove-bpe
done
