#!/bin/sh

####
#Abdul Rafae Khan#
#CUNY#
#JSALT INFORMAL MT#
####

source env/bin/activate

# suffix of source language files
SRC=ar

# suffix of target language files
TRG=en

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=16000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/path/to/mosesdecoder

if [ ! -d "$mosesdecoder" ]; then
    echo "Please set mosesdecoder variable correctly to point to Moses directory."
    exit
fi

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/path/to/subword-nmt 

if [ ! -d "$subword_nmt" ]; then
    echo "Please set subword_nmt variable correctly to point to subword-nmt directory."
    exit
fi

base_path=/path/to/base/directory
if [ ! -d "$base_path" ]; then
    echo "Please set subword_nmt variable correctly to point to subword-nmt directory."
    exit
fi

for EXP in actual-1k literal-1k actual-2k literal-2k
 do
   echo $SRC-$TRG/$EXP/
   #
   data_in=data/extracted/$SRC-$TRG/$EXP/ #TO DO: change
   if [ ! -d "$data_in" ]; then
		echo "Please set data_in variable correctly to point to input data directory."
		exit
   fi
   
   data_out=data/processed/$SRC-$TRG/$EXP/
   model_dir=models/$SRC-$TRG/$EXP/
   
   mkdir -p $data_out $model_dir
   
   train_file=train.$SRC-$TRG
   dev_file=dev.$SRC-$TRG
   test_file=test.$SRC-$TRG
   
   # tokenize
   for prefix in $train_file $dev_file $test_file
    do
      cat $data_in/$prefix.$SRC | \
      $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
      $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > $data_out/$prefix.tok.$SRC
   
      cat $data_in/$prefix.$TRG | \
      $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
      $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > $data_out/$prefix.tok.$TRG
   
    done
   
   # clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
   $mosesdecoder/scripts/training/clean-corpus-n.perl $data_out/$train_file.tok $SRC $TRG $data_out/$train_file.tok.clean 1 80
   
   # train truecaser
   $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $data_out/$train_file.tok.clean.$SRC -model $model_dir/truecase-model.$SRC
   $mosesdecoder/scripts/recaser/train-truecaser.perl -corpus $data_out/$train_file.tok.clean.$TRG -model $model_dir/truecase-model.$TRG
   
   # apply truecaser (cleaned training corpus)
   for prefix in $train_file
    do
     $mosesdecoder/scripts/recaser/truecase.perl -model $model_dir/truecase-model.$SRC < $data_out/$prefix.tok.clean.$SRC > $data_out/$prefix.tc.$SRC
     $mosesdecoder/scripts/recaser/truecase.perl -model $model_dir/truecase-model.$TRG < $data_out/$prefix.tok.clean.$TRG > $data_out/$prefix.tc.$TRG
    done
   
   # apply truecaser (dev/test files)
   for prefix in $dev_file $test_file
    do
     $mosesdecoder/scripts/recaser/truecase.perl -model $model_dir/truecase-model.$SRC < $data_out/$prefix.tok.$SRC > $data_out/$prefix.tc.$SRC
     $mosesdecoder/scripts/recaser/truecase.perl -model $model_dir/truecase-model.$TRG < $data_out/$prefix.tok.$TRG > $data_out/$prefix.tc.$TRG
    done
   
   # train BPE
   cat $data_out/$train_file.tc.$SRC $data_out/$train_file.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > $model_dir/$SRC$TRG.bpe
   
   # apply BPE
   
   for prefix in $train_file $dev_file $test_file
    do
     $subword_nmt/apply_bpe.py -c $model_dir/$SRC$TRG.bpe < $data_out/$prefix.tc.$SRC > $data_out/$prefix.bpe.$SRC
     $subword_nmt/apply_bpe.py -c $model_dir/$SRC$TRG.bpe < $data_out/$prefix.tc.$TRG > $data_out/$prefix.bpe.$TRG
    done
 done
