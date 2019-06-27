#!/usr/bin/env bash

# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

# Global config
source scripts/globals.sh
source env/bin/activate

# Preprocessing commands
CLEAN=$MOSES_SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl
SPM_TRAIN=$SENTENCEPIECE_BINS/spm_train
SPM_ENCODE=$SENTENCEPIECE_BINS/spm_encode

BPE_TOKENS=32000

URLS=(
    "http://statmt.org/wmt13/training-parallel-europarl-v7.tgz"
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://statmt.org/wmt13/training-parallel-un.tgz"
    "http://statmt.org/wmt15/training-parallel-nc-v10.tgz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://statmt.org/wmt15/test.tgz"
)
FILES=(
    "training-parallel-europarl-v7.tgz"
    "training-parallel-commoncrawl.tgz"
    "training-parallel-un.tgz"
    "training-parallel-nc-v10.tgz"
    "training-giga-fren.tar"
    "test.tgz"
)
CORPORA=(
    "training/europarl-v7.fr-en"
    "commoncrawl.fr-en"
    "un/undoc.2000.fr-en"
    "news-commentary-v10.fr-en"
    "giga-fren.release2.fixed"
)

if [ ! -d "$MOSES_SCRIPTS" && ! -L "$MOSES_SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=fr
lang=en-fr
prep=wmt15_en_fr
tmp=$prep/tmp
orig=orig

cd $DATA_ROOT

mkdir -p $orig $tmp $prep

cd $orig

for ((i=0;i<${#URLS[@]};++i));
do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done

gunzip giga-fren.release2.fixed.*.gz
cd ..

echo "pre-processing train data..."
for l in $src $tgt;
do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt;
do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test/newsdiscusstest2015-enfr-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt;
do
    awk '{if (NR%1333 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%1333 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

# TODO: use sentencepiece instead and do aeay with tokenization
BPE_CODE=$prep/code
echo "Training BPE..."
$SPM_TRAIN --model_prefix=$BPE_CODE --vocab_size=$BPE_TOKENS --model_type=bpe --input=$tmp/train.en,$tmp/train.fr

for L in $src $tgt;
do
    for f in train.$L valid.$L test.$L;
    do
        echo "Applying BPE to ${f}..."
        $SPM_ENCODE --model=${BPE_CODE}.model --output_format=piece < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt;
do
    cp $tmp/bpe.test.$L $prep/test.$L
done
