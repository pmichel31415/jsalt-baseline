#!/bin/bash
#SBATCH -n 4
#SBATCH -t 0
#SBATCH --mem 10g
#SBATCH -J EN_JA_DOWNLOAD
#SBATCH -o logs/log_ja_download.txt

# Command line arguments
HERE=`pwd`
SCRIPTS=$HERE/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
KYTEA=kytea
BPEROOT=$HERE/subword-nmt
BPE_TOKENS=32000

# File name
root_dir="/projects/tir3/users/pmichel1/data/en-ja"
TRAIN_FILE="${root_dir}/train"
DEV_FILE="${root_dir}/valid"
TEST_FILE="${root_dir}/test"
tmp="${root_dir}/temp"

# Create corpus dir
mkdir -p $root_dir

cd $root_dir

mkdir -p $root_dir $tmp

# ================== Download data ============================================

# Download JESC
wget -nv -O ${root_dir}/jesc.tar.gz https://goo.gl/idaoxo
# Download KFTT
wget -nv -O ${root_dir}/kftt.tar.gz http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz
# Download TED
wget -nv -O ${root_dir}/ted.en-ja.tgz https://wit3.fbk.eu/archive/2017-01-trnted//texts/en/ja/en-ja.tgz

# ================== Extract data =============================================

# Visualize file structures
tar -tf ${root_dir}/ted.en-ja.tgz
tar -tf ${root_dir}/jesc.tar.gz
tar -tf ${root_dir}/kftt.tar.gz

# Extract to files
for lang in 'en' 'ja';
do
    # Extract training files
    tar -zxvf ${root_dir}/ted.en-ja.tgz -C ${root_dir} en-ja/train.tags.en-ja.$lang
    tar -zxvf ${root_dir}/jesc.tar.gz -C ${root_dir} detokenized/train.$lang
    tar -zxvf ${root_dir}/kftt.tar.gz -C ${root_dir} kftt-data-1.0/data/orig/kyoto-train.$lang

    # Extract dev files
    tar -zxvf ${root_dir}/ted.en-ja.tgz -C ${root_dir} en-ja/IWSLT17.TED.tst2014.en-ja.${lang}.xml
    tar -zxvf ${root_dir}/jesc.tar.gz -C ${root_dir} detokenized/val.$lang
    tar -zxvf ${root_dir}/kftt.tar.gz -C ${root_dir} kftt-data-1.0/data/orig/kyoto-dev.$lang
    
    # Extract test files
    tar -zxvf ${root_dir}/ted.en-ja.tgz -C ${root_dir} en-ja/IWSLT17.TED.tst2015.en-ja.${lang}.xml
    tar -zxvf ${root_dir}/jesc.tar.gz -C ${root_dir} detokenized/test.$lang
    tar -zxvf ${root_dir}/kftt.tar.gz -C ${root_dir} kftt-data-1.0/data/orig/kyoto-test.$lang
done

# ================== Prepare bilingual data ===================================

# Create files
for lang in 'en' 'ja';
do
    # Remove XML tags
    sed '/^\s*</d' ${root_dir}/en-ja/train.tags.en-ja.${lang} | sed -e 's/^\s*//g' | sed -e 's/\s*$//g' > ${root_dir}/en-ja/train.$lang 
    sed '/<seg/!d' ${root_dir}/en-ja/IWSLT17.TED.tst2014.en-ja.${lang}.xml | sed -e 's/\s*<[^>]*>\s*//g' > ${root_dir}/en-ja/dev.$lang 
    sed '/<seg/!d' ${root_dir}/en-ja/IWSLT17.TED.tst2015.en-ja.${lang}.xml | sed -e 's/\s*<[^>]*>\s*//g' > ${root_dir}/en-ja/test.$lang 

    # De-segment (remove ascii spaces)
    if [ $lang = 'ja' ]; then
        sed -i 's/ //' ${root_dir}/en-ja/train.$lang
        sed -i 's/ //' ${root_dir}/detokenized/train.$lang
        sed -i 's/ //' ${root_dir}/en-ja/dev.$lang
        sed -i 's/ //' ${root_dir}/detokenized/val.$lang
        sed -i 's/ //' ${root_dir}/en-ja/test.$lang
        sed -i 's/ //' ${root_dir}/detokenized/test.$lang
    fi
    # Concatenate to training file
    cat ${root_dir}/en-ja/train.$lang ${root_dir}/detokenized/train.$lang ${root_dir}/kftt-data-1.0/data/orig/kyoto-train.$lang > ${TRAIN_FILE}.$lang

    # Concatenate to dev file
    cat ${root_dir}/en-ja/dev.$lang ${root_dir}/detokenized/val.$lang ${root_dir}/kftt-data-1.0/data/orig/kyoto-dev.$lang > ${DEV_FILE}.$lang

    # Concatenate to test file
    cat ${root_dir}/en-ja/test.$lang ${root_dir}/detokenized/test.$lang ${root_dir}/kftt-data-1.0/data/orig/kyoto-test.$lang > ${TEST_FILE}.$lang
done

# Tokenize
echo "Normalizing"
for L in en ja;
do
    for split in "train" "valid" "test"
    do
        cat ${root_dir}/${split}.$L | \
            perl $NORM_PUNC $L | \
            perl $REM_NON_PRINT_CHAR > $tmp/${split}.$L
    done
done

# BPE (with sentencepiece)
BPE_CODE=$root_dir/code
if [ ! -f ${BPE_CODE}.model ]
then
    echo "Training BPE"
    spm_train --model_prefix=$BPE_CODE --vocab_size=$BPE_TOKENS --character_coverage=0.9995 --model_type=bpe --input=$tmp/train.en,$tmp/train.ja
fi

for L in en ja;
do
    for f in train.$L valid.$L test.$L;
    do
        echo "Subword segmenting ${f}..."
        spm_encode --model=${BPE_CODE}.model --output_format=piece < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 2 $tmp/bpe.train ja en $root_dir/train 1 250
perl $CLEAN -ratio 2 $tmp/bpe.valid ja en $root_dir/valid 1 250

for L in $src $tgt;
do
    cp $tmp/bpe.test.$L $root_dir/test.$L
done

# ================== Cleanup ==================================================

# Delete residual files and folders
rm ${root_dir}/*.{tar.gz,tgz}

