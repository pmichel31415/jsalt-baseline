# JSALT Informal text translation baselines

## Prerequisites

- Set up environment
```bash
virtualenv --system-site-packages env
source env/bin/activate
```
- Install moses scripts: `git clone https://github.com/moses-smt/mosesdecoder.git`
- Install subword-nmt: `pip install subword-nmt` (TODO test with sentencepiece instead for en-fr)
- Install sentencepiece:
```bash
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
HERE=`pwd -P`
mkdir bin
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${HERE}/bin
make -j $(nproc)
make install
cd ../..
```
- install fairseq 
```bash
git clone https://github.com/pytorch/fairseq.git
cd fairseq
python setup.py install
cd ..
```
- Install sacrebleu (for BLEU): `pip install sacrebleu`
- Install kytea (for ja evaluation):
```bash
wget http://www.phontron.com/kytea/download/kytea-0.4.7.tar.gz
tar xvzf kytea-0.4.7.tar.gz
cd kytea-0.4.7
mkdir bin
./configure --prefix `pwd -P`/bin
make -j $(nproc)
make install
```

## Data preparation

Change paths in `scripts/globals.sh`. Then run:

```bash
# Prepare WMT15 en-fr (this can take a while and a lot of disk space)
bash scripts/prepare-en-fr.sh
# Prepare KFTT-JESC-TED en-ja (same setting as the MTNT paper)
bash scripts/prepare-en-ja.sh
```

## Training

You can change the settings in either `scripts/train*` files. Just run those:

```bash
# en-fr baseline (takes >1 day per epoch on my machine with a Titan X)
bash scripts/train-en-fr
# ja-en baseline (takes TBD, less than 2 days to converge IIRC)
bash scripts/train-ja-en,sh
```

## Translating

Use these commands to translate MTNT for example:

```bash
# Get MTNT
wget https://github.com/pmichel31415/mtnt/releases/download/v1.1/MTNT.1.1.tar.gz
tar xvzf MTNT.1.1.tar.gz
bash MTNT/split_tsv.sh
# Set environment
source scripts/globals.sh
source env/bin/activate
#### en-fr ####
# BPE segment the en source for en-fr
< MTNT/test/test.en-fr.en | \
    perl -l en $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl | \
    perl -l en $MOSES_SCRIPTS/tokenizer/tokenizer.perl | \
    subword-nmt apply-bpe -c $DATA_ROOT/en-fr/code \
    > mtnt.test.en-fr.bpe.en
# Translate
< mtnt.test.en-fr.bpe.en | \
    fairseq-interactive \
    $DATA_ROOT/en-fr/data-bin \
    --path $CKPT_ROOT/jsalt_baseline_en_fr/checkpoint_best.pt \
    --beam 5 \
    --lenpen 1 \
    --buffer-size 100 \
    --batch-size 32 | \
    grep "^H" | cut -f3 | \
    sed 's/@@ //g' | \
    perl -l en $MOSES_SCRIPTS/tokenizer/detokenizer.perl \
    > mtnt.test.en-fr.out.fr
# BLEU score
< mtnt.test.en-fr.out.fr | sacrebleu -tok intl MTNT/test/test.en-fr.fr
#### ja-en ####
# BPE segment the en source for ja-en
< MTNT/test/test.ja-en.ja | \
    perl -l ja $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl | \
    $SENTENCEPIECE_BINS/spm_encode --model $DATA_ROOT/en-ja/code.model \
    > mtnt.test.ja-en.bpe.en
# Translate
< mtnt.test.ja-en.bpe.ja | \
    fairseq-interactive \
    $DATA_ROOT/en-ja/data-bin \
    --path $CKPT_ROOT/jsalt_baseline_ja_en/checkpoint_best.pt \
    --beam 5 \
    --lenpen 1 \
    --buffer-size 100 \
    --batch-size 32 | \
    grep "^H" | cut -f3 | \
    sed 's/ //g;s/▁/ /g;s/^ //g' | \
    > mtnt.test.ja-en.out.en
# BLEU score
< mtnt.test.ja-en.out.en | sacrebleu -tok intl MTNT/test/test.ja-en.en
```

## TODO

- `fr-en` and `en-ja`
- sentencepiece for `en<->fr`
- scripts for evaluation on MTNT and newstest etc...
