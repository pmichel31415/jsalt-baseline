#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MOSES_SCRIPTS="${DIR}/../mosedecoder/scripts"
SENTENCEPIECE_BINS="${DIR}/../sentencepiece/bin"
KYTEA_BINS="${DIR}/../kytea/bin"
# Change the following
DATA_ROOT="/projects/tir3/users/pmichel1/data"
CKPT_ROOT="/projects/tir3/users/pmichel1/checkpoints"