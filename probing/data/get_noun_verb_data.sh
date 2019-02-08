#!/bin/bash

TARGET_DIR=$1

THIS_DIR=$(realpath $(dirname $0))

set -e
if [ ! -d $TARGET_DIR ]; then
  mkdir $TARGET_DIR
fi

function fetch_data() {
  mkdir -p $TARGET_DIR/raw
  pushd $TARGET_DIR/raw

  git clone https://github.com/google-research-datasets/noun-verb.git

  popd
}

fetch_data

# Convert Noun-Verb (sparse POS targets) to edge probing JSON format.
for split in "train" "dev" "test"; do
    python $THIS_DIR/convert-noun-verb.py \
        -i "${TARGET_DIR}/raw/noun-verb/${split}.conll" \
        -o "${TARGET_DIR}/${split}.json"
done

# Print dataset stats for sanity-check.
python ${THIS_DIR%jiant*}/jiant/probing/edge_data_stats.py -i $TARGET_DIR/*.json


