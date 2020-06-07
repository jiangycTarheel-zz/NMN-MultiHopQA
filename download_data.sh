# !/bin/bash
set -eu -o pipefail

mkdir -p raw_data
cd raw_data

### GloVe vectors ###
if [ ! -d glove ]
then
  mkdir glove
  cd glove
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  cd ..
fi

### HotpotQA ###
if [ ! -d hotpotqa ]
then
  mkdir hotpotqa
  cd hotpotqa
  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
  wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
  cd ..
fi
cd ..
