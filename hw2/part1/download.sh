#!/bin/bash

# BCN model
curl "http://140.112.252.117:5000/epoch-5.ckpt" -o ./model/submission/ckpts/epoch-5.ckpt
# ELMo model
wget "https://www.dropbox.com/s/j3ipfpsdm9weo0u/checkpoint_3_39000_60.356972.pt?dl=1" -O ./ELMo/elmo/models/test/checkpoint_3_39000_60.356972.pt
# ELMo embedding
wget "https://www.dropbox.com/s/eun8kxwf568zz96/embedding.npy?dl=1" -O ./ELMo/data/embedding.npy
# id2word
wget "https://www.dropbox.com/s/uftlgdsjc36mbon/idx2word.pkl?dl=1" -O ./ELMo/data/idx2word.pkl
# word2id
wget "https://www.dropbox.com/s/bjat2tv5hfpwzyd/word2idx.pkl?dl=1" -O ./ELMo/data/word2idx.pkl


python -m spacy download en
