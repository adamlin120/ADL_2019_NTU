import pickle
import argparse
import os
import random
import ipdb
import traceback
import sys
from collections import Counter
from tqdm import tqdm
import numpy as np
from .dataset import ElmoDataset

OOV_INDEX = 0
PAD_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3
OOV_TOKEN = '<OOV>'
PAD_TOKEN = '<PAD>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
NUM_SPECIAL_TOKEN = 4

EMBEDDING_DIM = 300


def main(args):
    threshold = args.threshold
    corpus_path = args.corpus
    elmo_data_dir = args.save_folder
    index_corpus_path = {'train': os.path.join(elmo_data_dir, 'corpus_index_train.pkl'),
                         'val': os.path.join(elmo_data_dir, 'corpus_index_val.pkl')}
    idx2word_path = os.path.join(elmo_data_dir, 'idx2word.pkl')
    word2idx_path = os.path.join(elmo_data_dir, 'word2idx.pkl')
    embedding_path = os.path.join(elmo_data_dir, 'embedding.npy')

    with open(corpus_path, 'r') as f:
        data = f.read().splitlines()
        data = random.sample(data, int(len(data) * args.ratio))

    word2count = Counter()
    for line in tqdm(data, desc="Count frequency"):
        words = line.split()
        word2count.update(words)

    idx2word = dict()
    for idx, token in [(OOV_INDEX, OOV_TOKEN), (PAD_INDEX, PAD_TOKEN),
                       (BOS_INDEX, BOS_TOKEN), (EOS_INDEX, EOS_TOKEN)]:
        idx2word[idx] = token

    word2vec = {}
    n_word = NUM_SPECIAL_TOKEN
    with open(args.glove, 'r', encoding='utf-8') as f:
        for row in tqdm(f, desc="Count number of valid word in Pre-trained Embedding"):
            row = row.split()
            word = ''.join(row[:-300])
            vec = np.asanyarray(row[-300:], dtype=np.float32)
            if word2count[word] >= threshold:
                word2vec[word] = vec
    n_word += len(word2vec)

    embedding = np.zeros((n_word, EMBEDDING_DIM), dtype=np.float32)
    word_idx = NUM_SPECIAL_TOKEN
    for word, count in tqdm(word2count.most_common(), desc="Construct Embedding"):
        if count < threshold:
            break
        try:
            embedding[word_idx, :] = word2vec[word]
            idx2word[word_idx] = word
            word_idx += 1
        except KeyError:
            pass
    assert word_idx == n_word
    assert np.all(embedding[:NUM_SPECIAL_TOKEN] == 0)
    assert np.all(embedding[NUM_SPECIAL_TOKEN:].sum(1) != 0)

    word2idx = dict()
    for i, word in tqdm(idx2word.items(), desc="Construct word2index"):
        word2idx[word] = i

    index_corpus = [[word2idx.get(word, OOV_INDEX) for word in line.split()]
                    for line in tqdm(data, desc="Convert word to index in corpus")]
    index_corpus = [line
                    for line in tqdm(index_corpus, desc="Remove empty sentence") if sum(line) > 0]
    index_corpus = [ElmoDataset.add_bos_eos(line)
                    for line in tqdm(index_corpus, desc="Add BOS EOS")]

    all_sent_ok = False
    while not all_sent_ok:
        all_sent_ok = True
        for i, line in tqdm(enumerate(index_corpus), desc="Breaking sent exceeds max_sent_len"):
            if len(line) > args.max_sent_len:
                all_sent_ok = False
                index_corpus.append(line[args.max_sent_len:])
                index_corpus[i] = line[:args.max_sent_len]

    random.shuffle(index_corpus)
    num_val = int(len(index_corpus) * args.val_ratio)
    index_corpus = {'train': index_corpus[num_val:],
                    'val': index_corpus[:num_val]}

    print(f"Vocabulary Size: {len(idx2word)}")
    print(f"Train corpus Size: {len(index_corpus['train'])}")
    print(f"Val corpus Size: {len(index_corpus['val'])}")

    with open(index_corpus_path['train'], 'wb') as f:
        pickle.dump(index_corpus['train'], f)
    with open(index_corpus_path['val'], 'wb') as f:
        pickle.dump(index_corpus['val'], f)
    with open(idx2word_path, 'wb') as f:
        pickle.dump(idx2word, f)
    with open(word2idx_path, 'wb') as f:
        pickle.dump(word2idx, f)
    np.save(embedding_path, embedding)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', default='./data/language_model/corpus.txt')
    parser.add_argument('-g', '--glove', default='./data/GloVe/glove.840B.300d.txt')
    parser.add_argument('--save_folder', default='./ELMo/data')
    parser.add_argument('--ratio', default=1/6, type=float)
    parser.add_argument('-t', '--threshold', default=3, type=int)
    parser.add_argument('--val_ratio', default=0.2, type=float)
    parser.add_argument('--max_sent_len', default=64, type=int)
    parser.add_argument('--random_seed', default=31, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    random.seed(args.random_seed)
    try:
        main(args)
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
