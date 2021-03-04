import sys
import os
import torch
import json
import numpy
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

import util
from dataset import gen_dataset
from dataset import prepare_vocab
from dataset import ExhaustiveDataset
from model import ExhaustiveModel
from eval import evaluate, predict_on_iob2

RANDOM_SEED = 233
util.set_random_seed(RANDOM_SEED)

EMBD_URL = "data/embedding/PubMed-shuffle-win-30.bin"
VOCAB_URL = "data/vocab.json"
TRAIN_URL = "data/train.iob2"
DEV_URL = "data/dev.iob2"
TEST_URL = "data/test.iob2"

LOG_PER_BATCH = 20

def train(n_epochs=1, embedding_url=None, char_feat_dim=50, freeze=False, train_url=TRAIN_URL, dev_url=DEV_URL,
          test_url=None, max_region=10, learning_rate=0.001, batch_size=100, early_stop=5, clip_norm=5,
          device='auto', save_only_best = True):
    device = util.get_device(device)

    train_set = ExhaustiveDataset(train_url, device=device, max_region=max_region)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                              collate_fn=train_set.collate_func)

    vocab = util.load(VOCAB_URL)
    n_words = len(vocab)
    char_vocab = util.load(VOCAB_URL.replace('vocab', 'char_vocab'))
    n_chars = len(char_vocab)

    model = ExhaustiveModel(hidden_size=200, n_tags=train_set.n_tags + 1, char_feat_dim=char_feat_dim,
                            embedding_url=embedding_url, bidirectional=True, max_region=max_region,
                            n_embeddings=n_words, n_chars = n_chars, embedding_dim=200, freeze=freeze)

    model = model.to(device)

    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    max_f1, max_f1_epoch, cnt = -1, 0, 0
    # ignore the padding part when calcuting loss
    tag_weights = torch.Tensor([1] * train_set.n_tags + [0]).to(device)
    best_model_url = None

    # train and evaluate model
    for epoch in range(n_epochs):
        # switch to train mode
        model.train()
        batch_id = 0
        for data, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model.forward(*data)
            # use weight parameter to skip padding part
            loss = criterion(outputs, labels, weight=tag_weights)
            loss.backward()
            # gradient clipping
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()

            endl = '\n' if batch_id % LOG_PER_BATCH == 0 else '\r'
            sys.stdout.write("epoch #%d, batch #%d, loss: %.6f, %s%s" %
                             (epoch, batch_id, loss.item(), datetime.now().strftime("%X"), endl))
            sys.stdout.flush()
            batch_id += 1

        cnt += 1
        # metrics on development set
        dev_metrics = evaluate(model, dev_url)
        if dev_metrics['f1'] > max_f1:
            max_f1 = dev_metrics['f1']
            max_f1_epoch = epoch
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            best_model_url = "data/model/exhaustive_model_epoch%d_%f.pt" % (epoch, max_f1)
            torch.save(model, best_model_url)
            cnt = 0

        print("maximum of f1 value: %.6f, in epoch #%d\n" % (max_f1, max_f1_epoch))
        if cnt >= early_stop > 0:
            break
    print('\n')

    if test_url and best_model_url:
        model = torch.load(best_model_url)
        print("best model url:", best_model_url)
        print("evaluating on test dataset:", test_url)
        test_url = "data/genia.test.iob2"
        predict_on_iob2(model, test_url)

def main():
    start_time = datetime.now()
    gen_dataset()
    embedding_url = prepare_vocab([TRAIN_URL, DEV_URL, TEST_URL],
                                   EMBD_URL, update=False, min_count=0)
    train(test_url=TEST_URL, embedding_url=embedding_url)
    print("finished in:", datetime.now() - start_time)
    pass

if __name__ == '__main__':
    main()