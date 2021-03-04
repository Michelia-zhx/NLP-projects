import os
import numpy as np
import joblib
import torch
from collections import defaultdict
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import util

def gen_dataset():
    datasets = ["data/train.txt", "data/dev.txt"]
    formsets = ["data/train.iob2", "data/dev.iob2"]
    for num in range(len(datasets)):
        with open(datasets[num], 'r', encoding='utf-8') as file:
            form = open(formsets[num], 'w', encoding='utf-8')
            lines = file.readlines()
            sentence = []
            label = []
            for i in range(0, len(lines), 3):
                sentence = lines[i].strip().split(' ')
                forms = []
                if lines[i+1] != '\n':
                    label = lines[i+1].split('|')
                    for j in range(len(sentence)):
                        itm = sentence[j]
                        num_tag = 0
                        for tag in label:
                            start = int(tag.split(',')[0])
                            end = int(tag.split(',')[1].split(' ')[0])
                            wordtype = tag.split(',')[1].split(' ')[1].split('#')[1].strip()
                            if start == j:
                                itm += "\t"+"B-{0}".format(wordtype)
                                num_tag += 1
                            elif start < j <= end:
                                itm += "\t"+"I-{0}".format(wordtype)
                                num_tag += 1
                        if num_tag < 4:
                            itm += (4-num_tag)*"\tO"+"\n"
                        num_tag = 0
                        form.write(itm)
                    form.write("\n")
                else:
                    for j in range(len(sentence)):
                        itm = sentence[j]
                        itm += 4*"\tO"+"\n"
                        form.write(itm)
                    form.write("\n")


def gen_sentence_tensors(sentence_list, device, data_url):
    vocab = util.load(util.dirname(data_url) + '/vocab.json')
    char_vocab = util.load(util.dirname(data_url) + '/char_vocab.json')

    sentences = list()
    sentence_words = list()
    sentence_word_lengths = list()
    sentence_word_indices = list()

    unk_idx = 1
    for sent in sentence_list:
        # word to word id
        sentence = torch.LongTensor([vocab[word] if word in vocab else unk_idx
                                     for word in sent]).to(device)
        # char of word to char id
        words = list()
        for word in sent:
            words.append([char_vocab[ch] if ch in char_vocab else unk_idx
                          for ch in word])
        # save word lengths
        word_lengths = torch.LongTensor([len(word) for word in words]).to(device)
        # sorting lengths according to length
        word_lengths, word_indices = torch.sort(word_lengths, descending=True)
        # sorting word according word length
        words = np.array(words)[word_indices.cpu().numpy()]
        word_indices = word_indices.to(device)
        words = [torch.LongTensor(word).to(device) for word in words]

        # padding char tensor of words
        words = pad_sequence(words, batch_first=True).to(device)
        # (max_word_len, sent_len)

        sentences.append(sentence)
        sentence_words.append(words)
        sentence_word_lengths.append(word_lengths)
        sentence_word_indices.append(word_indices)

    # record sentence length and padding sentences
    sentence_lengths = [len(sentence) for sentence in sentences]
    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True).to(device)
    # (batch_size, max_sent_len)

    return sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices


class ExhaustiveDataset(Dataset):

    def __init__(self, data_url, device, max_region=10):
        super().__init__()
        self.x, self.y = load_raw_data(data_url)

        categories = set()
        for dic in self.y:
            categories = categories.union(dic.values())
        self.categories = ['NA'] + sorted(categories)
        self.n_tags = len(self.categories)
        self.data_url = data_url
        self.max_region = max_region
        self.device = device

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def collate_func(self, data_list):
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        sentence_list, records_list = zip(*data_list)  # un zip
        max_sent_len = len(sentence_list[0])
        sentence_tensors = gen_sentence_tensors(sentence_list, self.device, self.data_url)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)

        region_labels = list()
        for records, length in zip(records_list, sentence_tensors[1]):
            labels = list()
            for region_size in range(1, self.max_region + 1):
                for start in range(0, max_sent_len - region_size + 1):
                    if start + region_size > length:
                        labels.append(self.n_tags)  # for padding
                    elif (start, start + region_size) in records:
                        labels.append(self.categories.index(records[start, start + region_size]))
                    else:
                        labels.append(0)
            region_labels.append(labels)
        region_labels = torch.LongTensor(region_labels).to(self.device)
        # (batch_size, n_regions)

        return sentence_tensors, region_labels, records_list


def gen_vocab_from_data(data_urls, pretrained_url, binary, update=False, min_count=1):
    # generate vocabulary and embeddings from data file, generated vocab files will be saved in data dir

    if isinstance(data_urls, str):
        data_urls = [data_urls]
    data_dir = os.path.dirname(data_urls[0])
    vocab_url = os.path.join(data_dir, "vocab.json")
    char_vocab_url = os.path.join(data_dir, "char_vocab.json")
    embedding_url = os.path.join(data_dir, "embeddings.npy") if pretrained_url else None

    if (not update) and os.path.exists(vocab_url):
        print("vocab file already exists")
        return embedding_url

    vocab = set()
    char_vocab = set()
    word_counts = defaultdict(int)
    print("generating vocab from", data_urls)
    for data_url in data_urls:
        with open(data_url, 'r', encoding='utf-8') as data_file:
            for row in data_file:
                if row == '\n':
                    continue
                token = row.split()[0]
                word_counts[token] += 1
                if word_counts[token] > min_count:
                    vocab.add(row.split()[0])
                char_vocab = char_vocab.union(row.split()[0])

    # sorting vocab according alphabet order
    vocab = sorted(vocab)
    char_vocab = sorted(char_vocab)

    # generate word embeddings for vocab
    if pretrained_url is not None:
        print("generating pre-trained embedding from", pretrained_url)
        kvs = KeyedVectors.load_word2vec_format(pretrained_url, binary=binary)
        embeddings = list()
        for word in vocab:
            if word in kvs:
                embeddings.append(kvs[word])
            else:
                embeddings.append(np.random.uniform(-0.25, 0.25, kvs.vector_size)),

    char_vocab = ['<pad', '<unk>'] + char_vocab
    vocab = ['<pad>', '<unk>'] + vocab
    util.dump(util.list_to_dict(vocab), vocab_url)
    util.dump(util.list_to_dict(char_vocab), char_vocab_url)

    if pretrained_url is None:
        return
    embeddings = np.vstack([np.zeros(kvs.vector_size),  # for <pad>
                            np.random.uniform(-0.25, 0.25, kvs.vector_size),  # for <unk>
                            embeddings])
    np.save(embedding_url, embeddings)
    return embedding_url

def infer_records(columns):
    records = dict()
    for col in columns:
        start = 0
        while start < len(col):
            end = start + 1
            if col[start][0] == 'B':
                while end < len(col) and col[end][0] == 'I':
                    end += 1
                records[(start, end)] = col[start][2:]
            start = end
    return records


def load_raw_data(data_url, update=False):
    # load from pickle
    save_url = data_url.replace('.iob', '.raw.pkl').replace('.iob2', '.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    records = list()
    with open(data_url, 'r', encoding='utf-8') as file:
        first_line = file.readline()
        print(first_line)
        n_columns = first_line.count('\t')
        # JNLPBA dataset don't contains the extra 'O' column
        if 'jnlpba' in data_url:
            n_columns += 1
        columns = [[x] for x in first_line.split()]
        for line in file:
            if line != '\n':
                line_values = line.split()
                for i in range(n_columns):
                    columns[i].append(line_values[i])

            else:  # end of a sentence
                sentence = columns[0]
                sentences.append(sentence)
                records.append(infer_records(columns[1:]))
                columns = [list() for i in range(n_columns)]
    joblib.dump((sentences, records), save_url)
    return sentences, records


def prepare_vocab(data_urls, pretrained_url, update=True, min_count=1):
    # prepare vocab and embedding
    binary = pretrained_url and pretrained_url.endswith('.bin')
    return gen_vocab_from_data(data_urls, pretrained_url, binary=binary, update=update, min_count=min_count)