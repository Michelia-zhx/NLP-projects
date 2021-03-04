import torch
import json
import numpy as np
import random

from os.path import dirname, join, normpath, exists
from os import makedirs
import time

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name='auto'):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

def calc_f1(tp, fp, fn, print_result=True):
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    if print_result:
        print(" precision = %f, recall = %f, micro_f1 = %f\n" % (precision, recall, f1))
    return precision, recall, f1

def load(json_url):
    with open(json_url, "r", encoding="utf-8") as json_file:
        obj = json.load(json_file)
    return obj

def dump(obj, json_url):
    with open(json_url, "w", encoding="utf-8", newline='\n') as json_file:
        json.dump(obj, json_file, separators=[',', ': '], indent=4, ensure_ascii=False)


def sort_dict_by_value(dic, reverse=False):
    return dict(sorted(dic.items(), key=lambda x: x[1], reverse=reverse))


def list_to_dict(lis):
    dic = dict()
    for ind, value in enumerate(lis):
        dic[value] = ind
    return dic

def dirname(p):
    """Returns the directory component of a pathname"""
    return p.split('/')[0]