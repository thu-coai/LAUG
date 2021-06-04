# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:36:48 2020

@author: truthless
"""
import os
import json
import zipfile
from convlab2.nlg.scgpt.utils import dict2dict, dict2seq

def read_zipped_json(filepath, filename):
    print("zip file path = ", filepath)
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

if __name__ == '__main__':

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            cur_dir)))), 'data/frames/')

    keys = ['train', 'val', 'test']
    results = {}
    results_test = {}
    for key in keys:
        data_key = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data_key)))
        if key == 'train' or key == 'val':
            results = dict(results, **data_key)
        else:
            results_test = dict(results_test, **data_key)

    def write_file(name, data, k):
        with open(f'{name}.tsv', 'w', encoding='utf-8') as f:
            for ID in data:
                sess = data[ID]['log']
                da_uttr_history = []
                for turn in sess:
                    if not turn['dialog_act']:
                        da_uttr_history.append(turn['text'].lower())
                        continue
                    da_seq = dict2seq(dict2dict(turn['dialog_act'])).replace('_', ' ').lower()
                    da_uttr = turn['text'].lower()
                    for i in range(1, k):
                        pad = ' ' if i > 1 else ' $ '
                        if len(da_uttr_history) >= i:
                            da_uttr = da_uttr_history[-i] + pad + da_uttr

                    f.write(f'{da_uttr}\t{da_seq}\n')
                    da_uttr_history.append(turn['text'].lower())

    k = 3
    if not os.path.exists(os.path.join(cur_dir,'data')):
        os.makedirs(os.path.join(cur_dir, 'data'))
    write_file(os.path.join(cur_dir, 'data/train'), results, k)
    write_file(os.path.join(cur_dir, 'data/test'), results_test, k)
