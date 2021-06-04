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
            cur_dir)))), 'data/multiwoz/')

    keys = ['train', 'val', 'test']
    data = {}
    for key in keys:
        data_key = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data_key)))
        data = dict(data, **data_key)

    with open(os.path.join(data_dir, 'valListFile'), 'r') as f:
        val_list = f.read().splitlines()
    with open(os.path.join(data_dir, 'testListFile'), 'r') as f:
        test_list = f.read().splitlines()
        
    results = {}
    results_val = {}
    results_test = {}

    for title, sess in data.items():
        logs = sess['log']
        turns = []
        turn = {'turn':0, 'sys':'', 'sys_da':''}
        current_domain = None
        for i, diag in enumerate(logs):
            text = diag['text'].replace('\t', ' ').replace('\n', ' ')
            da = diag['dialog_act']
            span = diag['span_info']
            if i % 2 == 0:
                turn['usr'] = text
                if current_domain:
                    da = eval(str(da).replace('Booking', current_domain))
                    span = eval(str(span).replace('Booking', current_domain))
                turn['usr_da'] = da
                turn['usr_span'] = span
                turns.append(turn)
            else:
                turn = {'turn': i//2 +1}
                turn['sys'] = text
                turn['sys_da'] = da
                turn['sys_span'] = span
            for key in da:
                domain = key.split('-')[0]
                if domain not in ['general', 'Booking']:
                    current_domain = domain
        title = title
        if title in val_list:
            current = results_val
        elif title in test_list:
            current = results_test
        else:
            current = results
        current[title] = turns
        
    results = eval(str(results).replace(" n't", " not"))
    results_val = eval(str(results_val).replace(" n't", " not"))
    results_test = eval(str(results_test).replace(" n't", " not"))


    def write_file(name, data, k):
        with open(f'{name}.tsv', 'w', encoding='utf-8') as f:
            for ID in data:
                sess = data[ID]
                da_uttr_history = []
                for turn in sess:
                    if not turn['usr_da']:
                        continue
                    if turn['sys']:
                        da_uttr_history.append(turn['sys'])
                    spans = turn['usr_span']
                    turn['usr_da'] = eval(str(turn['usr_da']).replace('Bus','Train'))
                    for key in turn['usr_da']:
                        if key.endswith('-Inform'):
                            for pairs in turn['usr_da'][key]:
                                for items in spans:
                                    if items[0] == key and items[1] == pairs[0]:
        #                                if pairs[1] != items[2]:
        #                                    print('before', pairs[1])
        #                                    print('after', items[2])
                                        pairs[1] = items[2]
                                        break
        #                        else:
        #                            print(pairs)
                    da_seq = dict2seq(dict2dict(turn['usr_da']))
                    da_uttr = turn['usr']
                    for i in range(1, k):
                        pad = ' ' if i > 1 else ' $ '
                        if len(da_uttr_history) >= i:
                            da_uttr = da_uttr_history[-i] + pad + da_uttr
                    f.write(f'{da_uttr}\t{da_seq}\n')
                    da_uttr_history.append(turn['usr'])

    k = 3
    if not os.path.exists(os.path.join(cur_dir,'data')):
        os.makedirs(os.path.join(cur_dir, 'data'))
    write_file(os.path.join(cur_dir, 'data/train'), dict(results, **results_val), k)
    write_file(os.path.join(cur_dir, 'data/test'), results_test, k)
