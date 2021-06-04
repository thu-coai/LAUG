# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:49:23 2020

@author: truthless
"""
from LAUG.nlu.gpt.utils import seq2dict
from LAUG.nlu.milu.dai_f1_measure import DialogActItemF1Measure


def normalize(data):
    string = str(data)
    
    digit2word = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten', '11': 'eleven',
        '12': 'twelve'
    }

    for key, value in digit2word.items():
        string = string.replace(' ' + key + ' ', ' ' + value + ' ')
    return eval(string)


def calculateF1gpt(data):
    data = normalize(data)
    dai_f1_metric = DialogActItemF1Measure()
    
    for item in data:
        predict = seq2dict(item[1].replace('=?','= ?').lower())
        target = seq2dict(item[0].replace(' \'','').split('&')[1])
        dai_f1_metric([predict], [target])
    
    metric = dai_f1_metric.get_metric(True)
    print(metric)
def calculateF1copy(data):
    data = normalize(data)
    dai_f1_metric = DialogActItemF1Measure()
    
    for item in data:
        predict = seq2dict(item[2].replace('i d','id').lower())
        target = seq2dict(item[1])
        dai_f1_metric([predict], [target])
    
    metric = dai_f1_metric.get_metric(True)
    print(metric)

if __name__ == '__main__':
    from sys import argv
    import json
    data=[]
    if argv[2]=='gpt':
        with open(argv[1], 'r', encoding='utf-8') as f:
            data=json.load(f)
        calculateF1gpt(data)
    if argv[2]=='copy':
        with open(argv[1], 'r', encoding='utf-8') as f:
            data=json.load(f)
        calculateF1copy(data)