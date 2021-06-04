# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:19:36 2020

@author: truthless
"""

import json
import sys

if __name__ == '__main__':
    test_file = sys.argv[1]
    output_file = sys.argv[2]

    src, trg = [], []
    with open(test_file, 'r', encoding='utf8') as f:
        for line in f:
            source, target = line.strip("\n").split("\t")
            src.append(source)
            trg.append(target)

    pred = []
    with open(output_file, 'r', encoding='utf8') as f:
        for line in f:
            predict = eval(line)[0].replace('i d', 'id').lower()
            pred.append(predict)

    with open(output_file, 'w', encoding='utf8') as f:
        output = list(zip(src,trg,pred))
        json.dump(output, f)
        