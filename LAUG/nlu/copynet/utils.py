# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:34:55 2020

@author: truthless
"""
def clear(s):
    s = s.replace("'", "")
    return s

def seq2dict(s):
    '''
    seq: [domain { intent ( slot = value ; ) @ } | ]
    dict: [domain-intent: [slot, value]]
    '''
    d = {}
    items = s.split()
    cur = 0 # 0 domain, 1 intent, 2 slot, 3 value
    dom = ''
    key = ''
    pair = []
    for ch in items:
        if ch in ['}', '|']:
            cur = 0
        elif ch in ['{', ')', '@']:
            cur = 1
        elif ch in ['(', ';']:
            cur = 2
        elif ch in ['=']:
            cur = 3
        else:
            if cur == 0:
                dom = ch
            elif cur == 1:
                key = dom + '-' + ch
                if key not in d:
                    d[key] = []
            elif cur == 2:
                pair = [ch]
            elif cur == 3 and pair:
                pair.append(ch)
        if ch in [';', ')'] and pair and key in d:
            d[key].append([pair[0], clear(''.join(pair[1:]))])
            pair = []
    try:
        assert cur == 0
    except:
        print('Informal sequence!')
        if pair:
            d[key].append([pair[0], clear(''.join(pair[1:]))])
    for key in d:
        if not d[key]:
            d[key].append(['none', 'none'])
    return d
