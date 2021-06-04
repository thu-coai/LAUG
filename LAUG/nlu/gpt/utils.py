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
    seq: [ domain { intent ( slot = value ; ) @ } | ]
    dict: [domain-intent: [slot, value]]
    '''
    d = {}
    items = s.split()
    cur = 0 # 0 domain, 1 intent, 2 slot, 3 value
    domain = []
    intent = []
    key = ''
    slot = []
    value = []
    for ch in items:
        if ch in ['}', '|']:
            cur = 0
            domain = []
            key = ''
        elif ch in ['{', ')', '@']:
            cur = 1
            if ch == '{':
                domain = ' '.join(domain)
            elif ch == ')':
                if type(slot) == list:
                    if slot:
                        slot = ' '.join(slot)
                    else:
                        slot = None
                    value = None
                else:
                    value = clear(' '.join(value))
                d[key].append([slot, value])
                slot = []
                value = []
                intent = []
        elif ch in ['(', ';']:
            cur = 2
            if ch == '(':
                intent = ' '.join(intent)
                key = domain + '-' + intent
                d[key] = []
            elif ch == ';':
                if value:
                    value = clear(' '.join(value))
                else:
                    slot = ' '.join(slot)
                    value = None
                d[key].append([slot, value])
                slot = []
                value = []
        elif ch in ['=']:
            cur = 3
            slot = ' '.join(slot)
        else:
            if cur == 0:
                domain.append(ch)
            elif cur == 1:
                intent.append(ch)
            elif cur == 2:
                slot.append(ch)
            elif cur == 3:
                value.append(ch)
    try:
        assert cur == 0
    except:
#        print(s)
        if key:
            flag = False
            if value and type(value) == list:
                value = clear(' '.join(value))
                flag = True
            if slot and type(slot) == list:
                slot = ' '.join(slot)
                if not value:
                    value = None
                flag = True
            if flag:
                d[key].append([slot, value])
            if not d[key]:
                d[key].append([None, None])
#        print(d)
#        print()
    return d

if __name__ == '__main__':
    from convlab2.nlg.scgpt.utils import tuple2dict, dict2seq
    da_tuple = [('Inform', 'Booking', 'none', 'none'), ('Inform', 'Hotel', 'Price', 'cheap'), ('Inform', 'Hotel', 'Choice', '1'), ('Inform', 'Hotel', 'Parking', 'none')]
    da_dict = tuple2dict(da_tuple)
    print(da_dict)
    da_seq = dict2seq(da_dict)
    print(da_seq)
    da_dict = seq2dict(da_seq)
    print(da_dict)

    da_tuple = [('Request', 'Hotel', 'Address', '?'), ('Request', 'Hotel', 'Area', '?'), ('Inform', 'Attraction', 'Area', 'center'), ('Inform', 'Hotel', 'Price', 'cheap')]
    da_dict = tuple2dict(da_tuple)
    print(da_dict)
    da_seq = dict2seq(da_dict)
    print(da_seq)
    da_dict = seq2dict(da_seq)
    print(da_dict)
