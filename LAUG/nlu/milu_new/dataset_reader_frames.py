# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import random
import zipfile
from typing import Dict, List, Any

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MultiLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from overrides import overrides

from LAUG.util.file_util import cached_path
from LAUG.nlu.milu_new.frames.defs import GENERAL_TYPE, REQUESTABLE_TYPE, INFORMABLE_TYPE

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("milu_frames")
class MILUDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    and converts it into a ``Dataset`` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Parameters
    ----------
    word_tag_delimiter: ``str``, optional (default=``"###"``)
        The text that separates each WORD from its TAG.
    token_delimiter: ``str``, optional (default=``None``)
        The text that separates each WORD-TAG pair from the next pair. If ``None``
        then the line will just be split on whitespace.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """
    def __init__(self,
                 context_size: int = 0,
                 agent: str = None,
                 random_context_size: bool = True,
                 token_delimiter: str = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        print('contruct milu dataset reader')
        self._context_size = context_size
        self._agent = agent 
        self._random_context_size = random_context_size
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._token_delimiter = token_delimiter

    @overrides
    def _read(self, file_path):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        if file_path.endswith("zip"):
            print("file_path = ", file_path)
            archive = zipfile.ZipFile(file_path, "r")
            data_file = archive.open(os.path.basename(file_path)[:-4])
        else:
            data_file = open(file_path, "r")

        logger.info("Reading instances from lines in file at: %s", file_path)

        dialogs = json.load(data_file)

        for dial_name in dialogs:
            dialog = dialogs[dial_name]["log"]
            context_tokens_list = []
            for i, turn in enumerate(dialog):
                if self._agent and self._agent == "user" and i % 2 == 1: 
                    context_tokens_list.append(turn["text"].lower().split()+ ["SENT_END"])
                    continue
                if self._agent and self._agent == "system" and i % 2 == 0: 
                    context_tokens_list.append(turn["text"].lower().split()+ ["SENT_END"])
                    continue
                if not turn["dialog_act"]:
                    continue

                tokens = turn["text"].lower().split() 
                slot_intents = []
                intents = []
                dialog_act = {}
                for dacts in turn["span_info"]:#domain-intent ,slot,value,begin_id, end_id
                    intents.append(dacts[0])
                    slot_intents.append(dacts[0])

                    if dacts[0] not in dialog_act:#domain-intent
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4]+1])])

                spans = turn["span_info"]
                tags = []
                slot_tags = []

                for i, _ in enumerate(tokens):
                    for span in spans:
                        if i == span[3]:
                            domain,intent = span[0].split('-')
                            tags.append("B-" + domain + '-' +intent + "+" + span[1]) #修改: B/I-domain-intent-slot
                            slot_tags.append("B-"+span[1])
                            break
                        if span[3] < i <= span[4]:
                            domain,intent = span[0].split('-')
                            tags.append("I-" + domain + '-' +intent + "+" + span[1]) #修改: B/I-domain-intent-slot
                            slot_tags.append("I-"+span[1])
                            break
                    else:
                        tags.append("O")
                        slot_tags.append("O")

                req_intents = []
                req_slots = []
                reqs = []
                # general_intents = []
                # intents = []
                # slot_intents = []
                # intent_cnt = {}
                for dacts in turn["dialog_act"]: # dacts: 'domain-intent': [[name, value], ...]
                    processed_da = []
                    for i, dact in enumerate(turn["dialog_act"][dacts]): # dact: [name, value]
                        temp_domain, temp_intent = dacts.split('-')
                        if temp_intent in GENERAL_TYPE and dacts not in intents:
                            intents.append(dacts)
                            processed_da.append(dact)
                        elif temp_intent in REQUESTABLE_TYPE and dacts not in intents:
                            intents.append(dacts)
                            req_intents.append(dacts)
                            req_slots.append(dact[0])
                            reqs.append(dacts + '-' + dact[0]) # domain - intent - slot
                            processed_da.append(dact)

                        # elif temp_intent == 'Inform':
                        #     slot_intents.append(dacts.split('-')[0]+"-Inform")
                        #     slot = dact[0] 
                        #     temp_key = dacts + '-' + slot
                        #     if temp_key not in intent_cnt.keys(): 
                        #         intent_cnt[temp_key] = 0
                            
                        #     if dacts in dialog_act:
                        #         possible_value = [s[1] for s in dialog_act[dacts] if s[0].lower() == slot.lower()] #相同的slot (不考虑大小写: parking / Parking)
                        #         if len(possible_value): 
                        #             if dacts not in intents: 
                        #                 assert(intent_cnt[temp_key] == 0)
                        #                 intents.append(dacts)

                        #             processed_da.append([slot, possible_value[intent_cnt[temp_key] if intent_cnt[temp_key] < len(possible_value) else -1]]) #按顺序替换　如果不足　就选最后一个)
                        #             intent_cnt[temp_key] += 1 
                    
                    turn['dialog_act'][dacts] = processed_da

                negative_keys = [key for key in turn['dialog_act'] if turn['dialog_act'][key] == [] ]
                [turn['dialog_act'].pop(key) for key in negative_keys]
                # print('da label = ', turn['dialog_act'])
                # print('span label = ', dialog_act)
                turn['dialog_act'].update(dialog_act) #add Informable da
                if turn['dialog_act'].keys() == []:
                    print("text = ", turn['text'])
                    print('overall label = ', turn['dialog_act'])
                    print('='*100)
                # print('overall label = ', turn['dialog_act'])
                # print('='*100)
                num_context = random.randint(0, self._context_size) if self._random_context_size else self._context_size
                if len(context_tokens_list) > 0 and num_context > 0:
                    wrapped_context_tokens = [Token(token) for context_tokens in context_tokens_list[-num_context:] for token in context_tokens]
                else:
                    wrapped_context_tokens = [Token("SENT_END")]
                wrapped_tokens = [Token(token) for token in tokens]
                context_tokens_list.append(tokens + ["SENT_END"])

                yield self.text_to_instance(wrapped_context_tokens, wrapped_tokens, tags, slot_tags, intents, slot_intents,turn['dialog_act'], req_intents, req_slots, reqs)#dialog_act)

    def text_to_instance(self, context_tokens: List[Token], tokens: List[Token], tags: List[str] = None, slot_tags: List[str]=None,
        intents: List[str] = None,slot_intents: List[str] = None, dialog_act: Dict[str, Any] = None, req_intents=None, req_slots=None, reqs=None) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        fields["context_tokens"] = TextField(context_tokens, self._token_indexers)
        fields["tokens"] = TextField(tokens, self._token_indexers)
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = MetadataField({"tags":tags})#SequenceLabelField(tags, fields["tokens"],label_namespace="tags")
            fields["slot_tags"] = SequenceLabelField(slot_tags, fields["tokens"],label_namespace='slot_tags') #需要slot_tags的vocab
        if intents is not None:
            fields["intents"] = MultiLabelField(intents, label_namespace="intent_labels")
            fields["slot_intents"] = MultiLabelField(slot_intents, label_namespace="slot_intent_labels")
            fields['req_intents'] = MultiLabelField(req_intents, label_namespace="req_intent_labels")
        if req_slots is not None:
            fields['reqs'] = MultiLabelField(req_slots,label_namespace='req_slot_labels')
            fields['full_reqs'] = MultiLabelField(reqs, label_namespace='req_full_labels')
        if dialog_act is not None:
            fields["metadata"] = MetadataField({"words": [x.text for x in tokens],
            'dialog_act': dialog_act})
        else:
            fields["metadata"] = MetadataField({"words": [x.text for x in tokens], 'dialog_act': {}})
        
        return Instance(fields)
