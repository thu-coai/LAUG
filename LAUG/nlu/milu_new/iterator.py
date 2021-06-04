from collections import deque
from typing import Iterable, Deque
import logging
import random
import numpy as np
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.dataset import Batch
from allennlp.data.fields import ArrayField,TextField
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DataIterator.register("milu")
class MiluIterator(DataIterator):
    """
    A very basic iterator that takes a dataset, possibly shuffles it, and creates fixed sized batches.
    It takes the same parameters as :class:`allennlp.data.iterators.DataIterator`
    """

    def transfer_instance_list(self,instance_list):
        key_list = ['tokens','context_tokens','tags','slot_tags','intents','slot_intents']
        result = []

        for idx,instance in enumerate(iter(instance_list)):
                tags = instance.fields['tags']['tags']
                slot_tags = instance.fields['slot_tags']
                slot_intent_dim = self.vocab.get_vocab_size('slot_intent_labels')
                transfer_tags = [[0,0] for i in range(slot_intent_dim)]
                mask = [[0]*(2+len(tags)) for i in range(slot_intent_dim)] 

                for i,tag in enumerate(iter(tags)):
                    for k in range(slot_intent_dim):
                        transfer_tags[k].append(self.vocab.get_token_index('O','slot_tags'))
                    if tag != 'O':
                        intent,value = tag.split('+')
                        prefix = intent[0]
                        intent = intent[2:]
                        if intent in self.vocab._token_to_index['slot_intent_labels']:
                            intent_id = self.vocab._token_to_index['slot_intent_labels'][intent]
                            slot_tag = prefix + '-' + value
                            if slot_tag in self.vocab._token_to_index['slot_tags']:
                                slot_tag_id = self.vocab._token_to_index['slot_tags'][slot_tag]
                                transfer_tags[intent_id][-1] = slot_tag_id
                            # else:
                            #     print("slot_tag = {}, 不在词表中".format(slot_tag))
                            #     print("tags = ", tags) 
                            #     print("tokens = ", instance.fields['tokens'])
                            #     print('='*100)
                            mask[intent_id][2:] = [1]*len(tags)

                instance_list[idx].fields['slot_tag_tensor'] = ArrayField(np.array(transfer_tags))
                instance_list[idx].fields['tag_mask_tensor'] = ArrayField(np.array(mask))

        return instance_list


    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        # First break the dataset into memory-sized lists:
        for instance_list in self._memory_sized_lists(instances):
            if shuffle:
                random.shuffle(instance_list)
            self.transfer_instance_list(instance_list) 
            iterator = iter(instance_list)
            excess: Deque[Instance] = deque()
            # Then break each memory-sized list into batches.
            for batch_instances in lazy_groups_of(iterator, self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batch = Batch(possibly_smaller_batches) 
                    yield batch
            if excess:
                yield Batch(excess)
