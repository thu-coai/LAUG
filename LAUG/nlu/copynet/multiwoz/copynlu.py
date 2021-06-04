# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:56:56 2020

@author: truthless
"""
import os
import torch

from allennlp.common.checks import check_for_gpu
from allennlp.data import DatasetReader
from allennlp.models.archival import load_archive

from convlab2.nlu.nlu import NLU
from convlab2.nlu.copynet.reader import CopyNetDatasetReader
from convlab2.nlu.copynet.copynet import CopyNet
from convlab2.nlu.copynet.utils import seq2dict
from convlab2.util.file_util import cached_path
DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "copynet_multiwoz_context.tar.gz")

class COPYNLU(NLU):
    """Copying Mechanism natural language understanding."""

    def __init__(self,
                archive_file=DEFAULT_ARCHIVE_FILE,
                cuda_device=DEFAULT_CUDA_DEVICE,
                model_file='https://convlab.blob.core.windows.net/convlab-2/copynet_multiwoz_context.tar.gz',
                context_size=3):
        """ Constructor for NLU class. """

        self.context_size = context_size
        cuda_device = 0 if torch.cuda.is_available() else DEFAULT_CUDA_DEVICE
        check_for_gpu(cuda_device)

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for COPYNLU is specified!")

            archive_file = cached_path(model_file)

        archive = load_archive(archive_file,
                            cuda_device=cuda_device)

        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.model = archive.model
        self.model.eval()


    def predict(self, utterance, context=list()):
        """
        Predict the dialog act of a natural language utterance and apply error model.
        Args:
            utterance (str): A natural language utterance.
        Returns:
            output (dict): The dialog act of utterance.
        """
        if len(utterance) == 0:
            return []

        instance = self.dataset_reader.text_to_instance(utterance)
        outputs = self.model.forward_on_instance(instance)
        for i in range(len(outputs['predictions'])):
            if sum(outputs['predictions'][i] >= self.model._target_vocab_size):
                prediction = list(outputs['predictions'][i])
                break
        else:
            prediction = list(outputs['predictions'][0])
        source_tokens = outputs['metadata']['source_tokens']

        if self.model._end_index in prediction:
            prediction = prediction[: prediction.index(self.model._end_index)]
        prediction = [self.model.vocab.get_token_from_index(index, self.model._target_namespace) if index < self.model._target_vocab_size else source_tokens[
                index - self.model._target_vocab_size] for index in prediction]
        
        da_seq = ' '.join(prediction)
        da_seq = da_seq.replace('i d', 'id')
        output = seq2dict(da_seq)
        tuples = []
        for domain_intent, svs in output.items():
            for slot, value in svs:
                domain, intent = domain_intent.split('-')
                if domain != 'general':
                    domain = domain.capitalize()
                intent = intent.capitalize()
                slot = slot.capitalize()
                tuples.append([intent, domain, slot, value])
        return tuples
    
if __name__ == "__main__":
    nlu = COPYNLU(archive_file="/tmp/models/copynet/run_007/model.tar.gz")
    test_contexts = [
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
        "SENT_END",
    ]
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?",
        "you're welcome! enjoy your visit! goodbye.",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i want to book a table for 60000 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "i will be departing out of qwertyuiop.",
        "What is the Name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for utt in test_utterances:
        print(utt)
        print(nlu.predict(utt))
    
