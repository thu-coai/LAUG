import os
import zipfile
import json
import torch
from unidecode import unidecode
import spacy
from LAUG.util.file_util import cached_path
from LAUG.nlu.nlu import NLU
from LAUG.nlu.jointBERT_new.dataloader import Dataloader
from LAUG.nlu.jointBERT_new.jointBERT import JointBERT
from LAUG.nlu.jointBERT_new.multiwoz.postprocess import recover_intent
from LAUG.nlu.jointBERT_new.multiwoz.preprocess import preprocess


class BERTNLU(NLU):
    def __init__(self, mode='usr', config_file='multiwoz_new_context.json',
                 model_file='https://convlab.blob.core.windows.net/convlab-2/bert_multiwoz_new_context.zip'):
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        self.mode = mode
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/{}'.format(config_file))
        config = json.load(open(config_file))
        # DEVICE = config['DEVICE']
        DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda:0'
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        if not os.path.exists(os.path.join(data_dir, 'intent_vocab.json')):
            preprocess(mode)

        intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
        tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
        req_vocab = json.load(open(os.path.join(data_dir, 'req_vocab.json')))
        req_slot_vocab = json.load(open(os.path.join(data_dir, 'req_slot_vocab.json')))
        slot_intent_vocab = json.load(open(os.path.join(data_dir,'slot_intent_vocab.json')))
        dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab, req_vocab=req_vocab, req_slot_vocab=req_slot_vocab, slot_intent_vocab=slot_intent_vocab,
                                pretrained_weights=config['model']['pretrained_weights'])

        print('intent num:', len(intent_vocab))
        print('tag num:', len(tag_vocab))

        best_model_path = os.path.join(output_dir, 'pytorch_model.bin')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim, dataloader.req_dim, dataloader)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.use_context = config['model']['context']
        self.dataloader = dataloader
        self.nlp = spacy.load('en_core_web_sm')
        print("BERTNLU loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = [token.text for token in self.nlp(unidecode(utterance)) if token.text.strip()]
        ori_tag_seq = ['O'] * len(ori_word_seq)
        if self.use_context:
            if len(context) > 0 and type(context[0]) is list and len(context[0]) > 1:
                context = [item[1] for item in context]
            context_seq = self.dataloader.tokenizer.encode('[CLS] ' + ' [SEP] '.join(context[-3:]))
        else:
            context_seq = self.dataloader.tokenizer.encode('[CLS]')
        intents = []
        da = {}
        reqs = []

        word_seq, tag_seq, new2ori = self.dataloader.bert_tokenize(ori_word_seq, ori_tag_seq)
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, context_seq,reqs, [[self.req2id[req_intent], self.reqslot2id[req_slot]] for req_intent, req_slot in self.dataloader.req_transfer(reqs)],
                       new2ori, word_seq, [self.dataloader.seq_tag2id(i) for i in tag_seq], self.dataloader.seq_intent2id(intents)]]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to(self.model.device) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, req_tensor, req_mask_tensor, word_mask_tensor, tag_mask_tensor, base_tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        with torch.no_grad():
            slot_logits, intent_logits, req_logits = self.model.forward(word_seq_tensor, word_mask_tensor,
                                                            context_seq_tensor=context_seq_tensor,
                                                            context_mask_tensor=context_mask_tensor)
        _,_,_,das = recover_intent(self.dataloader, intent_logits[0], req_logits[:self.dataloader.req_slot_dim],slot_logits[:self.dataloader.slot_intent_dim], base_tag_mask_tensor[:self.dataloader.slot_intent_dim],
                             batch_data[0][0], batch_data[0][-4])
        dialog_act = []
        for intent, slot, value in das:
            domain, intent = intent.split('-')
            dialog_act.append([intent, domain, slot, value])
        return dialog_act


if __name__ == '__main__':
    text = "I will need you departure and arrival city and time ."
    nlu = BERTNLU(config_file ='multiwoz_usr_context.json',model_file='output/usr_context/bert_multiwoz_usr_context.zip')
    print(nlu.predict(text, context=['', "I ' m looking for a train leaving on tuesday please ."]))
    text = "I don't care about the Price of the restaurant."
    print(nlu.predict(text))
