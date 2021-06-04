import torch
import os
import zipfile

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from convlab2.nlu.gpt.utils import seq2dict
from convlab2.nlu.gpt.decode import set_seed, sample_sequence
from convlab2.nlu.nlu import NLU
from convlab2.util.file_util import cached_path

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "gptnlu_multiwoz_context.zip")

class GPTNLU(NLU):
    
    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 use_cuda=True,
                 context_size=3,
                 model_file='https://convlab.blob.core.windows.net/convlab-2/gptnlu_multiwoz_context.zip'):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isfile(archive_file):
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
        
        self.model_name_or_path = os.path.join(model_dir, 'multiwoz_nlu')
        self.length = 50
        self.num_samples = 1
        self.temperature = 1.0
        self.repetition_penalty = 1.0
        self.top_k = 0
        self.top_p = 0.9
        self.seed = 42
        self.stop_token = '<|endoftext|>'
        self.context_size = context_size
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.seed, torch.cuda.device_count())
    
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        self.tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path)
        self.model = model_class.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
    
        if self.length < 0 and self.model.config.max_position_embeddings > 0:
            self.length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < self.length:
            self.length = self.model.config.max_position_embeddings  # No generation bigger than model size 
        elif self.length < 0:
            self.length = self.MAX_LENGTH  # avoid infinite loop
               
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
        
        raw_text = utterance
        for i in range(1, self.context_size):
            pad = ' ' if i > 1 else ' $ '
            if len(context) >= i:
                raw_text = context[-i] + pad + raw_text

        context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)
        out = sample_sequence(
            model=self.model,
            context=context_tokens,
            num_samples=self.num_samples,
            length=self.length,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            device=self.device,
        )
        o = out[0, len(context_tokens):].tolist()
        text = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
        text = text.split('& ')[-1]
        text = text[: text.find(self.stop_token) if self.stop_token else None]
        text = text.replace('=?', '= ?')
        output = seq2dict(text)
    
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
