# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from math import log10
from typing import Dict, Optional, List, Any

import allennlp.nn.util as util
import numpy as np
import torch
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from allennlp.models.model import Model
from allennlp.modules import Attention, ConditionalRandomField, FeedForward
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder,TokenEmbedder,Embedding
from allennlp.modules.attention import LegacyAttention
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure
from overrides import overrides
from torch.nn.modules.linear import Linear
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenCharactersIndexer

from LAUG.nlu.milu_new.dai_f1_measure import DialogActItemF1Measure
from LAUG.nlu.milu_new.multilabel_f1_measure import MultiLabelF1Measure

from LAUG.nlu.milu_new.frames.defs import GENERAL_TYPE, REQUESTABLE_TYPE, INFORMABLE_TYPE

@Model.register("milu_frames")
class MILU(Model):
    """
    The ``MILU`` encodes a sequence of text with a ``Seq2SeqEncoder``,
    then performs multi-label classification for closed-class dialog act items and 
    sequence labeling to predict a tag for each token in the sequence.

    Parameters
    ----------
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 intent_encoder: Seq2SeqEncoder = None,
                 req_encoder: Seq2SeqEncoder = None,
                 tag_encoder: Seq2SeqEncoder = None,
                 attention: Attention = None,
                 attention_function: SimilarityFunction = None,
                 context_for_intent: bool = True,
                 context_for_req: bool = True,
                 context_for_tag: bool = True,
                 attention_for_intent: bool = True,
                 attention_for_req: bool = True,
                 attention_for_tag: bool = True,
                 sequence_label_namespace: str = "tags",
                 slot_sequence_label_namespace:str = "slot_tags",
                 intent_label_namespace: str = "intent_labels",
                 slot_intent_label_namespace: str = "slot_intent_labels",
                 req_intent_label_namespace: str = "req_intent_labels",
                 req_slot_label_namespace: str = "req_slot_labels",
                 req_full_label_namespace: str = "req_full_labels",
                 feedforward: Optional[FeedForward] = None,
                 label_encoding: Optional[str] = None,
                 include_start_end_transitions: bool = True,
                 crf_decoding: bool = False,
                 constrain_crf_decoding: bool = None,
                 focal_loss_gamma: float = None,
                 nongeneral_intent_weight: float = 5.,
                 nongeneral_req_weight: float = 5.,
                 num_train_examples: float = None,
                 calculate_span_f1: bool = None,
                 dropout: Optional[float] = None,
                 verbose_metrics: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        print('vocab = ',vocab)
        print('contruct MILU model!')
        self.context_for_intent = context_for_intent
        self.context_for_req = context_for_req
        self.context_for_tag = context_for_tag
        self.attention_for_intent = attention_for_intent
        self.attention_for_req = attention_for_req
        self.attention_for_tag = attention_for_tag
        self.sequence_label_namespace = sequence_label_namespace
        self.slot_sequence_label_namesapce = slot_sequence_label_namespace
        self.intent_label_namespace = intent_label_namespace
        self.req_intent_label_namespace = req_intent_label_namespace
        self.req_slot_label_namespace = req_slot_label_namespace
        self.req_full_label_namespace = req_full_label_namespace
        # self.req_label_namespace = req_label_namespace
        self.slot_intent_label_namespace = slot_intent_label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(sequence_label_namespace)
        self.num_slot_tags = self.vocab.get_vocab_size(slot_sequence_label_namespace)
        self.num_intents = self.vocab.get_vocab_size(intent_label_namespace)
        self.num_req_intents = self.vocab.get_vocab_size(req_intent_label_namespace)
        self.num_req_slots = self.vocab.get_vocab_size(req_slot_label_namespace)
        self.num_req_full = self.vocab.get_vocab_size(req_full_label_namespace)
        self.num_slot_intents = self.vocab.get_vocab_size(slot_intent_label_namespace)
        self.encoder = encoder
        self.intent_encoder = intent_encoder
        self.req_encoder = intent_encoder
        self.tag_encoder = intent_encoder
        self._feedforward = feedforward
        self._verbose_metrics = verbose_metrics
        self.rl = False 


        myIndexer = TokenCharactersIndexer(min_padding_length=3)
        self.prefix_token_list = []
        self.prefix_token_character_list = []
        max_len = -1#for padding
        for i in range(self.num_slot_intents):
            pair = self.vocab.get_token_from_index(i,self.slot_intent_label_namespace)
            domain,intent =pair.split('-')
            domain_id = self.vocab.get_token_index(domain.lower(),"tokens") #vocab都是小写的 要转lower
            intent_id = self.vocab.get_token_index(intent.lower(),"tokens")
            characters_list= myIndexer.tokens_to_indices([Token(domain),Token(intent)],self.vocab,"token_characters")["token_characters"]
            max_len = max(max_len, len(characters_list[0]))
            max_len = max(max_len, len(characters_list[1]))
            self.prefix_token_list.append([domain_id,intent_id])
            self.prefix_token_character_list.append(characters_list)
        #padding
        for i in self.prefix_token_character_list:
            i[0] += [0] * (max_len - len(i[0]))
            i[1] += [0] * (max_len - len(i[1]))

        self.prefix_token_list = torch.LongTensor(self.prefix_token_list)
        self.prefix_token_character_list = torch.LongTensor(self.prefix_token_character_list)

        # req intent prefix
        self.req_prefix_token_list = []
        self.req_prefix_token_character_list = []
        max_len = -1 #for padding
        for i in range(self.num_req_intents):
            pair = self.vocab.get_token_from_index(i,self.req_intent_label_namespace)
            domain,intent =pair.split('-')
            domain_id = self.vocab.get_token_index(domain.lower(),"tokens") #vocab都是小写的 要转lower
            intent_id = self.vocab.get_token_index(intent.lower(),"tokens")
            characters_list= myIndexer.tokens_to_indices([Token(domain),Token(intent)],self.vocab,"token_characters")["token_characters"]
            max_len = max(max_len, len(characters_list[0]))
            max_len = max(max_len, len(characters_list[1]))
            self.req_prefix_token_list.append([domain_id,intent_id])
            self.req_prefix_token_character_list.append(characters_list)
        #padding
        for i in self.req_prefix_token_character_list:
            i[0] += [0] * (max_len - len(i[0]))
            i[1] += [0] * (max_len - len(i[1]))

        self.req_prefix_token_list = torch.LongTensor(self.req_prefix_token_list)
        self.req_prefix_token_character_list = torch.LongTensor(self.req_prefix_token_character_list)

        if attention:
            if attention_function:
                raise ConfigurationError("You can only specify an attention module or an "
                                         "attention function, but not both.")
            self.attention = attention
        elif attention_function:
            self.attention = LegacyAttention(attention_function)

        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        projection_input_dim = feedforward.get_output_dim() if self._feedforward else self.encoder.get_output_dim()
        if self.context_for_intent:
            projection_input_dim += self.encoder.get_output_dim()

        if self.attention_for_intent:
            projection_input_dim += self.encoder.get_output_dim()
        
        self.intent_projection_layer = Linear(projection_input_dim, self.num_intents)
        self.req_projection_layer = Linear(projection_input_dim,self.num_req_slots)

        if num_train_examples:
            try:
                pos_weight = torch.tensor([log10((num_train_examples - self.vocab._retained_counter[intent_label_namespace][t]) / 
                                self.vocab._retained_counter[intent_label_namespace][t]) for i, t in 
                                self.vocab.get_index_to_token_vocabulary(intent_label_namespace).items()])
            except:
                pos_weight = torch.tensor([1. for i, t in 
                                self.vocab.get_index_to_token_vocabulary(intent_label_namespace).items()])
        else:
            pos_weight = torch.tensor([(lambda t: nongeneral_intent_weight if t.split('-')[1] in ['request','affirm','negate','request_alts','request_compare'] else 1.)(t) for i, t in 
                            self.vocab.get_index_to_token_vocabulary(intent_label_namespace).items()])

        req_pos_weight = torch.tensor([1. for i, t in 
                            self.vocab.get_index_to_token_vocabulary(req_slot_label_namespace).items()]) 

        self.intent_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.req_loss = torch.nn.BCEWithLogitsLoss(pos_weight=req_pos_weight, reduction="none")

        tag_projection_input_dim = feedforward.get_output_dim() if self._feedforward else self.encoder.get_output_dim()
        if self.context_for_tag:
            tag_projection_input_dim += self.encoder.get_output_dim()
        if self.attention_for_tag:
            tag_projection_input_dim += self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(tag_projection_input_dim,
                                                           self.num_slot_tags))

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError("constrain_crf_decoding is True, but "
                                         "no label_encoding was specified.") 
            labels = self.vocab.get_index_to_token_vocabulary(slot_sequence_label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        if crf_decoding:
            self.crf = ConditionalRandomField(
                    self.num_slot_tags, constraints,
                    include_start_end_transitions=include_start_end_transitions
            )
        else:
            self.crf = None


        self._intent_f1_metric = MultiLabelF1Measure(vocab,
                                                namespace=intent_label_namespace)
        self._req_f1_metric = MultiLabelF1Measure(vocab, namespace=req_slot_label_namespace)
        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError("calculate_span_f1 is True, but "
                                          "no label_encoding was specified.")
            self._f1_metric = SpanBasedF1Measure(vocab,
                                                 tag_namespace=slot_sequence_label_namespace,
                                                 label_encoding=label_encoding)
        self._dai_f1_metric = DialogActItemF1Measure()

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")
        if feedforward is not None:
            check_dimensions_match(encoder.get_output_dim(), feedforward.get_input_dim(),
                                   "encoder output dim", "feedforward input dim")
        initializer(self)

    def get_tag_encoded_text(self, word_seq_tensor,word_mask_tensor):
        batch_size = word_seq_tensor.shape[0]
        max_len = word_seq_tensor.shape[1]
        prefix_token = self.prefix_token_list.repeat(batch_size,1).cuda()
        prefix_token_character = self.prefix_token_character_list.repeat(batch_size,1,1).cuda()
        prefix = {"tokens":prefix_token,"token_characters":prefix_token_character}
        intent_prefix = self.text_field_embedder(prefix).cuda()
        mask_prefix = torch.ones((intent_prefix.shape[0],intent_prefix.shape[1]), dtype=torch.long).cuda()

        repeat_word_seq_tensor = word_seq_tensor.unsqueeze(1).repeat(1,self.num_slot_intents,1,1).view(batch_size*self.num_slot_intents,max_len,-1)
        repeat_word_mask_tensor = word_mask_tensor.unsqueeze(1).repeat(1,self.num_slot_intents,1).view(-1,max_len)
        slot_word_seq_tensor = torch.cat((intent_prefix, repeat_word_seq_tensor),1)
        slot_word_mask_tensor = torch.cat((mask_prefix, repeat_word_mask_tensor),1)

        encoded_text = self.encoder(slot_word_seq_tensor, slot_word_mask_tensor)
        tag_encoded_text = self.tag_encoder(encoded_text, slot_word_mask_tensor) if self.tag_encoder else encoded_text 
        if self.dropout and self.tag_encoder:
            tag_encoded_text = self.dropout(tag_encoded_text)
    
        return tag_encoded_text, slot_word_mask_tensor

    def get_req_encoded_text(self, word_seq_tensor, word_mask_tensor):
        batch_size = word_seq_tensor.shape[0]
        max_len = word_seq_tensor.shape[1]
        prefix_token = self.req_prefix_token_list.repeat(batch_size,1).cuda()
        prefix_token_character = self.req_prefix_token_character_list.repeat(batch_size,1,1).cuda()
        prefix = {"tokens":prefix_token,"token_characters":prefix_token_character}
        intent_prefix = self.text_field_embedder(prefix).cuda()
        mask_prefix = torch.ones((intent_prefix.shape[0],intent_prefix.shape[1]), dtype=torch.long).cuda()

        repeat_word_seq_tensor = word_seq_tensor.unsqueeze(1).repeat(1,self.num_req_intents,1,1).view(batch_size*self.num_req_intents,max_len,-1)
        repeat_word_mask_tensor = word_mask_tensor.unsqueeze(1).repeat(1,self.num_req_intents,1).view(-1,max_len)
        req_word_seq_tensor = torch.cat((intent_prefix, repeat_word_seq_tensor),1)
        req_word_mask_tensor = torch.cat((mask_prefix, repeat_word_mask_tensor),1)

        encoded_text = self.encoder(req_word_seq_tensor, req_word_mask_tensor)
        req_encoded_text = self.req_encoder(encoded_text, req_word_mask_tensor) if self.req_encoder else encoded_text 
        if self.dropout and self.req_encoder:
            req_encoded_text = self.dropout(req_encoded_text)
    
        return req_encoded_text, req_word_mask_tensor   

    @overrides
    def forward(self,  # type: ignore
                context_tokens: Dict[str, torch.LongTensor],
                tokens: Dict[str, torch.LongTensor],
                tags: List[Dict[str, Any]]  = None,
                slot_tag_tensor:torch.LongTensor = None,
                tag_mask_tensor:torch.LongTensor = None,
                intents: torch.LongTensor = None,
                reqs:torch.LongTensor = None,
                full_reqs:torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None,
                # pylint: disable=unused-argument
                **kwargs) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------

        Returns
        -------
        """
        if self.context_for_intent or self.context_for_tag or\
            self.attention_for_intent or self.attention_for_tag:
            embedded_context_input = self.text_field_embedder(context_tokens)

            if self.dropout:
                embedded_context_input = self.dropout(embedded_context_input)

            context_mask = util.get_text_field_mask(context_tokens)
            encoded_context = self.encoder(embedded_context_input, context_mask)

            if self.dropout:
                encoded_context = self.dropout(encoded_context)

            encoded_context_summary = util.get_final_encoder_states( 
                encoded_context,
                context_mask,
                self.encoder.is_bidirectional())
            batch_size, max_len = encoded_context_summary.shape
            req_encoded_context_summary = encoded_context_summary.unsqueeze(1).repeat(1,self.num_req_intents,1).view(batch_size*self.num_req_intents,max_len)
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text) 

        intent_encoded_text = self.intent_encoder(encoded_text, mask) if self.intent_encoder else encoded_text #如果intent有encoder 就再encdoer
        req_encoded_text, req_mask = self.get_req_encoded_text(embedded_text_input, mask)

        if self.dropout and self.intent_encoder: 
            intent_encoded_text = self.dropout(intent_encoded_text) 
            req_encoded_text = self.dropout(req_encoded_text)

        is_bidirectional = self.intent_encoder.is_bidirectional() if self.intent_encoder else self.encoder.is_bidirectional()
        if self._feedforward is not None:
            encoded_summary = self._feedforward(util.get_final_encoder_states(
                intent_encoded_text,
                mask,
                is_bidirectional))
            req_encoded_summary = self._feedforward(util.get_final_encoder_states(
                req_encoded_text,
                req_mask,
                is_bidirectional))
        else:
            encoded_summary = util.get_final_encoder_states(
                intent_encoded_text,
                mask,
                is_bidirectional)
            req_encoded_summary = util.get_final_encoder_states(
                req_encoded_text,
                req_mask,
                is_bidirectional)

        attend_context = None
        
        if self.attention_for_intent or self.attention_for_tag:
            attention_weights = self.attention(encoded_summary, encoded_context, context_mask.float())
            attended_context = util.weighted_sum(encoded_context, attention_weights)
            req_attention_weights = self.attention(req_encoded_summary, req_encoded_text, context_mask.float())
            req_attended_context = util.weighted_sum(req_encoded_text, attention_weights)

        if self.context_for_intent:
            encoded_summary = torch.cat([encoded_summary, encoded_context_summary], dim=-1)
            req_encoded_summary = torch.cat([req_encoded_summary, req_encoded_context_summary], dim=-1)
        
        if self.attention_for_intent:
            encoded_summary = torch.cat([encoded_summary, attended_context], dim=-1)
            req_encoded_summary = torch.cat([req_encoded_summary, req_attended_context], dim=-1)

        tag_encoded_text, tag_mask = self.get_tag_encoded_text(embedded_text_input, mask)
        if self.context_for_tag:
            tag_encoded_text = torch.cat([tag_encoded_text, 
                encoded_context_summary.unsqueeze(dim=1).expand(
                    encoded_context_summary.size(0),
                    tag_encoded_text.size(1),
                    encoded_context_summary.size(1))], dim=-1)
        if self.attention_for_tag:
            tag_encoded_text = torch.cat([tag_encoded_text, 
                attended_context.unsqueeze(dim=1).expand(
                    attended_context.size(0),
                    tag_encoded_text.size(1),
                    attended_context.size(1))], dim=-1)

        intent_logits = self.intent_projection_layer(encoded_summary)
        req_logits = self.req_projection_layer(req_encoded_summary)
        # req_logits = self.req_projection_layer(encoded_summary)

        intent_probs = torch.sigmoid(intent_logits)
        req_probs = torch.sigmoid(req_logits)
        predicted_intents = (intent_probs > 0.5).long()
        predicted_reqs = (req_probs > 0.3).long()#change req threshold from 0.5 to 0.3
        sequence_logits = self.tag_projection_layer(tag_encoded_text)

        if self.crf is not None:
            best_paths = self.crf.viterbi_tags(sequence_logits, mask)
            # Just get the tags and ignore the score.
            predicted_tags = [x for x, y in best_paths]
        else:
            predicted_tags = self.get_predicted_tags(sequence_logits)
        #print('predicted_tags = ', predicted_tags)
        
        # output = {"sequence_logits": sequence_logits, "mask": mask, "tags": predicted_tags,
        # "intent_logits": intent_logits, "intent_probs": intent_probs, "intents": predicted_intents,
        # "req_logits":req_logits, "req_probs":req_probs,"reqs":predicted_reqs}
        output = {"tokens":tokens['tokens'].tolist(),"sequence_logits": sequence_logits, "mask": mask, "tags": predicted_tags,
        "intent_logits": intent_logits, "intent_probs": intent_probs, "intents": predicted_intents,
        "req_logits":req_logits, "req_probs":req_probs,"reqs":predicted_reqs}

        if tags is not None:
            if self.crf is not None:
                # Add negative log-likelihood as loss
                log_likelihood = self.crf(sequence_logits, slot_tag_tensor, tag_mask_tensor) 
                output["loss"] = -log_likelihood

                # Represent viterbi tags as "class probabilities" that we can
                # feed into the metrics
                class_probabilities = sequence_logits * 0.
                for i, instance_tags in enumerate(predicted_tags):
                    for j, tag_id in enumerate(instance_tags):
                        class_probabilities[i, j, tag_id] = 1
            else:
                slot_tag_tensor = slot_tag_tensor.view(-1, slot_tag_tensor.shape[-1])
                tag_mask_tensor = tag_mask_tensor.view(-1, slot_tag_tensor.shape[-1])
                loss = sequence_cross_entropy_with_logits(sequence_logits, slot_tag_tensor, tag_mask_tensor)  #tag,mask
                class_probabilities = sequence_logits
                output["loss"] = loss / self.num_slot_intents

            if self.calculate_span_f1:
                 self._f1_metric(class_probabilities, slot_tag_tensor, tag_mask_tensor.float())#mask.float()) 
        
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]

        if tags is not None and metadata:
            self.decode(output) 
            self._dai_f1_metric(output["dialog_act"], [x["dialog_act"] for x in metadata])
            rewards = self.get_rewards(output["dialog_act"], [x["dialog_act"] for x in metadata]) if self.rl else None

        if intents is not None:
            output["loss"] += torch.mean(self.intent_loss(intent_logits, intents.float()))
            self._intent_f1_metric(predicted_intents, intents)
        
        if full_reqs is not None:
            req_mask, req_label = self.get_req_mask_and_label(full_reqs)
            if True in req_mask:
            
                active_req_logits = req_logits[req_mask]
                active_req_label = req_label[req_mask]
                output["loss"] += torch.mean(self.req_loss(active_req_logits, active_req_label))
                #self._req_f1_metric(predicted_reqs, reqs)

        return output

    def get_req_mask_and_label(self, full_reqs):
        # full_reqs : bc, num_req_full
        # print('full_reqs.shape = ', full_reqs.shape)
        bc = full_reqs.shape[0]
        mask = torch.zeros((bc*self.num_req_intents), dtype=torch.float)
        label = torch.zeros((bc*self.num_req_intents, self.num_req_slots), dtype=torch.float)
        for i, req in enumerate(full_reqs):
            if torch.max(req) != 0:
                triple_id = torch.argmax(req).item()
                triple = self.vocab.get_token_from_index(triple_id, namespace=self.req_full_label_namespace)
                domain, intent, slot = triple.split('-')
                req_intent_id = self.vocab.get_token_index(domain + '-' + intent, namespace=self.req_intent_label_namespace)
                req_slot_id = self.vocab.get_token_index(slot, namespace=self.req_slot_label_namespace)
                mask[i*self.num_req_intents + req_intent_id] = torch.FloatTensor([1])
                label[i*self.num_req_intents + req_intent_id, req_slot_id] = torch.FloatTensor([1])
        mask = mask == 1
        return mask.cuda(), label.cuda()


    def get_predicted_tags(self, sequence_logits: torch.Tensor) -> torch.Tensor:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = sequence_logits
        all_predictions = all_predictions.detach().cpu().numpy()
        if all_predictions.ndim == 3:  #去掉prefix
            predictions_list = [all_predictions[i][2:] for i in range(all_predictions.shape[0])]  
        else:
            predictions_list = [all_predictions[2:]]
        all_tags = []
        for predictions in predictions_list:
            tags = np.argmax(predictions, axis=-1)
            all_tags.append(tags)

        return all_tags

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        ``output_dict["tags"]`` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace=self.slot_sequence_label_namesapce) 
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]
        ]
        #print('predict tags = ',output_dict['tags'])
        
        output_dict["intents"] = [
                [self.vocab.get_token_from_index(intent[0], namespace=self.intent_label_namespace) 
            for intent in instance_intents.nonzero().tolist()] 
            for instance_intents in output_dict["intents"]
        ]
        #print("predict intents = ", output_dict['intents'])

        # output_dict["reqs"] = [
        #         [self.vocab.get_token_from_index(req[0], namespace=self.req_label_namespace)
        #     for req in instance_reqs.nonzero().tolist()]
        #     for instance_reqs in output_dict['reqs']
        # ]
        #print('predict reqs = ',output_dict['reqs'])
        
        #prediction: dialog act
        #label: metadata
        output_dict["dialog_act"] = []
        batch_size = len(output_dict["tags"]) // self.num_slot_intents
        for i in range(batch_size):
            dialog_act = {}
            for intent in output_dict["intents"][i]:  #add general
                domain,real_intent = intent.split('-')
                if real_intent in GENERAL_TYPE:
                    pair = [None, None]
                    if intent not in dialog_act:
                        dialog_act[intent] = [pair]
                    else:
                        dialog_act[intent].append(pair)
                    
            for j, req in enumerate(output_dict['reqs'][i*self.num_req_intents:(i+1)*self.num_req_intents]): #add requestable
                intent = self.vocab.get_token_from_index(j, self.req_intent_label_namespace)
                #domain, intent = prefix.split('-')
                if intent in output_dict['intents'][i]:
                    for k, pick_flag in enumerate(req):
                        if pick_flag:
                            slot = self.vocab.get_token_from_index(k, self.req_slot_label_namespace)
                            pair = [slot, None]
                            if intent not in dialog_act:
                                dialog_act[intent] = [pair]
                            else:
                                dialog_act[intent].append(pair)

                # domain, intent, slot = req.split('-')
                # temp_intent = domain + '-' + intent
                # if temp_intent in output_dict['intents'][i]:
                #     pair = [slot, None]
                #     if temp_intent not in dialog_act:
                #         dialog_act[temp_intent] = [pair]
                #     else:
                #         dialog_act[temp_intent].append(pair)
            
            for j,tags in enumerate(output_dict["tags"][i*self.num_slot_intents:(i+1)*self.num_slot_intents]): # add informable
                seq_len = len(output_dict["words"][i])
                spans = bio_tags_to_spans(tags[:seq_len])
                intent = self.vocab.get_token_from_index(j,self.slot_intent_label_namespace)
                for span in spans:
                    if intent in output_dict["intents"][i]:
                        slot = span[0]
                        value = " ".join(output_dict["words"][i][span[1][0]:span[1][1]+1])
                        if intent not in dialog_act:
                            dialog_act[intent] = [[slot, value]]
                        else:
                            dialog_act[intent].append([slot, value])

            output_dict["dialog_act"].append(dialog_act)

        # print(' len of tokens = {}, bc = {}'.format(len(output_dict['tokens']), batch_size))
        # for i in range(batch_size):
        #     print('text = ',' '.join([self.vocab.get_token_from_index(j,"tokens") for j in output_dict['tokens'][i] if self.vocab.get_token_from_index(j,"tokens") != '@@PADDING@@']))
        #     print('intents = ', output_dict['intents'][i])
        #     print("reqs = ",output_dict['reqs'][i])
        #     print('dialog act = ', output_dict['dialog_act'][i])
        #     print('='*100)
        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        intent_f1_dict = self._intent_f1_metric.get_metric(reset=reset)
        metrics_to_return.update({"int_"+x[:1]: y for x, y in intent_f1_dict.items() if "overall" in x})
        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            metrics_to_return.update({"tag_"+x[:1]: y for x, y in f1_dict.items() if "overall" in x})
        metrics_to_return.update(self._dai_f1_metric.get_metric(reset=reset))
        return metrics_to_return
