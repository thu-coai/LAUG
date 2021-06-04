import warnings
import json
from typing import List
from overrides import overrides

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register("copynet")
class CopyNetPredictor(Predictor):
    """
    Predictor for the CopyNet model.
    """

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        warnings.warn(
            "The 'copynet' predictor has been deprecated in favor of "
            "the 'seq2seq' predictor.",
            DeprecationWarning,
        )

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source_string": source})
    
    def load_line(self, line: str) -> JsonDict:
        source, target = line.strip("\n").split("\t")
        return {"source_string": source, "target_string": target}
        
    def dump_line(self, outputs: JsonDict) -> str:
        return json.dumps(outputs['predictions']) + "\n"
    
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        source_tokens = output_dict['metadata']['source_tokens']
        predictions = output_dict['predictions']
        for idx, prediction in enumerate(predictions):
            if self._model._end_index in prediction:
                prediction = prediction[: prediction.index(self._model._end_index)]
            prediction = [self._model.vocab.get_token_from_index(index, self._model._target_namespace) if index < self._model._target_vocab_size else source_tokens[
                    index - self._model._target_vocab_size] for index in prediction]
            predictions[idx] = ' '.join(prediction)
        return output_dict
    
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        output_dicts = self.predict_batch_instance(instances)
        for output_dict in output_dicts:
            source_tokens = output_dict['metadata']['source_tokens']
            predictions = output_dict['predictions']
            for idx, prediction in enumerate(predictions):
                if self._model._end_index in prediction:
                    prediction = prediction[: prediction.index(self._model._end_index)]
                prediction = [self._model.vocab.get_token_from_index(index, self._model._target_namespace) if index < self._model._target_vocab_size else source_tokens[
                        index - self._model._target_vocab_size] for index in prediction]
                predictions[idx] = ' '.join(prediction)
        return output_dicts

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source = json_dict["source_string"]
        return self._dataset_reader.text_to_instance(source)
