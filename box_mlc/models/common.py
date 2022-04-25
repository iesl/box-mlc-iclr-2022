from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.predictors import Predictor
from allennlp.data.instance import Instance


@Predictor.register("multi-instance-typing")
class MultiInstanceTypingPredictor(Predictor):
    def _json_to_instance(self, json_dict: Dict[str, Any]) -> Instance:
        pass
        # return self._dataset_reader.text_to_instance(**json_dict)
