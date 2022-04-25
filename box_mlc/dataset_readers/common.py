from typing import Dict
from allennlp.common.registrable import Registrable


class JSONTransform(Registrable):
    default_implementation = "identity"

    def __call__(self, inp: Dict) -> Dict:
        return inp


@JSONTransform.register("identity")
class IdentityJSONTransform(JSONTransform):
    pass
