from typing import List, Tuple, Union, Dict, Any, Optional
import allennlp

allennlp_major_version = int(allennlp.__version__.split(".")[0])
from box_mlc.modules.box_embedding import BoxEmbedding
import logging

logger = logging.getLogger(__name__)


if allennlp_major_version < 2:
    from allennlp.training.trainer import EpochCallback, GradientDescentTrainer
    from allennlp.data.dataloader import TensorDict

    @EpochCallback.register("tensorboard-box-embedding-stats")
    class TensoboardBoxEmbeddingStats(EpochCallback):
        def __init__(self, box_embedding_modules: List[str]):
            self.box_embeddings_names = box_embedding_modules

        def __call__(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
        ) -> None:
            if is_master and trainer.model is not None:
                for name in self.box_embeddings_names:
                    module: BoxEmbedding = getattr(trainer.model, name, None)

                    if module is None:
                        logger.warning(f"{name} not an attribute on model.")
                    else:
                        for (
                            stat_name,
                            stat,
                        ) in module.get_statistics_and_histograms().items():
                            trainer._tensorboard.add_train_histogram(
                                name + "/" + stat_name, stat
                            )
