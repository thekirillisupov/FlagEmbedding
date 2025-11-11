import torch
from transformers import PreTrainedModel, AutoTokenizer
import logging

from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel

logger = logging.getLogger(__name__)


class CrossDecoderModel(AbsRerankerModel):
    """
    Model class for decoder only reranker.

    Args:
        base_model (PreTrainedModel): The underlying pre-trained model used for encoding and scoring input pairs.
        tokenizer (AutoTokenizer, optional): The tokenizer for encoding input text. Defaults to ``None``.
        train_batch_size (int, optional): The batch size to use. Defaults to ``4``.
        logit_calculation_type (str, optional): Type of logit calculation. Either 'margin_score' (yes_scores - no_scores) 
            or 'only_yes' (just yes_scores). Defaults to ``'margin_score'``.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: AutoTokenizer = None,
        train_batch_size: int = 4,
        logit_calculation_type: str = 'margin_score',
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=train_batch_size,
        )
        self.logit_calculation_type = logit_calculation_type

    def encode(self, features):
        """Encodes input features to logits.

        Args:
            features (dict): Dictionary with input features.

        Returns:
            torch.Tensor: The logits output from the model.
        """
        if features is None:
            return None

        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True)

        yes_scores = outputs.logits[:, -1, self.yes_loc]
        
        if self.logit_calculation_type == 'only_yes':
            scores = yes_scores
        else:
            no_scores = outputs.logits[:, -1, self.no_loc]
            scores = yes_scores - no_scores
        
        return scores.contiguous()