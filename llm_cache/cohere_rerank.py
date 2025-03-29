import os
import cohere  # pylint: disable=C0413
from typing import Tuple, Dict, Any

class CohereRerank:
    """Use the Cohere Rerank API to evaluate relevance of question and answer.

    Reference: https://docs.cohere.com/reference/rerank-1

    :param model: model name, defaults to 'rerank-english-v2.0', and multilingual option: rerank-multilingual-v2.0.
    :type model: str
    :param api_key: cohere api key, defaults to None.
    :type api_key: str

    Example:
        .. code-block:: python

            from gptcache.similarity_evaluation import CohereRerankEvaluation

            evaluation = CohereRerankEvaluation()
            score = evaluation.evaluation(
                {
                    'question': 'What is the color of sky?'
                },
                {
                    'answer': 'the color of sky is blue'
                }
            )
    """

    def __init__(self, model: str = "rerank-english-v2.0", api_key: str = None):
        self.co = cohere.Client(api_key)
        self.model = model

    def evaluation(self, src_dict: Dict[str, Any], cache_dict: Dict[str, Any], **kwargs) -> float:
        response = self.co.rerank(
            model=self.model,
            query=src_dict["question"],
            documents=cache_dict["answers"],
            top_n=kwargs["top_n"],
        )
        if response.results is None:
            return None
        return response.results

    def range(self) -> Tuple[float, float]:
        return 0.0, 1.0