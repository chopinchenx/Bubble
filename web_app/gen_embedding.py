import os.path
from typing import List, Union

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor


def load_embed_model(root_path: str, model_name: str = 'multi-qa-MiniLM-L6-cos-v1'):
    model_path = os.path.join(root_path, model_name)
    return SentenceTransformer(model_path)


def text2embedding(text: Union[str, List[str]], model=None) -> Union[List[Tensor], ndarray, Tensor]:
    if model is None:
        model = load_embed_model("D:\\GitHub\\LLM-Weights\\", 'multi-qa-MiniLM-L6-cos-v1')
    return model.encode(text)


if __name__ == "__main__":
    embed_model = load_embed_model("D:\\GitHub\\LLM-Weights\\")
    input_text = "熊猫"
    # 384 dim
    print(len(text2embedding(input_text, embed_model)))
