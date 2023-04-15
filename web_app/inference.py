import os.path

from transformers import AutoModel, AutoTokenizer


def load_llm_model(root_path: str, model_name: str):
    model_path = os.path.join(root_path, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, revision="main")
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, revision="main").half().cuda()
    model = model.eval()
    return tokenizer, model
