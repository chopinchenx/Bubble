import os
from typing import List

import numpy as np
import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from gen_embedding import text2embedding, load_embed_model


def dataset2qdrant(root_path, file_path, embed_length: int = 384):
    client = QdrantClient("localhost", port=2023)
    collection_name = "data_collection"
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embed_length, distance=Distance.COSINE)
    )

    count = 0
    file_dir = os.path.join(root_path, file_path)
    for root_path, dirs, files in os.walk(file_dir):
        for file in tqdm.tqdm(files):
            file_path = os.path.join(root_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.readlines()
                for per_line in text:
                    parts = per_line.split("##")
                    item = text2embedding(parts[1])
                    client.upsert(collection_name=collection_name,
                                  wait=True,
                                  points=[PointStruct(id=count, vector=list([float(x) for x in item.tolist()]),
                                                      payload={"title": parts[0], "response": parts[1]})]
                                  )
                    count += 1


def result4search(query_embedding, limit_count: int = 3) -> List:
    client = QdrantClient("localhost", port=2023)
    collection_name = "data_collection"
    search_result = client.search(collection_name=collection_name,
                                  query_vector=np.array([float(x) for x in query_embedding]),
                                  limit=limit_count,
                                  # search_paras={"exact": False, "hnsw_ef": 128}
                                  )
    answer = []
    for i, result in enumerate(search_result):
        summary = result.payload["response"]
        score = result.score
        answer.append({"id": i + 1, "score": score, "title": result.payload["title"], "response": summary})

    return answer


if __name__ == "__main__":
    # dataset2qdrant("D:\\GitHub\\Bubble\\", "database\\dialog")
    embed_model = load_embed_model("D:\\GitHub\\LLM-Weights\\")
    input_text = "太极拳"
    answers = result4search(text2embedding(input_text))
