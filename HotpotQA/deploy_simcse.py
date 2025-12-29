from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os
import torch

if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

app = Flask(__name__)

MODEL_NAME = "princeton-nlp/unsup-simcse-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)


def compute_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.cpu().numpy()


def compute_similarity(query_embedding, data_embeddings):
    query_embedding = query_embedding.reshape(1, -1)
    similarities = np.dot(data_embeddings, query_embedding.T).flatten()
    return similarities


@app.route('/retrieve', methods=['POST'])
def retrieve():
    try:
        req_data = request.json
        entity = req_data.get("entity", "")
        data = req_data.get("data", [])

        if not isinstance(entity, str) or not isinstance(data, list):
            return jsonify({"error": "Invalid input format. 'entity' must be a string and 'data' must be a list."}), 400

        if len(data) == 0:
            return jsonify({"error": "Data list is empty."}), 400
        query_embedding = compute_embeddings([entity])[0]
        data_embeddings = compute_embeddings(data)

        similarities = compute_similarity(query_embedding, data_embeddings)
        most_similar_index = np.argmax(similarities)
        most_similar_str = data[most_similar_index]

        return jsonify({"response": most_similar_str}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2022)
