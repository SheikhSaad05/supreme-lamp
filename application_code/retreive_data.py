import cv2
from pymongo import MongoClient
from db_connection import get_mongo_conn, get_milvus_conn
from face_model import prepare_model
import insightface
import numpy as np

import json
import base64
import time


# Load configuration from JSON
with open("model_config.json", "r") as f:
    model_data = json.load(f)

with open("db_config.json", "r") as f:
    db_data = json.load(f)


def search_person_by_image(model, image_path, basic_col, milvus_col, top_k=1, threshold = 0.4):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found or could not be read at path: {image_path}")
    
    faces = model.get(image)
    if not faces:
        return {"success": False, "message": "No face detected"}

    query_embedding = faces[0].embedding.tolist()
    milvus_col.load()

    results = milvus_col.search(
        data=[query_embedding],
        anns_field="vector",
        param={"metric_type": "COSINE"},
        limit=top_k,
        output_fields=["id"]
    )

    if not results or not results[0]:
        result["message"] = f"Failed match person"
        return json.dumps(result)


    hits = results[0]
    # Filter by similarity score
    filtered_hits = [hit for hit in hits if hit.distance >= (1 - threshold)]  # cosine sim threshold
    if not filtered_hits:
        return {"success": False, "message": f"No match found distance {threshold}"}
    
    matched_ids = [int(hit.id) for hit in filtered_hits]
    # matched_ids = [int(hit.id) for hit in hits]
    
    # matched_docs = list(basic_col.find({"milvus_person_id": {"$in": matched_ids}}))  # Query using milvus_id
    matched_docs = list(
        basic_col.find(
            {"milvus_person_id": {"$in": matched_ids}},
            {"_id": 0, "person_no": 1, "person_name": 1}
        )
    )
    return {
        "success": True,
        "matches": matched_docs
    }


if __name__ == '__main__':
    basic_col= get_mongo_conn(db_data)
    milvus_col = get_milvus_conn(db_data)
    model = prepare_model(model_data)

    # Test Image
    image_path = r"test_folder\Ronaldo.jpg"

    result = search_person_by_image(model,image_path, basic_col, milvus_col)
    print(result)