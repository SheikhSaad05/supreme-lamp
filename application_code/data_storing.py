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

def process_image_and_store(model,image_path,person_no, person_name, basic_col, milvus_col):
    # Read image from local drive
    result = {"success": False, "message": ""}
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image from {image_path}")
            return
        
        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer)

        person_image = str(image_base64)[2:-1]
        # document["person_image"] = str(image_base64)[2:-1]
        # Perform face detection
        faces = model.get(image)
        if not faces:
            result["message"] = f"Failed to read Image from {image_path}"
            return json.dumps(result)
        face_embeddings = faces[0].embedding.tolist()
        
        # Store the first time card info in MongoDB
        # ------------------------------------------
        existing_person = basic_col.find_one({"person_no": person_no})
        if existing_person:
            # If person_id already exists, update the existing document
            basic_col.update_one({"person_no": person_no}, {"$set": {
                "person_name": person_name,
                "person_image": person_image
                # "person_embedding": face_embeddings
            }})
            milvus_person_id = existing_person["milvus_person_id"]
            # Check in Milvus and
            milvus_col.delete(f"id in [{milvus_person_id}]")
            milvus_col.insert([[face_embeddings]])
            milvus_col.flush()

            result["message"] = f"Person with person_no {person_no} updated in MongoDB and Milvus"
        else:
            insert_result = milvus_col.insert([[face_embeddings]])
            milvus_person_id = insert_result.primary_keys[0]

            document = {
                'milvus_person_id': milvus_person_id,
                'person_no': person_no,
                'person_name': person_name,
                'person_image': person_image
            }
            milvus_col.flush()
            basic_col.insert_one(document)
            # Create index (after inserting data)
            if not milvus_col.has_index():
                milvus_col.create_index(
                    field_name="vector",
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128}
                    }
                )

            milvus_col.load()

            # Query the inserted record from Milvus
            results = milvus_col.query(
                expr=f"id == {milvus_person_id}",
                output_fields=["id", "vector"]
            )

            print("Inserted Record:")
            print(results[0])


            result["message"] = "New person inserted successfully"

        result["success"] = True
    except Exception as ex:
        result["message"] = f"Error occurred while processing the image: {ex}"

    return json.dumps(result)


if __name__ == '__main__':
    basic_col= get_mongo_conn(db_data)
    milvus_col = get_milvus_conn(db_data)
    model = prepare_model(model_data)
    # Example usage:
    image_path = r"test_folder\Ronaldo.jpg"
    # person_id = 1
    person_no = "151005"
    person_name = "Ronaldo"
    result = process_image_and_store(model,image_path,person_no, person_name, basic_col, milvus_col)
    print(result)