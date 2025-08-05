from pymongo import MongoClient
import os, json
import insightface
from pymilvus import connections, db,FieldSchema, CollectionSchema, DataType, Collection, utility


# Load configuration from JSON
with open("db_config.json", "r") as f:
    db_config = json.load(f)


def get_mongo_conn(db_config):
    try:
        mongo_conf = db_config["mongodb"]
        client = MongoClient(f"mongodb://{mongo_conf['host']}:{mongo_conf['port']}/?serverSelectionTimeoutMS=3000")
        client.server_info()
        db = client[mongo_conf["database"]]
        # logs_col = db[mongo_conf]["daily_logs"]
        # config_col = db["config_file"]
        basic_col = db[mongo_conf["basic_collection"]]
        
        return basic_col

        # coll.insert_one(data)
        # print(True)
    except Exception as e:
        print(False)
        return {'error': str(e)}

def get_milvus_conn(db_config):
    try:
        milvus_conf = db_config["milvus"]
        connections.connect(host=milvus_conf["host"], port =milvus_conf["port"])
        # Create database if not exists (only supported in pymilvus 2.4+)
        if "facial_mapping" not in db.list_database():
            db.create_database("facial_mapping")
            print("Database 'facial_mapping' created.")

        
        milvus_collection_name = milvus_conf["collection_name"]

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512)  # Using BERT embeddings
        ]

        schema = CollectionSchema(fields, description="Face Embeddings storage")

        if not utility.has_collection(milvus_collection_name):
            Collection(name=milvus_collection_name, schema=schema)


        milvus_col = Collection(milvus_collection_name)
        return milvus_col
    
    except Exception as e:
        print(False)
        return {'error': str(e)}


