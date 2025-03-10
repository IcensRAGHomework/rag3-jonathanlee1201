import datetime
import chromadb
import traceback
import pandas as pd
import sqlite3

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

from datetime import datetime

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    conn = sqlite3.connect('chroma.sqlite3')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS travel (
        id INTEGER PRIMARY KEY,
        file_name TEXT,
        name TEXT,
        type TEXT,
        address TEXT,
        tel TEXT,
        city TEXT,
        town TEXT,
        date INTEGER,
        document TEXT
    )
    ''')

    df = pd.read_csv('COA_OpenData.csv')

    # init ChromaDB
    client = chromadb.Client()

    # create Collection
    collection = client.create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"}
    )

    # add data to Collection
    for index, row in df.iterrows():
        metadata = {
            "file_name": "COA_OpenData.csv",
            "name": row['Name'],
            "type": row['Type'],
            "address": row['Address'],
            "tel": row['Tel'],
            "city": row['City'],
            "town": row['Town'],
            "date": int(datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp()),
        }

        document = row['HostWords']

        collection.add(
            ids=str(index),
            documents=document,
            metadatas=metadata
        )

        cursor.execute('''
        INSERT INTO travel (file_name, name, type, address, tel, city, town, date, document)
        VALUES (:file_name, :name, :type, :address, :tel, :city, :town, :date, :document)
        ''', {**metadata, "document": document})

    # commit
    conn.commit()
    conn.close()

    return collection

    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == "__main__":
    generate_hw01()
