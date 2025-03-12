import chromadb
import traceback
import pandas as pd
import openai

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

from datetime import datetime

dbpath = "./"

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

def generate_db():
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

    df = pd.read_csv('COA_OpenData.csv')
    
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

    return collection



def generate_hw01():
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


def generate_hw02(question, city=None, store_type=None, start_date=None, end_date=None):
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

     # 查詢集合
    results = collection.query(
        query_texts=[question],
         n_results=40
    )

    filtered_results = []

    print("ToQuery:", question, city, store_type, start_date.timestamp(), end_date.timestamp())

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance;
        metadata = results["metadatas"][0][i] 
        name = metadata.get("name", "Unknown")
        store_t = metadata.get("type", "Unknown")
        time_t = metadata.get("date", 0)
        print(name, store_t, time_t, distance)
    
        if similarity >= 0.7: 
            if city and metadata.get("city", "Unknown") not in city: 
                continue
            if store_type and metadata.get("type", "Unknown") not in store_type:
                continue
            if start_date.timestamp() and start_date.timestamp() > metadata.get("date", 0):
                continue
            if end_date.timestamp() and end_date.timestamp() < metadata.get("date", 0):
                continue
            filtered_results.append((metadata.get("name", "Unknown"), similarity))

    filtered_results.sort(key=lambda x: x[1], reverse=True)

     # 取前 10 個結果
    top_results = [store_name for store_name, _ in filtered_results[:10]]
    
    for i in range(min(5, len(filtered_results))):
        print(f"Filtered_Res: {filtered_results[i]}")

    return top_results


def generate_hw03(question, store_name, new_store_name, city, store_type):
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
    
     # 查詢集合
    results = collection.query(
        query_texts=[question],
         n_results=40
    )

    filtered_results = []

    print("ToQuery2:", question, city, store_type, start_date.timestamp(), end_date.timestamp())

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance;
        metadata = results["metadatas"][0][i] 
        name = metadata.get("name", "Unknown")
        store_t = metadata.get("type", "Unknown")
        time_t = metadata.get("date", 0)
        print(name, store_t, time_t, distance)
    
        if similarity >= 0.7: 
            if city and metadata.get("city", "Unknown") not in city: 
                continue
            if store_type and metadata.get("type", "Unknown") not in store_type:
                continue
            if start_date.timestamp() and start_date.timestamp() > metadata.get("date", 0):
                continue
            if end_date.timestamp() and end_date.timestamp() < metadata.get("date", 0):
                continue
            filtered_results.append((metadata.get("name", "Unknown"), similarity))

    filtered_results.sort(key=lambda x: x[1], reverse=True)

     # 取前 10 個結果
    top_results = [store_name for store_name, _ in filtered_results[:10]]
    
    for i in range(min(5, len(filtered_results))):
        print(f"Filtered_Res: {filtered_results[i]}")

    return top_results
    
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
    #create_database()
    
    #Collection
    collection = generate_hw01()
    print(collection)

    question = "我想要找有關茶餐點的店家"
    city = ["宜蘭縣", "新北市"]
    store_type = ["美食"]
    start_date = datetime(2024, 4, 1)
    end_date = datetime(2024, 5, 1)

    results = generate_hw02(question, city, store_type, start_date, end_date)
    print(results)
