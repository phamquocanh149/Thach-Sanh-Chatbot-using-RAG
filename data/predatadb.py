import json
import chromadb
from chromadb.utils import embedding_functions

# 1. Đọc file JSON
with open("truyenfull.json", "r", encoding="utf-8-sig") as f:
    data_thach_sanh = json.load(f)

print(f"Đã load thành công {len(data_thach_sanh)} bản ghi từ file!")

# 2. Khởi tạo ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./thach_sanh_db")

# 3. Cài đặt mô hình Embedding tiếng Việt
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# 4. Tạo Collection
collection = chroma_client.get_or_create_collection(
    name="truyen_thach_sanh",
    embedding_function=sentence_transformer_ef
)

# 5. Tách dữ liệu và XỬ LÝ METADATA
documents = []
metadatas = []
ids = []

for chunk in data_thach_sanh:
    documents.append(chunk["text"])
    ids.append(chunk["id"])
    
    # Ép mảng (List) thành chuỗi (String) để ChromaDB không báo lỗi
    meta = chunk["metadata"].copy()
    for key, value in meta.items():
        if isinstance(value, list):
            meta[key] = ", ".join(value) 
    
    metadatas.append(meta)

# 6. Lưu vào Database
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Đã lưu thành công {collection.count()} đoạn văn bản vào ChromaDB!")