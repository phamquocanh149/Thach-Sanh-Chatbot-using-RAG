import chromadb
from chromadb.utils import embedding_functions

print("Đang kết nối vào CSDL và tải model Embedding...")

# 1. Kết nối lại với thư mục chứa Database
chroma_client = chromadb.PersistentClient(path="./thach_sanh_db")

# 2. BẮT BUỘC dùng lại đúng model Embedding lúc nạp dữ liệu
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)

# 3. Lấy Collection (Bảng dữ liệu) đã tạo
collection = chroma_client.get_collection(
    name="truyen_thach_sanh",
    embedding_function=sentence_transformer_ef
)

print(f"Kết nối thành công! Tổng số đoạn truyện trong DB: {collection.count()}\n")
print("-" * 50)

# ==========================================
# BÀI TEST 1: TÌM KIẾM THEO NGỮ NGHĨA (Vector Search thuần)
# ==========================================
query_1 = "Thạch Sanh dùng vũ khí gì để đánh lại con chim khổng lồ?"
print(f"BÀI TEST 1 - Câu hỏi: '{query_1}'")

results_1 = collection.query(
    query_texts=[query_1],
    n_results=2 # Lấy 2 kết quả liên quan nhất
)

for i in range(len(results_1['ids'][0])):
    print(f"\nKết quả top {i+1} (ID: {results_1['ids'][0][i]}):")
    print(f"- Khoảng cách (càng nhỏ càng sát nghĩa): {results_1['distances'][0][i]:.4f}")
    print(f"- Nội dung: {results_1['documents'][0][i]}")
    print(f"- Bối cảnh (Metadata): {results_1['metadatas'][0][i]['scene']}")

print("\n" + "=" * 50 + "\n")

# ==========================================
# BÀI TEST 2: TÌM KIẾM KẾT HỢP LỌC METADATA
# ==========================================
query_2 = "Chuyện gì đã xảy ra vậy?"
filter_location = "Ngục tối" # Cố tình ép DB chỉ tìm các sự kiện ở Ngục tối

print(f"BÀI TEST 2 - Câu hỏi: '{query_2}' | Kèm bộ lọc Địa điểm: '{filter_location}'")

results_2 = collection.query(
    query_texts=[query_2],
    n_results=1,
    where={"location": filter_location} # Bộ lọc Metadata phát huy tác dụng ở đây
)

if results_2['ids'][0]:
    print(f"\nKết quả tìm được ở '{filter_location}' (ID: {results_2['ids'][0][0]}):")
    print(f"- Nội dung: {results_2['documents'][0][0]}")
    print(f"- Nhân vật có mặt: {results_2['metadatas'][0][0].get('characters', 'Không rõ')}")
else:
    print("\nKhông tìm thấy sự kiện nào khớp với bộ lọc này!")

print("-" * 50)