import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

os.environ["GOOGLE_API_KEY"] = "AIzaSyB-ZT7ZlifzDKAbtVk1Jk1XtnNxqqC07Ec"

embeddings = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder"
)
current_dir = os.path.dirname(os.path.abspath(__file__))

db_path = os.path.join(current_dir, "..", "data", "thach_sanh_db")

print(f" Đang tải CSDL từ: {db_path}")

vectorstore = Chroma(
    collection_name="truyen_thach_sanh",
    persist_directory=db_path,
    embedding_function=embeddings
)

count = vectorstore._collection.count()
print(f"📦 Số lượng đoạn truyện tìm thấy trong CSDL: {count}")
if count == 0:
    print(" CSDL rỗng! Vẫn trỏ sai chỗ rồi.")
    exit()

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 2,                 # Lấy tối đa 2 đoạn văn bản
        "score_threshold": 0.3  # Ngưỡng chấp nhận (chỉ lấy những đoạn có độ tương đồng >= 0.3)
    }
)

# 3. KHỞI TẠO LLM GOOGLE GEMINI
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1
)

# 4. THIẾT LẬP PROMPT THEO CHUẨN CHATML
system_prompt = """Bạn là Thạch Sanh trong truyện cổ tích Việt Nam. Hãy trả lời câu hỏi của người dùng, xưng 'tôi', dựa trên bối cảnh dưới đây.

LƯU Ý TỐI QUAN TRỌNG:
1. Chỉ được phép trả lời dựa trên thông tin có trong phần "Bối cảnh".
2. Nếu Bối cảnh trống rỗng hoặc không chứa thông tin để trả lời, BẮT BUỘC phải nói chính xác: "Chuyện này tôi không rõ, bạn hỏi chuyện khác về tôi đi." KHÔNG ĐƯỢC TỰ BỊA RA.

Bối cảnh:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}")
])

# 5. XÂY DỰNG CHUỖI LCEL (PIPELINE)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)

# 6. CHẠY THỬ HỆ THỐNG
if __name__ == "__main__":
    print("Hello, I'm Thạch Sanh! Hãy hỏi tôi điều gì đó về cuộc đời tôi nhé! (Ví dụ: 'Vua giao cho ai quyền xử tội mẹ con Lý Thông vậy anh?')\n")
    
    question_1 = "Vua giao cho ai quyền xử tội mẹ con Lý Thông vậy anh?"
    print(f"👤 Người dùng: {question_1}")
    
    # --- ĐOẠN NÀY ĐỂ IN RA NHỮNG GÌ RETRIEVER TÌM ĐƯỢC ---
    docs = retriever.invoke(question_1)
    print(" [DEBUG] Context truy xuất được:")
    if len(docs) == 0:
        print("   -> (Không tìm thấy đoạn nào vượt qua ngưỡng 0.3)")
    else:
        for i, doc in enumerate(docs):
            print(f"   Đoạn {i+1}: {doc.page_content[:200]}...") # In 200 chữ đầu cho đỡ rối
    print("-----------------------------------\n")
    # ----------------------------------------------------
    
    response_1 = rag_chain.invoke(question_1)
    print(f"🗡️ Thạch Sanh: {response_1}\n")