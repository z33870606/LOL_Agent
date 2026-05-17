from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def check_database_content():
    print("🔍 正在載入本地 Chroma 資料庫...")
    
    # 1. 必須使用與寫入時一模一樣的 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. 連結到你的本地資料夾
    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)
    
    # 3. 獲取資料庫內的所有資料
    # db.get() 會回傳包含 'documents', 'metadatas', 'ids' 等的字典
    data = db.get()
    
    total_chunks = len(data['documents'])
    print(f"\n📊 資料庫中目前共有 {total_chunks} 筆文字片段 (Chunks)。\n")
    
    if total_chunks == 0:
        print("⚠️ 資料庫是空的！")
        return

    # 4. 印出全部資料來人工檢查
    print(f"=== 👀 預覽所有 {total_chunks} 筆資料 ===")
    
    # 用迴圈跑遍所有資料
    for i in range(total_chunks):
        print(f"\n片段 ID: {data['ids'][i]}")
        print(f"標籤 (Metadata): {data['metadatas'][i]}")
        # 只印出前 200 個字元，避免畫面被文字洗版
        print(f"內容摘要: {data['documents'][i][:200]}...") 

if __name__ == "__main__":
    check_database_content()