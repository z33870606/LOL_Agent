from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def check_precomputed_summary():
    # 初始化與你的 main.py 相同的 Embedding 模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 連線到本地資料庫
    db = Chroma(
        persist_directory="./patch_db",
        embedding_function=embeddings
    )
    
    # 使用過濾器，精準抓取 type 為 pre_computed_summary 的文件
    docs = db.get(where={"type": "pre_computed_summary"})
    
    if docs and docs.get('documents') and len(docs['documents']) > 0:
        print("=== 🚨 資料庫中儲存的版本總覽 (pre_computed_summary) 🚨 ===\n")
        print(docs['documents'][0])
        print("\n=======================================================")
    else:
        print("資料庫中找不到任何預先計算的總覽文件！")

if __name__ == "__main__":
    check_precomputed_summary()