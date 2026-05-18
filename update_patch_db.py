import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_latest_patch_info():
    """
    爬取官方列表頁，取得「最新」更新公告的 URL 與版本號
    """
    base_url = "https://www.leagueoflegends.com"
    list_url = f"{base_url}/en-us/news/tags/patch-notes/"
    
    print("🔍 正在檢查最新版本公告...")
    try:
        response = requests.get(list_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        link = soup.select_one("a[href*='patch-'][href*='-notes']")
        
        if not link:
            return None, None
            
        patch_url = base_url + link["href"]
        
        match = re.search(r'patch-(\d+-\d+)-notes', patch_url)
        if match:
            version = match.group(1).replace('-', '.')
            return patch_url, version
        
        return patch_url, "unknown"
    except Exception as e:
        print(f"錯誤 檢查最新版本失敗: {e}")
        return None, None

def fetch_and_process_patch_notes(url: str, version: str):
    """
    使用 LangChain 工具下載並透過「Markdown 標題」切塊存入 ChromaDB
    """
    print(f"📥 開始下載並解析版本 {version} 的資料...")
    loader = AsyncHtmlLoader([url])
    docs = loader.load()

    if not docs:
        print("錯誤 無法載入網頁。")
        return

    # 1. 轉成純文字 Markdown
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    raw_text = docs_transformed[0].page_content

    # 💡 重構重點 1：資料正規化 (清洗 Markdown 格式)
    # 將標題中的超連結格式 ### [Ashe](...) 還原為乾淨的 ### Ashe
    clean_text = re.sub(r'(#+)\s*\[(.*?)\]\(.*?\)', r'\1 \2', raw_text)
    # 移除內文中可能干擾字串比對的粗體 (**) 與斜體 (__) 標籤
    clean_text = clean_text.replace('**', '').replace('__', '')

    with open("debug_raw_patch.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)
    print("📝 已輸出 debug_raw_patch.txt (格式已清洗)")

    # 2. 建立 Markdown 標題切塊器
    headers_to_split_on = [
        ("##", "category"),
        ("###", "subject"),
        ("####", "sub_subject")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 3. 第一次切塊：依據標題萃取 Metadata
    md_header_splits = markdown_splitter.split_text(clean_text)

    # 4. 第二次切塊：依據長度切割
    chunk_size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )
    final_docs = chunk_size_splitter.split_documents(md_header_splits)
    
    # 5. 補齊原有的通用 Metadata
    for doc in final_docs:
        doc.metadata["source"] = url
        doc.metadata["patch"] = version
        doc.metadata["type"] = "patch_note"
        
        # 💡 重構重點 2：強制清洗 Metadata 字典內的屬性值
        for key in ["category", "subject", "sub_subject"]:
            if key in doc.metadata:
                # 確保屬性被轉為字串並去除了首尾的隱藏空白字元
                doc.metadata[key] = str(doc.metadata[key]).strip()
                
                # 開發者除錯提示：印出包含 Ashe 的 Metadata 讓你安心
                if "Ashe" in doc.metadata[key]:
                    print(f"🎯 成功清洗並建立英雄 Metadata: {doc.metadata[key]}")

    # 6. 存入 ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)

    existing_data = db.get()
    if existing_data['ids']:
        print(f"🧹 正在清空 {len(existing_data['ids'])} 筆舊版本資料...")
        db.delete(ids=existing_data['ids'])

    db.add_documents(final_docs)
    print(f"完成 版本 {version} 的資料已成功存入本地資料庫！")

def ensure_latest_patch_in_db():
    url, version = get_latest_patch_info()
    
    if not url or version == "unknown":
        print("警告 無法獲取最新版本資訊，跳過更新。")
        return

    print(f"💡 網路上最新版本為: {version}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists("./patch_db"):
        print("警告 找不到本地資料庫，準備建立新庫...")
        fetch_and_process_patch_notes(url, version)
        return

    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)
    
    existing_docs = db.similarity_search("patch", k=1, filter={"patch": version})
    
    if existing_docs:
        print(f"確認 版本 {version} 已經存在於本地資料庫中，無需更新。")
    else:
        print(f"發現 新版本 {version}！開始自動擷取並更新...")
        fetch_and_process_patch_notes(url, version)

if __name__ == "__main__":
    ensure_latest_patch_in_db()
