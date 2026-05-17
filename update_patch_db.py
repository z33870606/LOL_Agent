import os
import re
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
# 💡 這裡新增了 MarkdownHeaderTextSplitter
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
        
        # 尋找第一個包含 patch-notes 的連結 (通常是最新的)
        link = soup.select_one("a[href*='patch-'][href*='-notes']")
        
        if not link:
            return None, None
            
        patch_url = base_url + link["href"]
        
        # 用正則表達式從網址中翠取出版本號 (例如 14-8 變成 14.8)
        match = re.search(r'patch-(\d+-\d+)-notes', patch_url)
        if match:
            version = match.group(1).replace('-', '.')
            return patch_url, version
        
        return patch_url, "unknown"
    except Exception as e:
        print(f"❌ 檢查最新版本失敗: {e}")
        return None, None

def fetch_and_process_patch_notes(url: str, version: str):
    """
    使用 LangChain 工具下載並透過「Markdown 標題」切塊存入 ChromaDB
    """
    print(f"📥 開始下載並解析版本 {version} 的資料...")
    loader = AsyncHtmlLoader([url])
    docs = loader.load()

    if not docs:
        print("❌ 無法載入網頁。")
        return

    # 1. 轉成純文字 Markdown
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    clean_text = docs_transformed[0].page_content

    # 2. 建立 Markdown 標題切塊器
    # Riot 公告通常的結構：## 大分類 -> ### 具體項目 -> #### 細節或次項目
    headers_to_split_on = [
        ("##", "category"),       # 例如: "Champions", "Items", "Bugfixes"
        ("###", "subject"),       # 例如: "Morgana", "Infinity Edge"
        ("####", "sub_subject")   # 預防有些裝備或系統改動有更深層的標題
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 3. 第一次切塊：依據標題 (自動將標題萃取為 Metadata)
    md_header_splits = markdown_splitter.split_text(clean_text)

    # 4. 第二次切塊：防止單一英雄/機制的改動說明過長，套用字數限制做二次切割
    # 這裡的 chunk_docs 會繼承第一步抓到的標題 Metadata
    chunk_size_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_docs = chunk_size_splitter.split_documents(md_header_splits)
    
    # 5. 補齊原有的通用 Metadata (將網址、版本號加入每一塊碎片中)
    for doc in final_docs:
        doc.metadata["source"] = url
        doc.metadata["patch"] = version
        doc.metadata["type"] = "patch_note"

    # 6. 存入 ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)
    db.add_documents(final_docs)
    print(f"✅ 版本 {version} 的資料已成功以「標題切塊法」存入本地資料庫！")

def ensure_latest_patch_in_db():
    """
    主流程：檢查最新版本 -> 比對資料庫 -> 決定是否下載
    """
    url, version = get_latest_patch_info()
    
    if not url or version == "unknown":
        print("⚠️ 無法獲取最新版本資訊，跳過更新。")
        return

    print(f"💡 網路上最新版本為: {version}")

    # 檢查本地資料庫是否已經有這個版本
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists("./patch_db"):
        print("⚠️ 找不到本地資料庫，準備建立新庫...")
        fetch_and_process_patch_notes(url, version)
        return

    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)
    
    # 隨便搜一個字，並過濾這個 patch，看有沒有結果
    existing_docs = db.similarity_search("patch", k=1, filter={"patch": version})
    
    if existing_docs:
        print(f"🆗 版本 {version} 已經存在於本地資料庫中，無需更新。")
    else:
        print(f"🆕 發現新版本 {version}！開始自動擷取並更新...")
        fetch_and_process_patch_notes(url, version)

if __name__ == "__main__":
    ensure_latest_patch_in_db()