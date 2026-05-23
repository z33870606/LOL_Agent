import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain 相關套件
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

summary_model = init_chat_model("groq:openai/gpt-oss-120b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_latest_patch_info():
    """
    爬取英雄聯盟官方更新公告列表頁，取得「最新」版本公告的 URL 與版本號。
    """
    base_url = "https://www.leagueoflegends.com"
    list_url = f"{base_url}/en-us/news/tags/patch-notes/"
    
    print("正在檢查最新版本公告...")
    try:
        response = requests.get(list_url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 尋找包含 patch- 及 -notes 的連結
        link = soup.select_one("a[href*='patch-'][href*='-notes']")
        
        if not link:
            return None, None
            
        patch_url = base_url + link["href"]
        
        # 萃取版本號 (例如從 patch-14-23-notes 轉為 14.23)
        match = re.search(r'patch-(\d+-\d+)-notes', patch_url)
        if match:
            version = match.group(1).replace('-', '.')
            return patch_url, version
        
        return patch_url, "unknown"
    except Exception as e:
        print(f"檢查最新版本失敗: {e}")
        return None, None


def fetch_and_process_patch_notes(url: str, version: str):
    """
    下載網頁內容、清理格式、使用 LLM 預先計算版本總結，最後切塊存入 ChromaDB 向量資料庫。
    """
    print(f"開始下載並解析版本 {version} 的資料...")
    loader = AsyncHtmlLoader([url])
    docs = loader.load()

    if not docs:
        print("無法載入網頁。")
        return

    # Step 1: 網頁轉純文字與資料清洗
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    raw_text = docs_transformed[0].page_content

    # 清洗 Markdown 格式：還原超連結與移除粗斜體標籤，避免干擾後續的關鍵字比對
    clean_text = re.sub(r'(#+)\s*\[(.*?)\]\(.*?\)', r'\1 \2', raw_text)
    clean_text = clean_text.replace('**', '').replace('__', '')

    with open("debug_raw_patch.txt", "w", encoding="utf-8") as f:
        f.write(clean_text)
    print("已輸出 debug_raw_patch.txt (格式已清洗)")

    # Step 2: 使用 LLM 預先計算版本總結
    print("正在呼叫 LLM 閱讀完整更新檔，並產生版本總結...")
    safe_text = clean_text
    
    # 掃描並在第一個出現的無用區塊前一刀切斷
    cutoff_keywords = ["## Arena", "## Bugfixes", "## Mythic Shop", "## Upcoming Skins"]
    for keyword in cutoff_keywords:
        cutoff_index = safe_text.find(keyword)
        if cutoff_index != -1:
            safe_text = safe_text[:cutoff_index]
            break  # 找到第一個斷點就停止尋找
            
    # 以防該版本剛好沒有上述標題，還是保留 20000 字元
    safe_text = safe_text[:20000]
    
    # 建立總結 Prompt
    summary_prompt = f"""
    請閱讀以下英雄聯盟 {version} 版本的完整更新公告，並撰寫一份約 500 字的「版本精華總結」。
    
    【絕對遵守規則】
    1. 系統或機制的重大改變
    2. 被增強 (Buff) 的主要英雄清單
    3. 被削弱 (Nerf) 的主要英雄清單
    4. 重要的裝備 (Items) 改動清單
    5. 重要的符文(Rune) 改動清單
    6. 嚴格防幻覺：你列出的「所有英雄名稱」，必須 100% 完全照抄原文出現過的名字！絕對不可以自行替換、聯想或捏造原文沒有提到的英雄（例如：若原文寫 Ambessa，絕不能改成 Aurelion Sol）。
    
    以下是更新公告原文：
    {safe_text}
    """
    
    try:
        summary_result = summary_model.invoke(summary_prompt)
        pre_computed_summary = summary_result.content
        print("LLM 版本總結產生完成！")
    except Exception as e:
        print(f"產生總結時發生錯誤: {e}")
        pre_computed_summary = "系統提示：預先總結產生失敗，請確認 API 連線狀態。"

    # 將總結打包成獨立 Document，並加上專屬標籤
    summary_doc = Document(
        page_content=pre_computed_summary,
        metadata={
            "source": url,
            "patch": version,
            "type": "pre_computed_summary",  # 專屬標籤，與一般單一英雄的 patch_note 區隔
            "subject": "Patch Highlights"
        }
    )

    # Step 3: 文本結構化切塊 (Chunking)
    headers_to_split_on = [
        ("##", "category"),
        ("###", "subject"),
        ("####", "sub_subject")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # 第一次切塊：依據標題萃取 Metadata
    md_header_splits = markdown_splitter.split_text(clean_text)

    # 第二次切塊：依據長度切割，避免單一區塊過長
    chunk_size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    final_docs = chunk_size_splitter.split_documents(md_header_splits)
    
    # 補齊原有的通用 Metadata，並進行屬性值清洗
    for doc in final_docs:
        doc.metadata["source"] = url
        doc.metadata["patch"] = version
        doc.metadata["type"] = "patch_note"
        
        for key in ["category", "subject", "sub_subject"]:
            if key in doc.metadata:
                # 確保屬性被轉為字串並去除了首尾的隱藏空白字元
                doc.metadata[key] = str(doc.metadata[key]).strip()

    # Step 4: 存入 ChromaDB 向量資料庫
    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)

    # 確保資料庫永遠只保留最新版本的資料
    existing_data = db.get()
    if existing_data['ids']:
        print(f"正在清空 {len(existing_data['ids'])} 筆舊版本資料...")
        db.delete(ids=existing_data['ids'])

    # 將切塊後的公告文本與 LLM 總結一併存入資料庫
    all_docs_to_store = final_docs + [summary_doc]
    db.add_documents(all_docs_to_store)
    
    print(f"完成！版本 {version} 的資料（含 LLM 總結）已成功存入本地資料庫！")


def ensure_latest_patch_in_db():
    """
    系統啟動時的檢查點：確認本地端是否為最新版本，若非最新則自動觸發更新流程。
    """
    url, version = get_latest_patch_info()
    
    if not url or version == "unknown":
        print("警告：無法獲取最新版本資訊，跳過更新流程。")
        return

    print(f"網路上最新版本為: {version}")

    if not os.path.exists("./patch_db"):
        print("找不到本地資料庫，準備建立新庫...")
        fetch_and_process_patch_notes(url, version)
        return

    db = Chroma(persist_directory="./patch_db", embedding_function=embeddings)
    
    existing_docs = db.similarity_search("patch", k=1, filter={"patch": version})
    
    if existing_docs:
        print(f"確認：版本 {version} 已經存在於本地資料庫中，無需更新。")
    else:
        print(f"發現新版本 {version}！開始自動擷取並更新資料庫...")
        fetch_and_process_patch_notes(url, version)


if __name__ == "__main__":
    ensure_latest_patch_in_db()