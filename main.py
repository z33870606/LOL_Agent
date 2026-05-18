import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from update_patch_db import ensure_latest_patch_in_db

# ==============================
# 1. 初始化系統與全域變數
# ==============================


def initialize_system(csv_path: str = "oracles_match_data.csv"):
    """
    初始化 LOL_AGENT 系統環境，包含環境變數、向量模型與本地賽事數據。
    """
    load_dotenv()
    ensure_latest_patch_in_db()

    # 初始化 Embeddings
    init_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 初始化賽事資料
    file_path = Path(csv_path)
    if file_path.exists():
        try:
            init_df = pd.read_csv(file_path, low_memory=False)
            if 'champion' in init_df.columns and 'teamname' in init_df.columns:
                init_df['champion'] = init_df['champion'].astype(str)
                init_df['teamname'] = init_df['teamname'].astype(str)
            print(f"成功載入賽事數據，共 {len(init_df)} 筆。")
        except Exception as e:
            print(f"讀取 CSV 時發生未預期的錯誤: {e}")
            init_df = pd.DataFrame()
    else:
        print(f"找不到指定的 CSV 檔案: {file_path.absolute()}")
        init_df = pd.DataFrame()

    return init_embeddings, init_df

# 💡 修正點：在模組載入時就執行初始化，並將變數命名為工具所預期的 embeddings 與 df
embeddings, df = initialize_system()


# ==============================
# 2. Tools
# ==============================

def get_patch_db():
    """輔助函式：取得 ChromaDB 連線實例，避免重複初始化"""
    if not os.path.exists("./patch_db"):
        return None
    return Chroma(
        persist_directory="./patch_db",
        embedding_function=embeddings
    )

@tool
def search_patch_overview() -> str:
    """
    查詢「最新版本的整體改動總覽」。
    何時使用：當使用者詢問大範圍的版本資訊，例如「新版本更新了什麼？」、「有哪些英雄被削弱或增強？」、「版本總覽」。
    注意：如果使用者明確指定了某個英雄的名字（例如 Ashe），請不要使用此工具，請改用 search_champion_patch_notes。
    """
    db = get_patch_db()
    if not db:
        return "系統警告：無本地資料庫，請回答查無資料。"

    try:
        docs = db.similarity_search(
            query="patch overview champions items buffs nerfs",
            k=15
        )

        if not docs:
            return "系統警告：資料庫中目前沒有更新總覽，請直接回答查無最新版本資訊，嚴禁自行捏造。"

        rag_chunks = []
        for d in docs:
            category = d.metadata.get('category', '')
            subject = d.metadata.get('subject', '')
            sub_subject = d.metadata.get('sub_subject', '')
            
            headers = [h for h in [category, subject, sub_subject] if h]
            header_text = f"[{' > '.join(headers)}]" if headers else "[General]"
            rag_chunks.append(f"{header_text}\n{d.page_content}")
            
        rag = "\n".join(rag_chunks)
        return f"[Latest Patch Overview]\n本地資料：\n{rag}"

    except Exception as e:
        return f"資料庫查詢發生錯誤: {e}"


@tool
def search_champion_patch_notes(champion_name: str) -> str:
    """
    查詢「特定單一英雄」的最新版本改動數值與細節。
    何時使用：當使用者明確詢問某個具體英雄的改動時，例如「請問 Ashe 的更新是什麼？」、「李星有被改嗎？」。
    參數 champion_name：請傳入該英雄的英文名稱（例如 Ashe, LeeSin）。
    """
    # ==================== [除錯專區 1：檢查 Agent 傳入參數] ====================
    print("\n" + "="*40)
    print(f"[DEBUG] 🤖 Agent 正在呼叫 search_champion_patch_notes 工具")
    print(f"[DEBUG] 📥 Agent 傳入的原始參數 champion_name: '{champion_name}'")
    # =========================================================================

    db = get_patch_db()
    if not db:
        return "系統警告：無本地資料庫，請回答查無資料。"

    try:
        print(f"[DEBUG] 🔍 正在向 ChromaDB 進行過濾查詢 (subject='{champion_name}')...")
        docs = db.similarity_search(
            query=champion_name, 
            k=4, 
            filter={
                "$and": [
                    {"type": "patch_note"},
                    {"subject": champion_name}
                ]
            }
        )

       # ============= [除錯專區 2：檢查資料庫回傳結果] ====================
        print(f"[DEBUG] 📊 ChromaDB 回傳的文檔數量: {len(docs)}")
        if docs:
            print(f"[DEBUG] 📄 第一筆文檔的 Metadata: {docs[0].metadata}")
            print(f"[DEBUG] 📄 第一筆文檔的內容片段: {docs[0].page_content[:100]}...\n")
        else:
            print(f"[DEBUG] ⚠️ ChromaDB 找不到任何符合 filter 條件的資料！")
        print("="*40 + "\n")
        # =========================================================================

        if not docs:
            return f"系統警告：資料庫中沒有 {champion_name} 的改動資料。"

        rag_chunks = []
        for d in docs:
            subject = d.metadata.get('subject', '')
            sub_subject = d.metadata.get('sub_subject', '') 
            headers = [h for h in [subject, sub_subject] if h]
            header_text = f"[{' > '.join(headers)}]" if headers else "[General]"
            rag_chunks.append(f"{header_text}\n{d.page_content}")
            
        rag = "\n\n".join(rag_chunks)
        return f"[Patch Notes Data for {champion_name}]\n{rag}"

    except Exception as e:
        return f"資料庫查詢發生錯誤: {e}"


@tool
def get_champion_esports_stats(champion_name: str) -> str:
    """
    查詢特定英雄的職業賽事出場數、勝率與主要路線。
    何時使用：當使用者詢問關於比賽數據、勝率、出場次數等統計資訊時。
    參數 champion_name：請傳入該英雄的英文名稱。
    """
    if 'df' not in globals() or df.empty:
        return "系統警告：賽事資料庫未載入。"
    
    try:
        champ = df[
            (df['champion'].str.lower() == champion_name.lower()) &
            (df['position'] != 'team')
        ]
        
        if champ.empty:
            return f"查無 {champion_name} 的賽事資料。"

        total = champ.shape[0]
        win = champ[champ['result'] == 1].shape[0]
        rate = (win / total) * 100 if total > 0 else 0
        main = champ['position'].value_counts().idxmax()

        return f"[Champion Stats]\n英雄: {champion_name}\n出場: {total}\n勝率: {rate:.1f}%\n主要路線: {main}"
        
    except Exception as e:
        return f"數據運算發生錯誤: {e}"


@tool
def get_team_esports_stats(team_name: str) -> str:
    """
    查詢特定「職業隊伍」在職業賽事中的總場數與勝率。
    何時使用：當使用者詢問「特定職業隊伍」（如 T1, Gen.G, PSG 等）的比賽表現或勝率時。
    注意：如果使用者詢問的是「英雄」，請絕對不要使用此工具，改用 search_champion_patch_notes 或 get_champion_stats。
    """
    if 'df' not in globals() or df.empty:
        return "系統警告：賽事資料庫未載入，請確認 csv 檔案狀態。"
    
    try:
        team_df = df[
            (df['teamname'].str.lower() == team_name.lower()) &
            (df['position'] == 'team')
        ]
        
        if team_df.empty:
            return f"查無職業隊伍 {team_name} 的賽事資料。"

        total = team_df.shape[0]
        win = team_df[team_df['result'] == 1].shape[0]
        rate = (win / total) * 100 if total > 0 else 0

        return f"[Team Esports Stats]\n隊伍: {team_name}\n總場數: {total}\n勝率: {rate:.1f}%"

    except Exception as e:
        return f"隊伍數據運算發生錯誤: {e}"
    
# ==============================
# 3. Agent
# ==============================

# 修正 1：更新工具列表，使其與第二階段重構的函式名稱完全一致
tools = [
    search_patch_overview,
    search_champion_patch_notes,
    get_champion_esports_stats,
    get_team_esports_stats,
]

model = init_chat_model(
    "groq:openai/gpt-oss-120b",
    temperature=0
)

memory = MemorySaver()

# 💡 精簡且聚焦的 System Prompt
SYSTEM_PROMPT = """
你是一個專業的英雄聯盟 (LOL) 數據分析師 Agent。
你的任務是透過工具檢索最新版本改動、職業賽事數據與英雄對戰勝率，並提供精確、客觀的分析。

【絕對遵守的核心規則】
1. 語言轉換：所有工具的 `champion_name` 或 `team_name` 參數必須使用「英文官方名稱」（例如：將使用者輸入的「李星」轉換為 "Lee Sin"、「T1」保持 "T1"）。
2. 工具至上：如果工具回傳「系統警告」或「查無資料」，請直接告知使用者，嚴禁自行捏造數據或版本內容。
3. 職責邊界：禁止提供任何玩家操作技巧或出裝教學。

【回應結構指南】
請根據使用者詢問的內容類型，採用以下結構回答（若問題包含多個類型，請組合使用）：

- 關於「整體版本更新」：
  一句話總結改動核心，列出 Buff/Nerf 英雄清單（標示大致改動方向即可），並概述系統與裝備改動。

- 關於「單一英雄版本更新」：
  明確指出是 Buff 還是 Nerf。盤點檢索到的資料，逐一列出被動、Q、W、E、R 的技能數值變動，必須精確還原數據變化（如：傷害 50 => 60）。嚴禁遺漏檢索到的技能。

- 關於「職業賽場數據 (勝率/出場數)」：
  提供該英雄或隊伍的總出場數、具體勝率。若是英雄，需額外提供其主要路線。

"""

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

# ==============================
# 4. 執行與測試
# ==============================

def run_agent(query: str, session_id: str = "default_session") -> str:
    """
    執行 LOL Agent 並回傳分析結果。
    """
    try:
        # 修正 1：移除 messages 陣列中的 system prompt，只傳遞 user 的問題
        # 框架的記憶體機制會自動管理上下文
        result = agent.invoke(
            {
                "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            },
            config={"configurable": {"thread_id": session_id}}
        )

        # 修正 2：從回傳的狀態字典中，提取最後一則訊息 (也就是 Agent 的回答)
        final_message = result["messages"][-1].content
        return final_message

    except Exception as e:
        return f"Agent 執行時發生錯誤: {e}"

# ==============================
# 5. 主程式
# ==============================

if __name__ == "__main__":
    # 動態生成這次測試的 session ID
    my_session = f"terminal_session_{int(time.time())}"
    
    # 💡 在這裡測試中文輸入與複合型問題！
    query1 = "我要Galio的新版本改動。"
    
    print(f"正在分析您的問題：{query1}\n" + "-"*40)
    
    # 呼叫run_agent
    result1 = run_agent(query1, session_id=my_session)
    
    print(result1)
