import os
import pandas as pd
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ==============================
# 1. 初始化
# ==============================
from update_patch_db import ensure_latest_patch_in_db

load_dotenv()

ensure_latest_patch_in_db()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ 載入賽事數據
try:
    df = pd.read_csv("oracles_match_data.csv", low_memory=False)
    df['champion'] = df['champion'].astype(str)
    df['teamname'] = df['teamname'].astype(str)
    print(f"✅ 成功載入賽事數據，共 {len(df)} 筆。")
except:
    print("❌ 找不到 csv")
    df = pd.DataFrame()


# ==============================
# 2. Tools
# ==============================

@tool
def search_patch_overview(version: str) -> str:
    """查詢某個版本的完整改動"""
    try:
        if os.path.exists("./patch_db"):
            db = Chroma(
                persist_directory="./patch_db",
                embedding_function=embeddings
            )

            docs = db.similarity_search(
                query=f"patch {version}",
                k=5,
                filter={"patch": version}
            )

            rag = "\n".join([d.page_content for d in docs])
        else:
            rag = "無本地資料"
        return f"""

[Patch {version} Overview]

本地資料：
{rag}

請整理：
1. 版本重點
2. Buff / Nerf 英雄
3. 系統 / 裝備改動
"""
    except Exception as e:
        return f"錯誤: {e}"


@tool
def search_patch_notes(champion_name: str) -> str:
    """查詢某英雄 patch 改動"""
    try:
        if not os.path.exists("./patch_db"):
            return "❌ 無 patch DB"

        db = Chroma(
            persist_directory="./patch_db",
            embedding_function=embeddings
        )

        docs = db.similarity_search(
            query=f"{champion_name} buffs nerfs abilities patch",
            k=3,
            filter={"type": "champion"}
        )

        rag = "\n".join([d.page_content for d in docs]) if docs else "無本地資料"

        return f"""
[Patch Notes]

本地資料：
{rag}

請整理：
- buff / nerf
- 技能改動
"""
    except Exception as e:
        return f"錯誤: {e}"


@tool
def get_champion_stats(champion_name: str) -> str:
    """查詢職業勝率"""
    if df.empty:
        return "資料庫未載入"

    champ = df[
        (df['champion'].str.contains(champion_name, case=False, na=False)) &
        (df['position'] != 'team')
    ]

    if champ.empty:
        return "查無資料"

    total = champ.shape[0]
    win = champ[champ['result'] == 1].shape[0]
    rate = (win / total) * 100 if total > 0 else 0

    main = champ['position'].value_counts().idxmax()

    return f"""
[Champion Stats]

英雄: {champion_name}
出場: {total}
勝率: {rate:.1f}%
主要路線: {main}
"""


@tool
def search_lol_counters(champion_name: str) -> str:
    """查詢 counter"""
    try:
        web = DuckDuckGoSearchRun().run(
            f"{champion_name} counters win rate op.gg"
        )

        return f"""
[Counter Data]

{web}

請整理：
- counter 英雄
- 勝率
- 擊殺率
"""
    except Exception as e:
        return f"錯誤: {e}"


# ==============================
# 3. Agent
# ==============================

tools = [
    get_champion_stats,
    search_lol_counters,
    search_patch_notes,
    search_patch_overview
]

model = init_chat_model(
    "groq:openai/gpt-oss-120b",
    temperature=0
)

memory = MemorySaver()

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)


SYSTEM_PROMPT = """
你是一個職業級 LOL 數據分析師

【任務】
1. 分析版本更新
2. 分析英雄強度與 counter

【規則】
- 必須使用 tool
- 不可亂猜

【輸出】

版本：
1. 版本重點
2. Buff / Nerf
3. 系統改動
4. Bug修復

英雄：
1. Counter picks
2. win rate
3. patch 影響

禁止：
❌ 操作教學
❌ 出裝教學
"""


# ==============================
# 4. 執行
# ==============================

def normalize_query(q: str):
    if "版本" in q:
        q = q.replace("版本", "patch ")
    return q


def run_agent(query: str):
    query = normalize_query(query)

    return agent.invoke(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        },
        config={"configurable": {"thread_id": "lol_session"}}
    )


# ==============================
# 5. 主程式
# ==============================

if __name__ == "__main__":

    # ✅ 測試 query（你可以改這裡）
    query = "請給我26.10版本的Ash更新?"

    result = run_agent(query)

    print("\n=== 分析結果 ===\n")
    print(result["messages"][-1].content)