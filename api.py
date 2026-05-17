import os
import requests
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 從環境變數讀取 API Key，而不是寫死在程式碼裡
api_key = os.environ.get("GROQ_API_KEY")

url = "https://api.groq.com/openai/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}"
}

# 發送請求給 Groq 伺服器
response = requests.get(url, headers=headers)

# 印出所有可用的模型名稱
print("--- 支援的模型清單 ---")
for model in response.json().get("data", []):
    print(model["id"])