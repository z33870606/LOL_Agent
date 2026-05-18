import pandas as pd
import gdown
import os

def update_oracles_data():
    """從 Google Drive 自動下載最新的比賽數據"""
    print("正在從 Google Drive 獲取最新賽事數據...")
    
    #Google Drive 檔案 ID
    file_id = "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm"
    
    download_url = f"https://drive.google.com/uc?id={file_id}"
    output_filename = "oracles_match_data.csv"
    
    try:
        # 1. 如果舊檔案存在，先刪除避免衝突
        if os.path.exists(output_filename):
            os.remove(output_filename)

        # 2. 使用 gdown 下載檔案
        print("下載中")
        gdown.download(download_url, output_filename, quiet=False)
        
        # 3. 用 Pandas 讀取本地端的 CSV 檔案來確認
        df = pd.read_csv(output_filename, low_memory=False)
        print(f"賽事資料庫更新完成！目前總資料筆數：{len(df)} 筆")
        
    except Exception as e:
        print(f"賽事資料更新失敗: {str(e)}")

if __name__ == "__main__":
    update_oracles_data()
    # update_vector_db()  # 更新 Patch Notes 向量資料庫的函數
