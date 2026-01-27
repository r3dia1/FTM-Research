import os
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

SAVE_DIR = "collected_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 全域變數，紀錄目前正在寫入的檔案路徑
current_file_path = None

@app.route('/collect', methods=['POST'])
def collect_data():
    global current_file_path
    
    try:
        payload = request.json
        msg_type = payload.get('type')     # header 或 data
        content = payload.get('content')   # CSV 字串 (包含換行符號)

        if not content:
            return jsonify({"status": "empty"}), 200

        # 如果收到 Header，代表是一次新的測量開始，建立新檔案
        if msg_type == 'header':
            date_str = time.strftime("%Y%m%d_%H%M%S")
            filename = f"Server_Wide_{date_str}.csv"
            current_file_path = os.path.join(SAVE_DIR, filename)
            print(f"--- 新測量開始，建立檔案: {filename} ---")
            
            # 寫入 Header
            with open(current_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # 如果是 Data，且已經有檔案，就附加寫入
        elif msg_type == 'data' and current_file_path:
            with open(current_file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            # print(f"寫入一筆資料: {content.strip()}") # 測試時可開啟

        else:
            print("收到資料但尚未初始化 Header，略過")

        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # host='0.0.0.0' 允許區網連線
    app.run(host='0.0.0.0', port=5000)