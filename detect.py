from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://127.0.0.1:11434"

# 修改后的 Ollama 调用函数，支持流式响应
def call_ollama(model_name, user_prompt):
    try:
        url = f"{OLLAMA_URL}/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "prompt": user_prompt
        }
        response = requests.post(url, headers=headers, json=payload, stream=True)

        # 逐步处理流式响应
        result = ""
        for line in response.iter_lines():
            if line:
                # 将字节数据解码为字符串
                decoded_line = line.decode('utf-8')
                # 尝试解析为 JSON 格式
                try:
                    data = json.loads(decoded_line)
                    result += data.get("response", "")
                except Exception as e:
                    print(f"Error parsing line: {decoded_line}, Error: {e}")

        return result
    except Exception as e:
        return f"Error calling Ollama API: {str(e)}"

# API 路由：检测 PII 信息
@app.route("/detect", methods=["POST"])
def detect():
    user_message = request.json.get("message", "")
    result = call_ollama("llama3", user_message)
    return jsonify({"results": result})

# 健康检查 API
@app.route("/", methods=["GET"])
def health_check():
    return "Ollama API is running on http://127.0.0.1:11434", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
