from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://127.0.0.1:11434"

# Ollama 调用函数，支持 system prompt 和流式响应
def call_ollama(model_name, system_prompt, user_prompt):
    try:
        url = f"{OLLAMA_URL}/api/chat"
        headers = {"Content-Type": "application/json"}
        print ("system_prompt = ",system_prompt)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=payload, stream=True)

        # 处理流式响应
        result = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
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
    print('user_message = ', user_message)
    system_prompt = (
        "You are an expert in cybersecurity and data privacy. Detect PII from the given text using "
        "the following categories: ADDRESS, IP_ADDRESS, URL, SSN, PHONE_NUMBER, EMAIL, DRIVERS_LICENSE, "
        "PASSPORT_NUMBER, TAXPAYER IDENTIFICATION NUMBER, ID_NUMBER, NAME, USERNAME, KEYS, GEOLOCATION, "
        "AFFILIATION, DEMOGRAPHIC_ATTRIBUTE, TIME, HEALTH_INFORMATION, FINANCIAL_INFORMATION, "
        "EDUCATIONAL_RECORD. Return only a JSON object with identified PII."
    )
    result = call_ollama("llama3", system_prompt, user_message)
    return jsonify({"results": result})

# API 路由：PII 聚类
@app.route("/cluster", methods=["POST"])
def cluster():
    user_message = request.json.get("message", "")
    system_prompt = (
        "For the given message, find all segments of the message with the same contextual meaning as the given PII. "
        "List them in a JSON format where each PII is a key and its value is a list of similar contextual segments."
    )
    result = call_ollama("llama3", system_prompt, user_message)
    return jsonify({"results": result})

# API 路由：PII 抽象
@app.route("/abstract", methods=["POST"])
def abstract():
    user_message = request.json.get("message", "")
    system_prompt = (
        "Rewrite the text to abstract the protected information without changing other parts. "
        "Return the rewritten text in JSON format."
    )
    result = call_ollama("llama3", system_prompt, user_message)
    return jsonify({"results": result})

# 健康检查 API
@app.route("/", methods=["GET"])
def health_check():
    return "Ollama API is running on http://127.0.0.1:11434", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
