from flask import Flask, request, jsonify
from flask_cors import CORS  
import ollama
import json
# import jsonify
import numpy as np
from datetime import datetime
# from scipy.spatial.distance import cosine
import pandas as pd
from itertools import combinations
import ssl
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


app = Flask(__name__)
CORS(app) 
global_base_model = "llama3"

http_port = 3000
https_port = 3443

models = {
    "detect": {
        "modelName": "detectModel",
        "prompt": '''You are an expert in cybersecurity and data privacy. You are now tasked to detect PII from the given text, using the following taxonomy only:
        ADDRESS
        IP_ADDRESS
        URL
        SSN
        PHONE_NUMBER
        EMAIL
        DRIVERS_LICENSE
        PASSPORT_NUMBER
        TAXPAYER IDENTIFICATION NUMBER
        ID_NUMBER
        NAME
        USERNAME
        KEYS: Passwords, passkeys, API keys, encryption keys, and any other form of security keys.
        GEOLOCATION: Places and locations, such as cities, provinces, countries, international regions, or named infrastructures (bus stops, bridges, etc.).
        AFFILIATION: Names of organizations, such as public and private companies, schools, universities, public institutions, prisons, healthcare institutions, non-governmental organizations, churches, etc.
        DEMOGRAPHIC_ATTRIBUTE: Demographic attributes of a person, such as native language, descent, heritage, ethnicity, nationality, religious or political group, birthmarks, ages, sexual orientation, gender and sex.
        TIME: Description of a specific date, time, or duration.
        HEALTH_INFORMATION: Details concerning an individual's health status, medical conditions, treatment records, and health insurance information.
        FINANCIAL_INFORMATION: Financial details such as bank account numbers, credit card numbers, investment records, salary information, and other financial statuses or activities.
        EDUCATIONAL_RECORD: Educational background details, including academic records, transcripts, degrees, and certification.
        For the given message that a user sends to a chatbot, identify all the personally identifiable information using the above taxonomy only, and the entity_type should be selected from the all-caps categories.
        Note that the information should be related to a real person not in a public context, but okay if not uniquely identifiable.
        Result should be in its minimum possible unit.
        Return me ONLY a json object in the following format (no other extra text!): {"results": [{"entity_type": YOU_DECIDE_THE_PII_TYPE, "text": PART_OF_MESSAGE_YOU_IDENTIFIED_AS_PII]}'''
    },
    "cluster": {
        "modelName": "clusterModel",
        "prompt": '''For the given message, find ALL segments of the message with the same contextual meaning as the given PII. Consider segments that are semantically related or could be inferred from the original PII or share a similar context or meaning. List all of them in a list, and each segment should only appear once in each list.  Please return only in JSON format. Each PII provided will be a key, and its value would be the list PIIs (include itself) that has the same contextual meaning.

        Example 1:
        Input:
        <message>I will be the valedictorian of my class. Please write me a presentation based on the following information: As a student at Vanderbilt University, I feel honored. The educational journey at Vandy has been nothing less than enlightening. The dedicated professors here at Vanderbilt are the best. As an 18 year old student at VU, the opportunities are endless.</message>
        <pii1>Vanderbilt University</pii1>
        <pii2>18 year old</pii2>
        <pii3>VU</pii3>
        Expected JSON output:
        {'Vanderbilt University': ['Vanderbilt University', 'Vandy', 'VU', 'Vanderbilt'], '18 year old':['18 year old'], 'VU':[ 'VU', 'Vanderbilt University', 'Vandy', 'Vanderbilt']}
        
        Example 2:
        Input:
        <message>Do you know Bill Gates and the company he founded, Microsoft? Can you send me an article about how he founded it to my email at jeremyKwon@gmail.com please?</message>
        <pii1>Bill Gates</pii1>
        <pii2>jeremyKwon@gmail.com</pii2>
        Expected JSON output:
        {'Bill Gates': ['Bill Gates', 'Microsoft'], 'jeremyKwon@gmail.com':['jeremyKwon@gmail.com']}'''
    },
    "abstract": {
        "modelName": "abstractModel",
        "prompt": '''Rewrite the text to abstract the protected information, and don't change other parts. Please return with JSON format. 
        For example if the input is:
        <Text>I graduated from CMU, and I earn a six-figure salary now. Today in the office, I had some conflict with my boss, and I am thinking about whether I should start interviewing with other companies to get a better offer.</Text>
        <ProtectedInformation>CMU, Today</ProtectedInformation>
        Then the output JSON format should be: {"results": YOUR_REWRITE} where YOUR_REWRITE needs to be a string that no longer contains ProtectedInformation, here's a sample YOUR_REWRITE: I graduated from a prestigious university, and I earn a six-figure salary now. Recently in the office, I had some conflict with my boss, and I am thinking about whether I should start interviewing with other companies to get a better offer.'''
    },
}

base_options = {
    "seed": 40,
    "num_predict": 100,
    "top_k": 20,
    "top_p": 0.9,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": True,
    "stop": ["\n", "user:"]
}
def pull_model(model_name):
    try:
        ollama.pull(model_name)
        print(f"Model '{model_name}' has been pulled successfully.")
    except ollama.ResponseError as e:
        print(f"Failed to pull the model: {e}")


def create_model(model_key):
    model_info = models[model_key]
    modelfile = f"""
    FROM llama3
    SYSTEM "{model_info['prompt']}"
    """
    try:
        ollama.create(model=global_base_model, modelfile=modelfile)
        print(f"Model '{model_info['modelName']}' has been created successfully.")
    except ollama.ResponseError as e:
        print(f"Failed to create the model: {e}")
    

@app.route('/detect', methods=['POST'])
def detect():
    user_message = request.json.get('message')
    try:
        print("Waiting for DETECT response...")
        response = ollama.chat(
            model=global_base_model,
            messages=[{'role': 'user', 'content': user_message},
                      {'role': 'system', 'content': models["detect"]["prompt"]}
                      ],
            stream=True,
            format="json",
             options=base_options
        )
        # options=base_options
        print("detect response ready")
        results = []
        for chunk in response:
            print ("chunk content: ",chunk['message']['content'])
            results.append(chunk['message']['content'])
        combined_results = ''.join(results)
        print(combined_results)
        return jsonify(combined_results)
    except ollama.ResponseError as e:
        print("Error running Ollama:", e)
        return jsonify({"error": "Error running Ollama", "details": str(e)}), 500

# @app.route("/cluster", methods=["POST"])
# async def cluster():
#     user_message = request.json.get('message')

#     try:
#         print("Waiting for CLUSTER response...")
#         response = await ollama.chat({
#             "model": models["cluster"]["modelName"],
#             "messages": [{"role": "user", "content": user_message}],
#             "format": "json",
#         })

#         return jsonify({"results": response["message"]["content"]})
#         print("CLUSTER response sent.")
#     except Exception as error:
#         print("Error running Ollama:", error)
#         return jsonify({"error": "Error running Ollama", "details": str(error)}), 500

@app.route("/abstract", methods=["POST"])
async def abstract():
    user_message = request.json.get('message')

    try:
        print("Waiting for ABSTRACT response...")
        response = ollama.chat(
            model=global_base_model,
            messages=[{'role': 'user', 'content': user_message},
                      {'role': 'system', 'content': models["abstract"]["prompt"]}
                      ],
            stream=True,
            format="json",
             options=base_options
        )
        print("ABSTRACT response sent.")
        return jsonify({"results": response["message"]["content"]})
    except Exception as error:
        print("Error running Ollama:", error)
        return jsonify({"error": "Error running Ollama", "details": str(error)}), 500

@app.route("/openaiapikey", methods=["GET"])
async def get_openai_api_key():
    if openai_api_key:
        return jsonify({"apiKey": openai_api_key}), 200
    else:
        return jsonify({"error": "API key not found"}), 500


@app.route("/", methods=["GET"])
async def home():
    print("Get request received!")
    return "HI"

if __name__ == "__main__":
    pull_model("llama3")
    app.run(host='127.0.0.1', port=3000)

