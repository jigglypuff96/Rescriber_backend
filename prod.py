from flask import Flask, request, jsonify
import ollama
import json
import time
from datetime import datetime

# Flask app setup
app = Flask(__name__)

# Ollama model configuration
global_base_model = "llama3"

system_prompts = {
    "detect": '''You are an expert in cybersecurity and data privacy. You are now tasked to detect PII from the given text, using the following taxonomy only:
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
        Return me ONLY a json object in the following format (no other extra text!): {"results": [{"entity_type": YOU_DECIDE_THE_PII_TYPE, "text": PART_OF_MESSAGE_YOU_IDENTIFIED_AS_PII]}''',
    "cluster": '''For the given message, find ALL segments of the message with the same contextual meaning as the given PII. Consider segments that are semantically related or could be inferred from the original PII or share a similar context or meaning. List all of them in a list, and each segment should only appear once in each list. Please return only in JSON format.''',
    "abstract": '''Rewrite the text to abstract the protected information, and don't change other parts. Please return with JSON format.'''
}

base_options = {
    "format": "json",
    "temperature": 0,
}


def split_into_chunks(input_text, chunk_size=100):
    """Split a string into chunks of a specific size."""
    words = input_text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def process_request(model_name, prompt, input_text):
    """Call Ollama with chunked inputs and combine results."""
    chunks = split_into_chunks(input_text)
    results = []

    for chunk in chunks:
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': chunk}
                ],
                format="json",
                stream=False,
                options=base_options
            )

            message_content = response['message']['content']
            parsed_content = json.loads(message_content)
            results.extend(parsed_content.get('results', []))
        except Exception as e:
            print("Error processing chunk:", e)
            continue

    return results


@app.route('/detect', methods=['POST'])
def detect():
    """Handle detect endpoint."""
    data = request.get_json()
    input_text = data.get('message', '')

    if not input_text:
        return jsonify({"error": "No message provided"}), 400

    print("Processing detect request...")

    start_time = time.time()
    results = process_request(global_base_model, system_prompts['detect'], input_text)
    end_time = time.time()

    return jsonify({
        "results": results,
        "processing_time": end_time - start_time
    })


@app.route('/cluster', methods=['POST'])
def cluster():
    """Handle cluster endpoint."""
    data = request.get_json()
    input_text = data.get('message', '')

    if not input_text:
        return jsonify({"error": "No message provided"}), 400

    print("Processing cluster request...")

    start_time = time.time()
    results = process_request(global_base_model, system_prompts['cluster'], input_text)
    end_time = time.time()

    return jsonify({
        "results": results,
        "processing_time": end_time - start_time
    })


@app.route('/abstract', methods=['POST'])
def abstract():
    """Handle abstract endpoint."""
    data = request.get_json()
    input_text = data.get('message', '')

    if not input_text:
        return jsonify({"error": "No message provided"}), 400

    print("Processing abstract request...")

    start_time = time.time()
    results = process_request(global_base_model, system_prompts['abstract'], input_text)
    end_time = time.time()

    return jsonify({
        "results": results,
        "processing_time": end_time - start_time
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        ssl_context=('selfsigned.crt', 'selfsigned.key')
    )
