from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import json
import time
import threading

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
global_base_model = "llama3"

# System prompts
system_prompts = {
    "detect": '''You an expert in cybersecurity and data privacy. You are now tasked to detect PII from the given text, using the following taxonomy only:

  ADDRESS
  IP_ADDRESS
  URL
  SSN
  PHONE_NUMBER
  EMAIL
  DRIVERS_LICENSE
  PASSPORT_NUMBER
  TAXPAYER_IDENTIFICATION_NUMBER
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
    Return me ONLY a json in the following format: {"results": [{"entity_type": YOU_DECIDE_THE_PII_TYPE, "text": PART_OF_MESSAGE_YOU_IDENTIFIED_AS_PII]}''',
    "abstract": '''Rewrite the text to abstract the protected information, without changing other parts. For example:
        Input: <Text>I graduated from CMU, and I earn a six-figure salary. Today in the office...</Text>
        <ProtectedInformation>CMU, Today</ProtectedInformation>
        Output JSON: {"results": "I graduated from a prestigious university, and I earn a six-figure salary. Recently in the office..."} Please use "results" as the only key in the JSON object.'''
}

# Ollama options
base_options = {"format": "json", "temperature": 0}


def split_into_chunks(input_text, chunk_size=100):
    """Split a string into chunks of a specific size."""
    words = input_text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def get_response(model_name, system_prompt, user_message):
    """Send a request to the Ollama model and return its response."""
    response = ollama.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ],
        format="json",
        stream=False,
        options=base_options
    )
    return response['message']['content']


def process_request(model_name, prompt, input_text, extend_results=False):
    """Process a request by sending input text in chunks to the Ollama model."""
    chunks = split_into_chunks(input_text)
    results = []

    for chunk in chunks:
        try:
            message_content = get_response(model_name, prompt, chunk)
            if not message_content.strip():
                print("Empty message content received.")
                continue

            parsed_content = json.loads(message_content)
            if "results" in parsed_content:
                if extend_results:
                    results.extend(parsed_content["results"])
                else:
                    results.append(parsed_content["results"])
            else:
                print("Unexpected response format:", parsed_content)
        except Exception as e:
            print(f"Error processing chunk: {chunk}\nError: {e}")

    return results


@app.route('/detect', methods=['POST'])
def detect():
    """Handle the detect endpoint."""
    print ("Detect request received!")
    return handle_request(global_base_model, system_prompts["detect"], extend_results=True)


@app.route('/abstract', methods=['POST'])
def abstract():
    """Handle the abstract endpoint."""
    print ("Abstract request received!")
    response = handle_request(global_base_model, system_prompts["abstract"], extend_results=False)
    # Flatten results and join them for consistent abstract format
    response_data = response.json
    if "results" in response_data:
        response_data["results"] = ' '.join(response_data["results"])
    return jsonify(response_data)


def handle_request(model_name, prompt, extend_results):
    """Handle a generic request for either 'detect' or 'abstract'."""
    data = request.get_json()
    input_text = data.get('message', '')
    print ("INPUT TEXT: ",input_text)

    if not input_text:
        return jsonify({"error": "No message provided"}), 400

    start_time = time.time()
    try:
        results = process_request(model_name, prompt, input_text, extend_results)
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500
    end_time = time.time()

    return jsonify({"results": results, "processing_time": end_time - start_time})


def initialize_server(test_message):
    """Simulate an initial detect request internally to initialize the model."""
    print("Initializing server with test message...")
    try:
        start_time = time.time()
        results = process_request(global_base_model, system_prompts['detect'], test_message, extend_results=True)
        end_time = time.time()
        print("Initialization complete. Now you can start using the tool!")
        print(f"Results: {results}\nProcessing time: {end_time - start_time}")
    except Exception as e:
        print(f"Error initializing server: {str(e)}")


if __name__ == "__main__":
    # Start server initialization in a separate thread
    test_message = "Hi, welcome to Rescriber!"
    print("Processing initial detect request...")
    threading.Thread(target=initialize_server, args=(test_message,), daemon=True).start()

    # Start Flask server
    app.run(
        host="0.0.0.0",
        port=5331,
        ssl_context=('python_cert/selfsigned.crt', 'python_cert/selfsigned.key')
    )
