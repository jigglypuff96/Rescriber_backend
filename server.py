from flask import Flask, request, jsonify
from flask_cors import CORS  
import ollama

app = Flask(__name__)
CORS(app) 

models = {
    "detect": {
        "modelName": "detectModel",
        "prompt": """You are an expert in cybersecurity and data privacy. You are now tasked to detect PII from the given text, using the following taxonomy only:
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
        Return me ONLY a json object in the following format (no other extra text!): {"results": [{"entity_type": YOU_DECIDE_THE_PII_TYPE, "text": PART_OF_MESSAGE_YOU_IDENTIFIED_AS_PII]}"""
    },
    "cluster": {
        "modelName": "clusterModel",
        "prompt": """For the given message, find ALL segments of the message with the same contextual meaning as the given PII. Consider segments that are semantically related or could be inferred from the original PII or share a similar context or meaning. List all of them in a list, and each segment should only appear once in each list.  Please return only in JSON format. Each PII provided will be a key, and its value would be the list PIIs (include itself) that has the same contextual meaning.

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
        {'Bill Gates': ['Bill Gates', 'Microsoft'], 'jeremyKwon@gmail.com':['jeremyKwon@gmail.com']}"""
    },
    "abstract": {
        "modelName": "abstractModel",
        "prompt": """Rewrite the text to abstract the protected information, and don't change other parts. Please return with JSON format. 
        For example if the input is:
        <Text>I graduated from CMU, and I earn a six-figure salary now. Today in the office, I had some conflict with my boss, and I am thinking about whether I should start interviewing with other companies to get a better offer.</Text>
        <ProtectedInformation>CMU, Today</ProtectedInformation>
        Then the output JSON format should be: {"results": YOUR_REWRITE} where YOUR_REWRITE needs to be a string that no longer contains ProtectedInformation, here's a sample YOUR_REWRITE: I graduated from a prestigious university, and I earn a six-figure salary now. Recently in the office, I had some conflict with my boss, and I am thinking about whether I should start interviewing with other companies to get a better offer."""
    }
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
        ollama.create(model=model_info["modelName"], modelfile=modelfile)
        print(f"Model '{model_info['modelName']}' has been created successfully.")
    except ollama.ResponseError as e:
        print(f"Failed to create the model: {e}")

@app.route('/detect', methods=['POST'])
def detect():
    user_message = request.json.get('message')
    try:
        print("Waiting for DETECT response...")
        response = ollama.chat(
            model=models["detect"]["modelName"],
            messages=[{'role': 'user', 'content': user_message}],
            stream=True,
            format ="json"
        )
        results = []
        for chunk in response:
            results.append(chunk['message']['content'])
        return jsonify({"results": ''.join(results)})
    except ollama.ResponseError as e:
        print("Error running Ollama:", e)
        return jsonify({"error": "Error running Ollama", "details": str(e)}), 500

@app.route('/cluster', methods=['POST'])
def cluster():
    user_message = request.json.get('message')
    try:
        print("Waiting for CLUSTER response...")
        response = ollama.chat(
            model=models["cluster"]["modelName"],
            messages=[{'role': 'user', 'content': user_message}],
            stream=True,
            format ="json"
        )
        results = []
        for chunk in response:
            results.append(chunk['message']['content'])
        return jsonify({"results": ''.join(results)})
    except ollama.ResponseError as e:
        print("Error running Ollama:", e)
        return jsonify({"error": "Error running Ollama", "details": str(e)}), 500

@app.route('/abstract', methods=['POST'])
def abstract():
    user_message = request.json.get('message')
    try:
        print("Waiting for ABSTRACT response...")
        response = ollama.chat(
            model=models["abstract"]["modelName"],
            messages=[{'role': 'user', 'content': user_message}],
            stream=True,
            format ="json"
        )
        results = []
        for chunk in response:
            results.append(chunk['message']['content'])
        return jsonify({"results": ''.join(results)})
    except ollama.ResponseError as e:
        print("Error running Ollama:", e)
        return jsonify({"error": "Error running Ollama", "details": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "HI"

if __name__ == "__main__":
    pull_model("llama3")
    create_model("detect")
    create_model("cluster")
    create_model("abstract")
    app.run(port=3000)
