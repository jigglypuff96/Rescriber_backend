from flask import Flask, request, jsonify
from flask_cors import CORS  
import ollama
import json
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine
import pandas as pd
from itertools import combinations

app = Flask(__name__)
CORS(app) 
global_base_model = "llama3"
all_entities = []
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
        ollama.create(model=global_base_model, modelfile=modelfile)
        print(f"Model '{model_info['modelName']}' has been created successfully.")
    except ollama.ResponseError as e:
        print(f"Failed to create the model: {e}")

def extract_embedding(embedding_dict_str):
    try:
        embedding_dict = eval(embedding_dict_str)
        embedding = embedding_dict.get('embedding', [])
        if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
            return embedding
        else:
            raise ValueError("Extracted embedding is not a list of numbers.")
    except Exception as e:
        raise ValueError(f"Error extracting embedding: {e}")

def calculate_distance(embedding1, embedding2):
    return cosine(embedding1, embedding2)

def find_nearest_entity(df, input_entity):
    input_embedding_str = df[df['name'] == input_entity]['embedding'].values[0]
    input_embedding = extract_embedding(input_embedding_str)
    
    nearest_entity = None
    nearest_distance = float('inf')
    
    for index, row in df.iterrows():
        if row['name'] != input_entity:
            entity_embedding_str = row['embedding']
            entity_embedding = extract_embedding(entity_embedding_str)
            distance = calculate_distance(input_embedding, entity_embedding)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_entity = row['name']
    return nearest_entity

def create_union_set(df):
    """
# Often, it's necessary to find the nearest elements, but if there are only a few, it's actually not needed.
# Nearest elements:
#     {frozenset({'Vanderbilt University', 'Vanderbilt'}), frozenset({'Vandy', 'Vanderbilt'}), frozenset({'Vanderbilt', 'VU'})}
# Valid items: ['Vanderbilt University', 'Vanderbilt']
# NLTK NER response:
#     {'results': [{'entity_type': 'ORGANIZATION', 'text': 'Vanderbilt University'},
#                  {'entity_type': 'ORGANIZATION', 'text': 'Vandy'},
#                  {'entity_type': 'ORGANIZATION', 'text': 'Vanderbilt'},
#                  {'entity_type': 'ORGANIZATION', 'text': 'VU'}]}
# It should directly form pairs.
"""

    union_set = set()
    for input_entity in df['name']:
        nearest_entity = find_nearest_entity(df, input_entity)
        if nearest_entity:
            union_set.add(frozenset([input_entity, nearest_entity]))
    
    return union_set


def is_valid_pair(e1, e2):
    if len(e1) > len(e2):
        t1, t2 = e1, e2
    else:
        t1, t2 = e2, e1
    
    t1 = t1.lower()
    t2 = t2.lower()
    it = iter(t1)
    return all(c in it for c in t2) 

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)

        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

def ufcluster(nltk_entities, csv_path='entity_embeddings.csv'):
    pairs=[]

    #change if-else logic and threshold

    entity_types_count = len(set([entity['entity_type'] for entity in nltk_entities]))
    if entity_types_count <= 2 and len(nltk_entities) < 20:
        pairs.extend(list(combinations([entity['text'] for entity in nltk_entities], 2)))
    else:   
        df = pd.read_csv(csv_path)
        # embeddings = np.array([extract_embedding(embedding) for embedding in df['embedding']])
        pairs.extend(list(create_union_set(df)))
        
    if 1< entity_types_count < 5 and len(nltk_entities) < 100:
        additional_pairs = list(combinations([entity['text'] for entity in nltk_entities], 2))
        print("Additional pairs based on conditions:")
        print(additional_pairs)
        pairs.extend(additional_pairs)
    
    print(pairs)


    
    valid_pairs = [] 
    invalid = []

    for pair in pairs:
        items = list(pair) 
        print("here")
        print(items)
        if is_valid_pair(items[0], items[1]):
            valid_pairs.append(items) 
            print("valid items",items)  
        else:
            invalid.extend(items)
            print("invalid",items)

    uf = UnionFind()

    for pair in valid_pairs:
        items = list(pair)
        uf.add(items[0])
        uf.add(items[1])
        uf.union(items[0], items[1])

    for e in invalid:
        uf.add(e)

    union_sets = {}
    for item in uf.parent:
        root = uf.find(item)
        if root not in union_sets:
            union_sets[root] = []
        union_sets[root].append(item)
    
    result = {}
    
    for k,v in union_sets.items():
        if len(v)>4:
            continue
        else:
            result[k]=v

    
    return result

@app.route('/update-entities', methods=['POST'])
def update_entities():
    #It should be named all entities
    global nltk_entities
    new_entities = request.json.get('entities', [])
    # all_entities.extend(new_entities)
    nltk_entities = new_entities
    return jsonify({"message": "Entities updated successfully", "all_entities": nltk_entities})


@app.route('/clusteruf', methods=['POST'])
def clusteruf():
    global nltk_entities
    try:
        if not nltk_entities:
            return jsonify({"error": "No NLTK entities found. Please run the NER endpoint first."}), 400

        # entities_text = [entity['text'] for entity in nltk_entities]
        result = ufcluster(nltk_entities)
        print("cluser2")
        print(result)
        return jsonify({"results": result})
    except Exception as e:
        print("Error processing clustering:", e)
        return jsonify({"error": "Error processing clustering", "details": str(e)}), 500


from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk


# Ensure you have the necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')



def extract_entities(nltk_tree):
    entities = []
    for subtree in nltk_tree:
        if isinstance(subtree, nltk.Tree):
            entity_type = subtree.label()
            entity = " ".join([word for word, pos in subtree.leaves()])
            entities.append({"entity_type": entity_type, "text": entity})
    return entities

def get_embedding(name):
    start_time = datetime.now()
    embedding = ollama.embeddings(model='llama3', prompt=name)
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    return embedding, duration


@app.route('/detect', methods=['POST'])
def detect():
    user_message = request.json.get('message')
    try:
        print("Waiting for DETECT response...")
        response = ollama.chat(
            model=global_base_model,
            messages=[{'role': 'user', 'content': user_message},
                      {'role': 'system','content': models["detect"]["prompt"]}
                      ],
            stream=True,
            format="json",

        )
        results = []
        for chunk in response:
            results.append(chunk['message']['content'])
        return jsonify({"results": ''.join(results)})
    except ollama.ResponseError as e:
        print("Error running Ollama:", e)
        return jsonify({"error": "Error running Ollama", "details": str(e)}), 500

@app.route('/nltk-ner', methods=['POST'])
def nltk_ner():
    global nltk_entities
    user_message = request.json.get('message')
    try:
        # Tokenize the text
        words = word_tokenize(user_message)
        
        # POS tagging
        pos_tags = pos_tag(words)
        
        # Named Entity Recognition
        named_entities = ne_chunk(pos_tags)
        
        # Extract entities
        nltk_entities = extract_entities(named_entities)
        
        return jsonify({"results": nltk_entities})
    except Exception as e:
        print("Error running NLTK NER:", e)
        return jsonify({"error": "Error running NLTK NER", "details": str(e)}), 500

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    global nltk_entities
    try:
        if not nltk_entities:
            return jsonify({"error": "No NLTK entities found. Please run the NER endpoint first."}), 400
        
        embedding_data = []
        
        for entity in nltk_entities:
            name = entity['text']
            embedding, query_time = get_embedding(name)
            embedding_data.append({
                "name": name,
                "embedding": embedding,
                "query_time": query_time
            })
        

        df = pd.DataFrame(embedding_data)
        output_path = "entity_embeddings.csv"
        df.to_csv(output_path, index=False)
        
        return jsonify({"message": f"Embeddings saved to {output_path}"})
    except Exception as e:
        print("Error generating embeddings:", e)
        return jsonify({"error": "Error generating embeddings", "details": str(e)}), 500


@app.route('/cluster', methods=['POST'])
def cluster():
    user_message = request.json.get('message')
    try:
        print("Waiting for CLUSTER response...")
        response = ollama.chat(
            model=global_base_model,
            messages=[{'role': 'user', 'content': user_message},
                      {'role': 'system','content': models["cluster"]["prompt"]}
                      ],
            stream=True,
            format ="json",
            # system = models["cluster"]["prompt"]
        )
        results = []
        for chunk in response:
            results.append(chunk['message']['content'])
        return jsonify({"results": ''.join(results)})
    except ollama.ResponseError as e:
        print("Error running Ollama:", e)
        return jsonify({"error": "Error running Ollama", "details": str(e)}), 500

merge_clustering_response = {}

@app.route('/update-cluster-results', methods=['POST'])
def update_cluster_results():
    global merge_clustering_response
    new_results = request.json.get('results', {})
    for key, value in new_results.items():
        if key in merge_clustering_response:
            merge_clustering_response[key] = list(set(merge_clustering_response[key] + value))
        else:
            merge_clustering_response[key] = value
    return jsonify({"message": "Clustering results updated successfully", "merge_clustering_response": merge_clustering_response})


@app.route('/', methods=['GET'])
def home():
    return "HI"

if __name__ == "__main__":
    pull_model("llama3")

    

    app.run(port=3000)