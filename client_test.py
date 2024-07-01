import requests
import threading
from time import sleep

base_url = "http://localhost:3000"
entities = []

# Lock to manage access to the shared `entities` variable
entities_lock = threading.Lock()

def post_request(endpoint, data):
    url = f"{base_url}/{endpoint}"
    response = requests.post(url, json=data)
    return response.json()

def detect_entities(user_message):
    global entities
    response = post_request('detect', {'message': user_message})
    print("Detect Response:", response)
    with entities_lock:
        entities.extend(response.get('results', []))
        print("Entities updated from detect_entities.")
        trigger_post_processing()

def nltk_ner(user_message):
    global entities
    response = post_request('nltk-ner', {'message': user_message})
    print("NLTK NER Response:", response)
    with entities_lock:
        entities.extend(response.get('results', []))
        print("Entities updated from nltk_ner.")
        trigger_post_processing()

def generate_embeddings():
    global entities
    print("Generating embeddings...")
    response = post_request('generate-embeddings', {})
    print("Generate Embeddings Response:", response)
    return response

def cluster_uf():
    print("Clustering UF...")
    response = post_request('clusteruf', {})
    print("Cluster UF Response:", response)
    return response

def cluster(user_message):
    print("Clustering...")
    response = post_request('cluster', {'message': user_message})
    print("Cluster Response:", response)
    return response

def trigger_post_processing():
    # Trigger the post-processing steps when entities are updated
    print("trigger post processing")
    generate_embeddings()
    cluster_uf()
    cluster(user_message)

def main(user_message):
    # Start the detect and nltk_ner functions in separate threads
    detect_thread = threading.Thread(target=detect_entities, args=(user_message,))
    nltk_ner_thread = threading.Thread(target=nltk_ner, args=(user_message,))

    detect_thread.start()
    nltk_ner_thread.start()

    detect_thread.join()
    nltk_ner_thread.join()

    # After both threads are done, notify that merging is complete
    # Todo: merge
    # Todo: manage csv create and delete
    print("Merging results from detect and nltk_ner.")



if __name__ == "__main__":
    # user_message = "Vanderbilt University is located in Nashville. Vandy is a great place to study."
    user_message= """I will be the valedictorian of my class. Please write me a presentation based on the following information: As a student at Vanderbilt University, I feel honored. The educational journey at Vandy has been nothing less than enlightening. The dedicated professors here at Vanderbilt are the best. As an 18 year old student at VU, the opportunities are endless."""
    main(user_message)


# output:
# NLTK NER Response: {'results': [{'entity_type': 'ORGANIZATION', 'text': 'Vanderbilt University'}, {'entity_type': 'ORGANIZATION', 'text': 'Vandy'}, {'entity_type': 'ORGANIZATION', 'text': 'Vanderbilt'}, {'entity_type': 'ORGANIZATION', 'text': 'VU'}]}
# Entities updated from nltk_ner.
# trigger post processing
# Generating embeddings...
# Detect Response: {'results': '{ "results": [{"entity_type": "EDUCATIONAL_RECORD", "text": "I will be the valedictorian of my class."}, {"entity_type": "AFFILIATION", "text": "Vanderbilt University"}, {"entity_type": "AFFILIATION", "text": "VU"}] }'}
# Generate Embeddings Response: {'message': 'Embeddings saved to entity_embeddings.csv'}
# Clustering UF...
# Cluster UF Response: {'results': {'Vanderbilt University': ['Vanderbilt University', 'Vandy', 'Vanderbilt', 'VU']}}
# Clustering...
# Cluster Response: {'results': '{\n    "Vanderbilt University": ["Vanderbilt University", "Vandy", "VU", "Vanderbilt"],\n    "18 year old": ["18 year old"]\n}'}
# Entities updated from detect_entities.
# trigger post processing
# Generating embeddings...
# Generate Embeddings Response: {'message': 'Embeddings saved to entity_embeddings.csv'}
# Clustering UF...
# Cluster UF Response: {'results': {'Vanderbilt University': ['Vanderbilt University', 'Vandy', 'Vanderbilt', 'VU']}}
# Clustering...
# Cluster Response: {'results': '{\n"Vanderbilt University": ["Vanderbilt University", "Vandy", "VU", "Vanderbilt"],\n"18 year old": ["18 year old"],\n"VU": ["VU", "Vanderbilt University", "Vandy", "Vanderbilt"]\n}'}
# Merging results from detect and nltk_ner.
