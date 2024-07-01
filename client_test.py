import requests
import threading
import json
import pandas as pd
import os

base_url = "http://localhost:3000"
entities = []
merge_clustering_response = {}
entities_lock = threading.Lock()
clustering_lock = threading.Lock()

def post_request(endpoint, data):
    url = f"{base_url}/{endpoint}"
    response = requests.post(url, json=data)
    try:
        return response.json()
    except ValueError:
        print(f"Error decoding JSON response from {endpoint}: {response.text}")
        return {}

def normalize_entities(results):
    """Normalize the entities format to a list of dictionaries."""
    print("clean entities")
    print(results)
    if isinstance(results, str):
        try:
            results = json.loads(results)
            if 'results' in results:
                return results['results']
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return results

def detect_entities(user_message):
    global entities
    response = post_request('detect', {'message': user_message})
    print("Detect Response:", response)
    with entities_lock:
        normalized_results = normalize_entities(response.get('results', []))
        entities.extend(normalized_results)
        print("Entities updated from detect_entities.")
        merge_entities_results()

def nltk_ner(user_message):
    global entities
    response = post_request('nltk-ner', {'message': user_message})
    print("NLTK NER Response:", response)
    with entities_lock:
        normalized_results = normalize_entities(response.get('results', []))
        entities.extend(normalized_results)
        print("Entities updated from nltk_ner.")
        merge_entities_results()

def generate_embeddings():
    global entities
    print("Generating embeddings...")
    response = post_request('generate-embeddings', {})
    print("Generate Embeddings Response:", response)
    return response

def cluster_uf():
    global merge_clustering_response
    print("Clustering UF...")
    response = post_request('clusteruf', {})
    print("Cluster UF Response:", response)
    with clustering_lock:
        uf_results = response.get('results', {})
        for key, value in uf_results.items():
            if len(value)>4:
                continue

            if key in merge_clustering_response:
                if len(merge_clustering_response[key])>3:
                    continue
                merge_clustering_response[key] = list(set(merge_clustering_response[key] + value))
            else:
                merge_clustering_response[key] = value
        merge_clustering_response_updated()

def cluster(user_message):
    global merge_clustering_response
    print("Clustering...")
    response = post_request('cluster', {'message': user_message})
    print("Cluster Response:", response)
    with clustering_lock:
        cluster_results = json.loads(response.get('results', '{}'))
        for key, value in cluster_results.items():
            if len(value)>4:
                continue
            if key in merge_clustering_response:
                continue
                # if len(merge_clustering_response[key])>3:
                #     continue
                # merge_clustering_response[key] = list(set(merge_clustering_response[key] + value))
            else:
                merge_clustering_response[key] = value
        merge_clustering_response_updated()

def merge_clustering_response_updated():
    print("Merged Clustering Response:", merge_clustering_response)
    # Update the global variable on the server
    response = post_request('update-cluster-results', {'results': merge_clustering_response})
    print("Update Clustering Response:", response)

def merge_entities_results():
    global entities
    print("Merged Entities:", entities)
    response = post_request('update-entities', {'entities': entities})
    print("Update Entities Response:", response)
    trigger_post_processing()

def trigger_post_processing():
    print("Trigger post processing")
    generate_embeddings()
    cluster_uf()
    cluster(user_message)

def main(user_message):
    detect_thread = threading.Thread(target=detect_entities, args=(user_message,))
    nltk_ner_thread = threading.Thread(target=nltk_ner, args=(user_message,))

    detect_thread.start()
    nltk_ner_thread.start()

    detect_thread.join()
    nltk_ner_thread.join()

    print("Merging results from detect and nltk_ner.")
    merge_entities_results()

if __name__ == "__main__":
    user_message = """I will be the valedictorian of my class. Please write me a presentation based on the following information: As a student at Vanderbilt University, I feel honored. The educational journey at Vandy has been nothing less than enlightening. The dedicated professors here at Vanderbilt are the best. As an 18 year old student at VU, the opportunities are endless."""
#     user_message ="""Here is the offer statistics for some students of the class of 2023:
# North America: MIT, UCLA, University of California Berkeley, Harvard University, Stanford University, Massachusetts Institute of Technology, Princeton University, University of Chicago, University of Toronto, McGill University, University of California Los Angeles, UChi, CMU, Carnegie Mellon University, UCB, University of British Columbia
# Canada: UBC
# Europe: ETHZ, ETH Zurich, Oxford, University of Cambridge, Imperial College London, London School of Economics, LSE, IC
# """
    main(user_message)
