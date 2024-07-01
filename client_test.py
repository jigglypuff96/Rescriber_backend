import requests
import threading
import json
import pandas as pd
import os

base_url = "http://localhost:3000"
entities = []
entities_lock = threading.Lock()

def post_request(endpoint, data):
    url = f"{base_url}/{endpoint}"
    response = requests.post(url, json=data)
    return response.json()

def normalize_entities(results):
    """Normalize the entities format to a list of dictionaries."""
    print("clean entites")
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
        trigger_post_processing()

def nltk_ner(user_message):
    global entities
    response = post_request('nltk-ner', {'message': user_message})
    print("NLTK NER Response:", response)
    with entities_lock:
        normalized_results = normalize_entities(response.get('results', []))
        entities.extend(normalized_results)
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
    print("Trigger post processing")
    generate_embeddings()
    cluster_uf()
    cluster(user_message)

def merge_entities_results():
    # merging is effectively done during the extend operations in the detect_entities and nltk_ner functions, then the merge_results function is unnecessary. We can directly proceed with the post-processing after both threads complete.
    global entities
    # merged_entities = {}
    # for entity in entities:
    #     entity_type = entity['entity_type']
    #     text = entity['text']
    #     if entity_type in merged_entities:
    #         if text not in merged_entities[entity_type]:
    #             merged_entities[entity_type].append(text)
    #     else:
    #         merged_entities[entity_type] = [text]
    # entities.clear()
    # for entity_type, texts in merged_entities.items():
    #     for text in texts:
    #         entities.append({'entity_type': entity_type, 'text': text})
    print("Merged Entities:", entities)
    response = post_request('update-entities', {'entities': entities})
    print("Update Entities Response:", response)
    trigger_post_processing()
    #Note: server only cal nltk_entities, need change

def save_entities_to_csv():
    global entities
    df = pd.DataFrame(entities)
    output_path = "entity_embeddings.csv"
    df.to_csv(output_path, index=False)
    print(f"Entities saved to {output_path}")
    return output_path

def delete_csv(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted.")
    else:
        print(f"{file_path} does not exist.")

def main(user_message):
    detect_thread = threading.Thread(target=detect_entities, args=(user_message,))
    nltk_ner_thread = threading.Thread(target=nltk_ner, args=(user_message,))

    detect_thread.start()
    nltk_ner_thread.start()

    detect_thread.join()
    nltk_ner_thread.join()

    print("Merging results from detect and nltk_ner.")
    merge_entities_results()


    # Note: create at server, send post to delete csv in server
    # csv_path = save_entities_to_csv() 
    # Perform additional operations with the CSV if needed
    # delete_csv(csv_path)

if __name__ == "__main__":
    user_message = """I will be the valedictorian of my class. Please write me a presentation based on the following information: As a student at Vanderbilt University, I feel honored. The educational journey at Vandy has been nothing less than enlightening. The dedicated professors here at Vanderbilt are the best. As an 18 year old student at VU, the opportunities are endless."""
    main(user_message)


