import requests
import threading
import json
import pandas as pd
import os
from datetime import datetime
import csv

# 记录开始时间
start_time = datetime.now()

base_url = "http://localhost:3000"
entities = []
merge_clustering_response = {}
entities_lock = threading.Lock()
clustering_lock = threading.Lock() 

tNLTK = []
record = []
tclusteruf = []
tmergedentities = []
fcluster = []
fmergedcluster =[]
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
    if not tNLTK:
        time1 = datetime.now()
        diff1 = time1 - start_time
        tNLTK.append(results)
        record.append(f"time1 {diff1}" )
        record.append(results)

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
        if not tclusteruf:
            tclusteruf.append(merge_clustering_response)
            time2 = datetime.now()
            diff2 = time2-start_time
            record.append(f"time2 {diff2}" )
            record.append(tclusteruf)

def cluster(user_message):
    global merge_clustering_response
    global fcluster
    print("Clustering...")
    response = post_request('cluster', {'message': user_message})
    print("Cluster Response:", response)
    with clustering_lock:
        cluster_results = json.loads(response.get('results', '{}'))
        fcluster = []
        fcluster.append(cluster_results)
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
    global fmergedcluster
    print("Merged Clustering Response:", merge_clustering_response)
    # Update the global variable on the server
    response = post_request('update-cluster-results', {'results': merge_clustering_response})
    print("Update Clustering Response:", response)
    fmergedcluster= []
    fmergedcluster.append(response)

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
    time3 = datetime.now()
    diff3 = time3-start_time
    record.append(f"time3 {diff3}" )
    # tmergedentities.append(entities)
    record.append(entities)
    
    time4 = datetime.now()
    diff4 = time4-start_time
    record.append(f"time4 {diff4}")
    record.append(fcluster)
    record.append(fmergedcluster)
    with open('record.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(record)

    print("Record appended successfully.")
    
    

    

if __name__ == "__main__":
    user_message = """I will be the valedictorian of my class. Please write me a presentation based on the following information: As a student at Vanderbilt University, I feel honored. The educational journey at Vandy has been nothing less than enlightening. The dedicated professors here at Vanderbilt are the best. As an 18 year old student at VU, the opportunities are endless."""
#     user_message ="""Here is the offer statistics for some students of the class of 2023:
# North America: MIT, UCLA, University of California Berkeley, Harvard University, Stanford University, Massachusetts Institute of Technology, Princeton University, University of Chicago, University of Toronto, McGill University, University of California Los Angeles, UChi, CMU, Carnegie Mellon University, UCB, University of British Columbia
# Canada: UBC
# Europe: ETHZ, ETH Zurich, Oxford, University of Cambridge, Imperial College London, London School of Economics, LSE, IC
# """
    # user_message = """My friend Julie and I are both 19 years old so can’t drink really even tho we are in Korea. How can I handle social situations where there might be pressure to drink"""
#     user_message= """
# Review the following dataset and come up with insightful observations:
# Trip ID	Destination	Start date	End date	Duration (days)	Traveler name	Traveler age	Traveler gender	Traveler nationality	Accommodation type	Accommodation cost	Transportation type	Transportation cost
# 1	London, UK	5/1/2023	5/8/2023	7	John Smith	35	Male	American	Hotel	1200	Flight	600
# 2	Phuket, Thailand	6/15/2023	6/20/2023	5	Jane Doe	28	Female	Canadian	Resort	800	Flight	500
# 3	Bali, Indonesia	7/1/2023	7/8/2023	7	David Lee	45	Male	Korean	Villa	1000	Flight	700
# 4	New York, USA	8/15/2023	8/29/2023	14	Sarah Johnson	29	Female	British	Hotel	2000	Flight	1000
# """
    main(user_message)
