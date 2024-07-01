import requests

def call_detect_endpoint(message):
    url = "http://localhost:3000/detect"
    payload = {"message": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Detect response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

def call_nltk_ner_endpoint(message):
    url = "http://localhost:3000/nltk-ner"
    payload = {"message": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("NLTK NER response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

def call_generate_embeddings_endpoint():
    url = "http://localhost:3000/generate-embeddings"
    response = requests.post(url)
    if response.status_code == 200:
        print("Generate Embeddings response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

def call_cluster1_endpoint(message):
    url = "http://localhost:3000/cluster"
    payload = {"message": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Cluster response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

def call_cluster2_endpoint():
    url = "http://localhost:3000/clusteruf"
    response = requests.post(url)
    if response.status_code == 200:
        print("Cluster response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

if __name__ == "__main__":
#     test_message = """
# Here is the offer statistics for some students of the class of 2023:
# North America: MIT, UCLA, University of California Berkeley, Harvard University, Stanford University, Massachusetts Institute of Technology, Princeton University, University of Chicago, University of Toronto, McGill University, University of California Los Angeles, UChi, CMU, Carnegie Mellon University, UCB, University of British Columbia
# Canada: UBC
# Europe: ETHZ, ETH Zurich, Oxford, University of Cambridge, Imperial College London, London School of Economics, LSE, IC
# """
    test_message = "I will be the valedictorian of my class. Please write me a presentation based on the following information: As a student at Vanderbilt University, I feel honored. The educational journey at Vandy has been nothing less than enlightening. The dedicated professors here at Vanderbilt are the best. As an 18 year old student at VU, the opportunities are endless."


    # call_detect_endpoint(test_message)
    call_nltk_ner_endpoint(test_message)


    call_generate_embeddings_endpoint()
    # call_cluster1_endpoint(test_message)
    call_cluster2_endpoint()
