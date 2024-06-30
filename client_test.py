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

def call_cluster_endpoint(embeddings):
    url = "http://localhost:3000/cluster"
    payload = {"message": embeddings}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Cluster response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

if __name__ == "__main__":
    test_message = "University of California, Irvine is located in Irvine, California. Matt is a student at UCI."


    call_detect_endpoint(test_message)
    call_nltk_ner_endpoint(test_message)


    call_generate_embeddings_endpoint()