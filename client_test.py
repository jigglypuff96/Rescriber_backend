import requests

def call_detect_endpoint(message):
    url = "http://localhost:3000/detect"
    payload = {"message": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Detect response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

def call_cluster_endpoint(message):
    url = "http://localhost:3000/cluster"
    payload = {"message": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Cluster response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

def call_abstract_endpoint(message):
    url = "http://localhost:3000/abstract"
    payload = {"message": message}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Abstract response:", response.json())
    else:
        print("Error:", response.status_code, response.json())

if __name__ == "__main__":
    test_message = "My name is John Doe, I live at 1234 Elm Street, and my email is john.doe@example.com."

    # Call detect endpoint
    call_detect_endpoint(test_message)

    # Call cluster endpoint
    call_cluster_endpoint(test_message)

    # Call abstract endpoint
    call_abstract_endpoint(test_message)
