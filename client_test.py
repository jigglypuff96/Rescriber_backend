import requests

base_url = "http://localhost:3000"

def test_home():
    response = requests.get(base_url)
    if response.status_code == 200:
        print("Home endpoint test passed!")
        print("Response:", response.text)
    else:
        print("Home endpoint test failed!")
        print("Status code:", response.status_code)
        print("Response:", response.text)

def test_detect():
    test_message = {
        "message": "My name is John Doe, and I live at 123 Main St. You can contact me at john.doe@example.com or call me at 555-1234."
    }

    response = requests.post(f"{base_url}/detect", json=test_message)
    if response.status_code == 200:
        print("Detect endpoint test passed!")
        print("Response:", response.json())
    else:
        print("Detect endpoint test failed!")
        print("Status code:", response.status_code)
        print("Response:", response.text)

if __name__ == "__main__":

    print("\nTesting detect endpoint...")
    test_detect()
