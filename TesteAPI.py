import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "features": [0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235,
                 -0.03482076, -0.04340085, -0.00259226, 0.01990749, -0.01764613]
}

response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())