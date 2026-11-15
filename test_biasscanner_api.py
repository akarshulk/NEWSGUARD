
import requests
import json

API_URL = "https://app.biasscanner.org:8080"

payload = {
    "type": "prompt",
    "text": (
        "The government has completely failed the people "
        "with its disastrous and irresponsible policies."
    ),
    "language": "en"
}

try:
    print("Testing BiasScanner API...")

    response = requests.post(
        API_URL,
        json=payload,
        timeout=90
    )

    print("Status code:", response.status_code)
    print("Response:")
    print(response.text[:10000])

    response.raise_for_status()

except requests.exceptions.RequestException as error:
    print("API request failed:", error)
