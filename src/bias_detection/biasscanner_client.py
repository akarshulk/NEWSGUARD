
import requests


BIASSCANNER_ENDPOINT = "https://app.biasscanner.org:8080"


def analyze_with_biasscanner(text, language="en", timeout=90):
    """Send news text to BiasScanner and return its structured results."""

    text = str(text or "").strip()

    if not text:
        raise ValueError("News text cannot be empty.")

    payload = {
        "type": "prompt",
        "text": text,
        "language": language
    }

    try:
        response = requests.post(
            BIASSCANNER_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            "BiasScanner request timed out."
        ) from error

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"BiasScanner request failed: {error}"
        ) from error

    except ValueError as error:
        raise RuntimeError(
            "BiasScanner returned an invalid JSON response."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError("Unexpected BiasScanner response format.")

    sentences = result.get("sentences", [])
    overall_bias = result.get("overall_bias", {})

    if not isinstance(sentences, list):
        raise RuntimeError("Invalid sentence-level response.")

    if not isinstance(overall_bias, dict):
        overall_bias = {}

    return {
        "sentences": [
            item for item in sentences
            if isinstance(item, dict)
        ],
        "overall_bias": overall_bias
    }
