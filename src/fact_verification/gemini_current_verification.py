
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from tavily import TavilyClient


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/interactions"
)

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite"
]

MAX_GENERAL_RESULTS = 8
MAX_OFFICIAL_RESULTS = 8
REQUEST_TIMEOUT = 90

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


# =========================================================
# API KEYS
# =========================================================

def get_tavily_api_key():
    """Read existing Tavily API key from project .env."""

    key = os.getenv("TAVILY_API_KEY")

    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("TAVILY_API_KEY")
        except Exception:
            key = None

    if not key:
        raise ValueError(
            "TAVILY_API_KEY not found in the project .env file."
        )

    return key


def get_gemini_api_key():
    """Read Gemini API key from environment or Streamlit secrets."""

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None

    if not key:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Configure it in .streamlit/secrets.toml."
        )

    return key


# =========================================================
# SOURCE CLASSIFICATION
# =========================================================

OFFICIAL_SUFFIXES = (
    ".gov",
    ".gov.in",
    ".gov.il",
    ".gov.uk",
    ".gov.au",
    ".govt.nz",
    ".go.jp",
    ".go.kr",
    ".gouv.fr",
    ".gc.ca",
    ".europa.eu",
    ".mil",
    ".nic.in"
)

TRUSTED_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "pbs.org",
    "npr.org",
    "politifact.com",
    "snopes.com",
    "factcheck.org",
    "nytimes.com",
    "washingtonpost.com",
    "abcnews.go.com",
    "abcnews.com",
    "nbcnews.com",
    "cbsnews.com",
    "theguardian.com",
    "usatoday.com",
    "sky.com",
    "aljazeera.com",
    "france24.com",
    "dw.com",
    "cnn.com",
    "bloomberg.com",
    "ft.com"
}


def get_domain(url):
    return (
        urlparse(str(url or "")).netloc
        .lower()
        .removeprefix("www.")
    )


def is_official_domain(domain):
    """Check whether a domain uses a recognized government suffix."""

    return any(
        domain == suffix.lstrip(".")
        or domain.endswith(suffix)
        for suffix in OFFICIAL_SUFFIXES
    )


def trusted_news_domain(domain):
    """Return the recognized publisher domain, if available."""

    for trusted in TRUSTED_NEWS_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return trusted

    return None


# =========================================================
# TARGETED OFFICIAL DOMAIN DISCOVERY
# =========================================================

def get_official_domains(claim):
    """
    Identify likely government domains based on the claim.

    This is a targeted-search helper, not a factual verdict.
    """

    text = str(claim or "").lower()

    country_domains = {
        "israel": [
            "gov.il",
            "knesset.gov.il"
        ],
        "india": [
            "india.gov.in",
            "pib.gov.in",
            "sansad.in",
            "nic.in"
        ],
        "united states": [
            "whitehouse.gov",
            "usa.gov",
            "congress.gov",
            "senate.gov"
        ],
        "usa": [
            "whitehouse.gov",
            "usa.gov",
            "congress.gov",
            "senate.gov"
        ],
        "united kingdom": [
            "gov.uk",
            "parliament.uk"
        ],
        "britain": [
            "gov.uk",
            "parliament.uk"
        ],
        "uk": [
            "gov.uk",
            "parliament.uk"
        ],
        "canada": [
            "canada.ca",
            "ourcommons.ca"
        ],
        "australia": [
            "australia.gov.au",
            "aph.gov.au"
        ],
        "japan": [
            "go.jp",
            "kantei.go.jp"
        ],
        "france": [
            "gouvernement.fr",
            "elysee.fr"
        ],
        "germany": [
            "bundesregierung.de"
        ]
    }

    domains = []

    for country, country_list in country_domains.items():

        if re.search(
            r"\b" + re.escape(country) + r"\b",
            text
        ):
            domains.extend(country_list)

    if "netanyahu" in text:
        domains.extend([
            "gov.il",
            "knesset.gov.il"
        ])

    if "modi" in text:
        domains.extend([
            "india.gov.in",
            "pib.gov.in"
        ])

    if "prime minister" in text or "president" in text:
        # These searches are country-targeted above.
        pass

    return list(dict.fromkeys(domains))


# =========================================================
# TAVILY SEARCH
# =========================================================

def search_current_evidence(claim):
    """
    Search general news and targeted official sources using Tavily.

    Uses the existing Tavily API key.
    """

    client = TavilyClient(
        api_key=get_tavily_api_key()
    )

    today = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).date().isoformat()

    all_results = []

    # -----------------------------------------------------
    # 1. GENERAL WEB SEARCH
    # -----------------------------------------------------

    general_response = client.search(
        query=claim,
        topic="general",
        search_depth="advanced",
        max_results=MAX_GENERAL_RESULTS,
        include_answer=False
    )

    all_results.extend(
        general_response.get("results", [])
    )

    # -----------------------------------------------------
    # 2. TARGETED OFFICIAL-SOURCE SEARCH
    # -----------------------------------------------------

    official_domains = get_official_domains(claim)

    if official_domains:

        official_query = (
            f"{claim} current official government "
            "institutional record or position holder"
        )

        official_response = client.search(
            query=official_query,
            topic="general",
            search_depth="advanced",
            max_results=MAX_OFFICIAL_RESULTS,
            include_domains=official_domains,
            include_answer=False
        )

        all_results.extend(
            official_response.get("results", [])
        )

    # -----------------------------------------------------
    # 3. REMOVE DUPLICATES
    # -----------------------------------------------------

    unique_results = {}

    for item in all_results:

        url = str(
            item.get("url") or ""
        ).strip()

        if not url.startswith(("http://", "https://")):
            continue

        key = url.rstrip("/").lower()

        if key not in unique_results:
            unique_results[key] = item

    # -----------------------------------------------------
    # 4. STRUCTURE SOURCES
    # -----------------------------------------------------

    sources = []

    for index, item in enumerate(
        unique_results.values(),
        start=1
    ):

        url = item["url"]
        domain = get_domain(url)

        if is_official_domain(domain):
            source_type = "official_primary"

        elif trusted_news_domain(domain):
            source_type = "independent_news"

        else:
            source_type = "other"

        sources.append({
            "source_id": f"T{index}",
            "title": str(
                item.get("title") or "Untitled"
            ),
            "url": url,
            "domain": domain,
            "source_type": source_type,
            "published_date": (
                item.get("published_date")
                or item.get("date")
                or "Not available"
            ),
            "content": str(
                item.get("content") or ""
            )[:2500],
            "search_score": item.get("score")
        })

    return sources, today


# =========================================================
# RESPONSE PARSING
# =========================================================

def extract_output_text(data):
    """Extract final text from Gemini Interactions response."""

    if data.get("output_text"):
        return str(data["output_text"]).strip()

    text_parts = []

    for step in data.get("steps", []):

        if step.get("type") != "model_output":
            continue

        for block in step.get("content", []):

            if block.get("type") == "text":

                value = block.get("text", "")

                if value:
                    text_parts.append(value)

    if text_parts:
        return "\n".join(text_parts).strip()

    output = data.get("output", [])

    if isinstance(output, list):

        for block in output:

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                text_parts.append(
                    block.get("text", "")
                )

    return "\n".join(text_parts).strip()


def parse_json_response(text):
    """Parse JSON while tolerating Markdown code fences."""

    cleaned = str(text or "").strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end <= start:
        raise ValueError(
            "Gemini did not return a valid JSON object."
        )

    return json.loads(
        cleaned[start:end + 1]
    )


# =========================================================
# EVIDENCE VALIDATION
# =========================================================

def validate_evidence(evidence, sources):
    """
    Accept only Tavily source IDs provided to Gemini.

    Definitive verdict threshold:
      - One relevant official primary source, OR
      - Two independent trusted news publishers.

    Two URLs from the same publisher count as one source.
    """

    source_map = {
        item["source_id"]: item
        for item in sources
    }

    verified_sources = []
    seen = set()

    for item in evidence:

        if not isinstance(item, dict):
            continue

        source_id = str(
            item.get("source_id", "")
        ).strip()

        if source_id not in source_map:
            continue

        if source_id in seen:
            continue

        seen.add(source_id)

        original = source_map[source_id]

        verified_sources.append({
            "source_id": source_id,
            "title": original["title"],
            "url": original["url"],
            "domain": original["domain"],
            "source_type": original["source_type"],
            "published_date": original["published_date"],
            "finding": str(
                item.get("finding", "")
            ).strip()
        })

    has_official_source = any(
        item["source_type"] == "official_primary"
        for item in verified_sources
    )

    independent_publishers = {
        trusted_news_domain(item["domain"])
        for item in verified_sources
        if item["source_type"] == "independent_news"
    }

    independent_publishers.discard(None)

    sufficient = (
        has_official_source
        or len(independent_publishers) >= 2
    )

    return verified_sources, sufficient


# =========================================================
# GEMINI PROMPT
# =========================================================

def build_verification_prompt(claim, sources, today):

    evidence_text = []

    for source in sources:

        evidence_text.append({
            "source_id": source["source_id"],
            "title": source["title"],
            "domain": source["domain"],
            "source_type": source["source_type"],
            "published_date": source["published_date"],
            "content": source["content"],
            "url": source["url"]
        })

    prompt = f"""
You are the factual evidence evaluator for NEWSGUARD AI.

CURRENT DATE: {today}

CLAIM:
{claim}

WEB SEARCH RESULTS:
{json.dumps(evidence_text, ensure_ascii=False, indent=2)}

TASK:
Evaluate whether the EXACT submitted claim is supported,
contradicted, or cannot be established from the supplied evidence.

RESEARCH RULES:

1. Use only the supplied search results.
2. Do not invent facts, sources, quotations or URLs.
3. Match the exact proposition, not just the person,
   organization or topic.
4. For present-tense claims, evaluate status as of the
   current date.
5. Check publication dates and distinguish historical
   information from current information.
6. Prefer official primary sources for government positions,
   appointments, laws and institutional facts.
7. Use independent reputable news sources as corroboration.
8. Do not count two articles from the same publisher
   as independent confirmation.
9. A source that only mentions the person or topic does
   not necessarily support the actual proposition.
10. If evidence is conflicting, outdated, irrelevant or
    insufficient, choose INSUFFICIENT_EVIDENCE.
11. Absence of a fact-check is not evidence that a claim is false.
12. Return only source IDs supplied in the search results.
13. Do not treat Wikipedia as an official primary source.
14. Explain any transition or qualification that affects
    the interpretation of the submitted claim.

VERDICTS:

SUPPORTED:
The exact claim is supported by at least one relevant
official primary source or at least two independent
reputable news publishers.

CONTRADICTED:
The exact claim is contradicted by at least one relevant
official primary source or at least two independent
reputable news publishers.

INSUFFICIENT_EVIDENCE:
The evidence does not satisfy either threshold.

RETURN ONLY VALID JSON:

{{
  "verdict": "SUPPORTED, CONTRADICTED, or INSUFFICIENT_EVIDENCE",
  "summary": "Brief assessment of the claim.",
  "reasoning": "Explain how retrieved evidence supports or contradicts it.",
  "evidence": [
    {{
      "source_id": "T1",
      "finding": "What this particular source establishes."
    }}
  ]
}}

If no source directly addresses the claim, return
INSUFFICIENT_EVIDENCE with an empty evidence list.
"""

    return prompt


# =========================================================
# GEMINI API WITH AUTOMATIC RETRY AND FALLBACK
# =========================================================

def call_gemini_with_fallback(
    prompt,
    timeout=REQUEST_TIMEOUT
):
    """
    Try primary Gemini model first.

    Retry temporary failures once.
    Then try the fallback model.

    Authentication and invalid request errors are
    reported directly rather than hidden.
    """

    api_key = get_gemini_api_key()

    last_error = None
    last_status = None

    for model in GEMINI_MODELS:

        print(f"Trying Gemini model: {model}")

        for attempt in range(2):

            try:

                response = requests.post(
                    GEMINI_URL,
                    headers={
                        "x-goog-api-key": api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "input": prompt,
                        "store": False
                    },
                    timeout=timeout
                )

                if response.ok:

                    data = response.json()

                    print(
                        f"Gemini verification successful using {model}"
                    )

                    return data, model

                last_status = response.status_code

                last_error = (
                    f"HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )

                if response.status_code in RETRYABLE_STATUS_CODES:

                    if attempt == 0:

                        print(
                            f"{model} returned HTTP "
                            f"{response.status_code}. "
                            "Retrying once..."
                        )

                        time.sleep(2)
                        continue

                    print(
                        f"{model} remains unavailable. "
                        "Trying the fallback model..."
                    )

                    break

                if response.status_code == 404:

                    print(
                        f"{model} is unavailable. "
                        "Trying the fallback model..."
                    )

                    break

                raise RuntimeError(
                    f"Gemini API returned HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:1000]}"
                )

            except requests.exceptions.Timeout as error:

                last_error = (
                    f"{model} timed out: {error}"
                )

                if attempt == 0:

                    print(
                        "Request timed out. Retrying..."
                    )

                    time.sleep(2)
                    continue

                print(
                    f"{model} timed out again. "
                    "Trying fallback model..."
                )

                break

            except requests.exceptions.RequestException as error:

                last_error = str(error)

                if attempt == 0:
                    time.sleep(2)
                    continue

                break

    raise RuntimeError(
        "All Gemini models failed. "
        f"Last HTTP status: {last_status}. "
        f"Last error: {last_error}"
    )


# =========================================================
# GEMINI EVIDENCE EVALUATION
# =========================================================

def evaluate_evidence_with_gemini(
    claim,
    sources,
    today,
    timeout=REQUEST_TIMEOUT
):
    """Use Gemini for reasoning over Tavily search results."""

    prompt = build_verification_prompt(
        claim,
        sources,
        today
    )

    data, model_used = call_gemini_with_fallback(
        prompt,
        timeout=timeout
    )

    output_text = extract_output_text(data)

    if not output_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    result = parse_json_response(output_text)

    if not isinstance(result, dict):
        raise RuntimeError(
            "Gemini returned an unexpected response format."
        )

    result["_model_used"] = model_used

    return result


# =========================================================
# MAIN VERIFICATION FUNCTION
# =========================================================

def verify_current_claim(
    claim,
    timeout=REQUEST_TIMEOUT
):
    """
    Current factual verification:

        Tavily:
            General live web search
            Targeted official-source search

        Gemini:
            Evidence evaluation
            Primary model + automatic fallback

    """

    claim = str(claim or "").strip()

    if not claim:
        raise ValueError(
            "A factual claim is required."
        )

    today = datetime.now(
        ZoneInfo("Asia/Kolkata")
    ).date().isoformat()

    # -----------------------------------------------------
    # 1. TAVILY SEARCH
    # -----------------------------------------------------

    try:

        sources, today = search_current_evidence(claim)

    except Exception as error:

        raise RuntimeError(
            f"Tavily search failed: {error}"
        ) from error

    if not sources:

        return {
            "verdict": "Unverified",
            "summary": (
                "Tavily did not return usable web sources "
                "for the submitted claim."
            ),
            "reasoning": "",
            "as_of_date": today,
            "sources": [],
            "search_queries": [claim],
            "model_used": None,
            "search_result_count": 0
        }

    # -----------------------------------------------------
    # 2. GEMINI EVIDENCE EVALUATION
    # -----------------------------------------------------

    try:

        result = evaluate_evidence_with_gemini(
            claim,
            sources,
            today,
            timeout=timeout
        )

    except Exception as error:

        raise RuntimeError(
            f"Gemini evidence evaluation failed: {error}"
        ) from error

    # -----------------------------------------------------
    # 3. NORMALIZE VERDICT
    # -----------------------------------------------------

    raw_verdict = str(
        result.get("verdict", "")
    ).strip().upper()

    verdict_map = {
        "SUPPORTED": "Supported",
        "CONTRADICTED": "Contradicted",
        "INSUFFICIENT_EVIDENCE": "Unverified",
        "UNVERIFIED": "Unverified"
    }

    verdict = verdict_map.get(
        raw_verdict,
        "Unverified"
    )

    # -----------------------------------------------------
    # 4. VALIDATE CITED SOURCES
    # -----------------------------------------------------

    evidence = result.get("evidence", [])

    if not isinstance(evidence, list):
        evidence = []

    verified_sources, sufficient = validate_evidence(
        evidence,
        sources
    )

    if verdict in {"Supported", "Contradicted"} and not sufficient:

        verdict = "Unverified"

        reasoning = (
            "The retrieved evidence did not meet the source "
            "corroboration threshold. No definitive verdict "
            "has been assigned."
        )

    else:

        reasoning = str(
            result.get("reasoning", "")
        ).strip()

    summary = str(
        result.get("summary", "")
    ).strip()

    if not summary:
        summary = "No conclusive assessment was returned."

    # -----------------------------------------------------
    # 5. RETURN RESULTS
    # -----------------------------------------------------

    return {
        "verdict": verdict,
        "summary": summary,
        "reasoning": reasoning,
        "as_of_date": today,
        "sources": verified_sources,
        "search_queries": [claim],
        "raw_verdict": raw_verdict,
        "model_used": result.get("_model_used"),
        "search_result_count": len(sources)
    }
