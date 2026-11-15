
import os
import re
import unicodedata

import requests


API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


# Common words that do not identify the actual subject or assertion.
# Negation words such as "not" and "never" are intentionally retained.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "of", "on", "in", "at", "to", "for", "from", "by",
    "with", "that", "this", "these", "those", "who", "whom",
    "what", "when", "where", "why", "how", "did", "does", "do",
    "has", "have", "had", "their", "his", "her", "its", "they",
    "he", "she", "it", "and", "or", "but", "as", "following",
    "after", "before", "about", "among", "into", "over", "under",
    "according", "reported", "reportedly", "says", "said",
    "claim", "claims", "fact", "check", "factcheck", "fact-check"
}


# Basic normalizations. This is a conservative lexical filter,
# not a substitute for a semantic language model.
TOKEN_ALIASES = {
    "israeli": "israel",
    "israelis": "israel",
    "dead": "die",
    "died": "die",
    "death": "die",
    "dying": "die",
    "hacked": "hack",
    "hacking": "hack",
    "rumours": "rumor",
    "rumors": "rumor"
}


NEGATIVE_RATING = re.compile(
    r"\b(false|mostly false|incorrect|inaccurate|misleading|"
    r"misinformation|fake|not true|not correct|not accurate|"
    r"untrue|wrong|debunked|pants on fire)\b",
    re.IGNORECASE
)

POSITIVE_RATING = re.compile(
    r"\b(mostly true|true|correct|accurate|verified|right)\b",
    re.IGNORECASE
)


# Conservative matching thresholds.
MIN_CLAIM_COVERAGE = 0.75
MIN_CLAIM_F1 = 0.55
MIN_TITLE_COVERAGE = 0.65
MIN_TITLE_F1 = 0.50


def get_api_key():
    """Read the Google Fact Check API key."""

    key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GOOGLE_FACT_CHECK_API_KEY")
        except Exception:
            key = None

    if not key:
        raise ValueError(
            "Google Fact Check API key is missing. "
            "Configure it in .streamlit/secrets.toml."
        )

    return key


def search_fact_checks(query, page_size=10):
    """Search Google's published fact-check reviews."""

    query = str(query or "").strip()

    if not query:
        return []

    params = {
        "query": query[:500],
        "languageCode": "en",
        "pageSize": page_size,
        "key": get_api_key()
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=25
    )

    response.raise_for_status()

    data = response.json()
    results = []

    for claim in data.get("claims", []):

        reviews = []

        for review in claim.get("claimReview", []):

            publisher = review.get("publisher") or {}

            reviews.append({
                "publisher": publisher.get(
                    "name",
                    "Unknown publisher"
                ),
                "title": review.get("title", ""),
                "url": review.get("url", ""),
                "rating": review.get("textualRating", ""),
                "review_date": review.get("reviewDate", "")
            })

        results.append({
            "claim": claim.get("text", ""),
            "claimant": claim.get("claimant", ""),
            "claim_date": claim.get("claimDate", ""),
            "reviews": reviews
        })

    return results


def normalize_token(token):
    """Normalize a token for basic lexical comparison."""

    token = TOKEN_ALIASES.get(token, token)

    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"

    elif (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith("ss")
    ):
        token = token[:-1]

    return token


def claim_tokens(text):
    """Extract normalized content words from text."""

    text = unicodedata.normalize(
        "NFKD",
        str(text or "")
    )

    text = (
        text.encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )

    tokens = set()

    for token in re.findall(r"[a-z0-9]+", text):

        if token in STOP_WORDS:
            continue

        token = normalize_token(token)

        if token:
            tokens.add(token)

    return tokens


def overlap_metrics(original_claim, candidate_text):
    """Calculate query coverage, precision and F1 token overlap."""

    query_tokens = claim_tokens(original_claim)
    candidate_tokens = claim_tokens(candidate_text)

    if not query_tokens or not candidate_tokens:
        return {
            "query_coverage": 0.0,
            "candidate_precision": 0.0,
            "f1": 0.0,
            "matched_terms": []
        }

    common = query_tokens.intersection(candidate_tokens)

    coverage = len(common) / len(query_tokens)
    precision = len(common) / len(candidate_tokens)

    f1 = (
        2 * coverage * precision / (coverage + precision)
        if coverage + precision
        else 0.0
    )

    return {
        "query_coverage": coverage,
        "candidate_precision": precision,
        "f1": f1,
        "matched_terms": sorted(common)
    }


def filter_relevant_fact_checks(original_claim, fact_checks):
    """
    Retain only fact-check reviews that closely match the
    submitted claim AND the individual review title.

    Results about the same person or topic but a different
    allegation are excluded from verdict calculation.

    Returns:
        relevant_results, unmatched_results
    """

    relevant_results = []
    unmatched_results = []

    query_tokens = claim_tokens(original_claim)

    # Very short or generic inputs cannot be matched safely.
    if len(query_tokens) < 2:
        return [], list(fact_checks or [])

    for result in fact_checks or []:

        reviewed_claim = str(
            result.get("claim") or ""
        ).strip()

        reviews = result.get("reviews") or []

        claim_score = overlap_metrics(
            original_claim,
            reviewed_claim
        )

        claim_matches = (
            claim_score["query_coverage"] >= MIN_CLAIM_COVERAGE
            and claim_score["f1"] >= MIN_CLAIM_F1
        )

        relevant_reviews = []

        if claim_matches:

            for review in reviews:

                review_title = str(
                    review.get("title") or ""
                ).strip()

                # When Google provides no title, use the reviewed claim.
                comparison_text = review_title or reviewed_claim

                title_score = overlap_metrics(
                    original_claim,
                    comparison_text
                )

                title_matches = (
                    title_score["query_coverage"] >= MIN_TITLE_COVERAGE
                    and title_score["f1"] >= MIN_TITLE_F1
                )

                if title_matches:

                    relevant_review = dict(review)

                    relevant_review["relevance_score"] = round(
                        min(
                            claim_score["f1"],
                            title_score["f1"]
                        ),
                        3
                    )

                    relevant_review["matched_terms"] = (
                        title_score["matched_terms"]
                    )

                    relevant_reviews.append(relevant_review)

        if relevant_reviews:

            relevant_result = dict(result)

            # Keep only the reviews that passed relevance checks.
            relevant_result["reviews"] = relevant_reviews

            relevant_result["relevance_score"] = max(
                item["relevance_score"]
                for item in relevant_reviews
            )

            relevant_results.append(relevant_result)

        else:

            unmatched_results.append(result)

    return relevant_results, unmatched_results


def assess_fact_checks(fact_checks):
    """
    Assess ratings from an already relevance-filtered result list.

    This function must not be given raw, unfiltered Google results.
    """

    positive = 0
    negative = 0
    ratings = []

    for claim in fact_checks or []:

        for review in claim.get("reviews", []):

            rating = str(
                review.get("rating") or ""
            ).strip()

            if not rating:
                continue

            ratings.append(rating)

            # Negative ratings are checked first.
            if NEGATIVE_RATING.search(rating):
                negative += 1

            elif POSITIVE_RATING.search(rating):
                positive += 1

    if positive and negative:

        verdict = "Unverified"

        explanation = (
            "Relevant fact-check reviews contain conflicting ratings. "
            "The claim cannot be assigned a clear verdict from these results."
        )

    elif positive:

        verdict = "Supported"

        explanation = (
            "Closely matching fact-check reviews contain supportive ratings. "
            "Read the original reviews to confirm their exact scope and context."
        )

    elif negative:

        verdict = "Contradicted"

        explanation = (
            "Closely matching fact-check reviews contain negative ratings. "
            "Read the original reviews to confirm their exact scope and context."
        )

    else:

        verdict = "Unverified"

        explanation = (
            "No clear interpretable rating was found among the "
            "closely matching fact-check reviews."
        )

    return {
        "verdict": verdict,
        "explanation": explanation,
        "ratings": ratings
    }
