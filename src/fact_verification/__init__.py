
from fact_verification.google_factcheck import (
    search_fact_checks,
    assess_fact_checks
)
from fact_verification.gdelt_search import search_news


def verify_claim(claim, max_news=10):
    """
    Search published fact-check reviews and related news coverage.

    Google Fact Check reviews determine the assessment.
    GDELT results provide context only.
    """

    claim = str(claim or "").strip()

    if not claim:
        raise ValueError("A headline or factual claim is required.")

    fact_checks = []
    related_news = []
    errors = []

    # Search Google Fact Check
    try:
        fact_checks = search_fact_checks(claim)
    except Exception as error:
        errors.append(f"Google Fact Check search: {error}")

    # Search GDELT for related coverage
    try:
        related_news = search_news(
            claim,
            max_records=max_news
        )
    except Exception as error:
        errors.append(f"GDELT news search: {error}")

    assessment = assess_fact_checks(fact_checks)

    verdict = assessment.get("verdict", "Unverified")
    explanation = assessment.get(
        "explanation",
        "No clear fact-check verdict was found."
    )

    if not fact_checks:
        verdict = "Unverified"
        explanation = (
            "No matching published fact-check review was found. "
            "Related news coverage is provided for context only. "
            "The claim remains Unverified."
        )

    return {
        "claim": claim,
        "verdict": verdict,
        "explanation": explanation,
        "ratings": assessment.get("ratings", []),
        "fact_checks": fact_checks,
        "related_news": related_news,
        "errors": errors
    }
