
from fact_verification.google_factcheck import (
    search_fact_checks,
    filter_relevant_fact_checks,
    assess_fact_checks
)

from fact_verification.gdelt_search import search_news

from fact_verification.gemini_current_verification import (
    verify_current_claim
)


def verify_claim(claim, max_news=10):
    """
    Combined verification:

    1. Gemini + Google Search: primary current verification.
    2. Google Fact Check: historical fact-check context.
    3. GDELT: related news coverage only.

    Historical reviews cannot independently override
    the current Gemini assessment.
    """

    claim = str(claim or "").strip()

    if not claim:
        raise ValueError(
            "A headline or factual claim is required."
        )

    errors = []

    all_fact_checks = []
    relevant_fact_checks = []
    related_news = []

    historical_assessment = {
        "verdict": "Unverified",
        "explanation": "No relevant historical fact-check was found.",
        "ratings": []
    }

    current_assessment = {
        "verdict": "Unverified",
        "summary": "Current verification was not completed.",
        "reasoning": "",
        "as_of_date": "",
        "sources": [],
        "search_queries": []
    }

    # -----------------------------------------------------
    # 1. HISTORICAL GOOGLE FACT-CHECK SEARCH
    # -----------------------------------------------------

    try:

        all_fact_checks = search_fact_checks(claim)

        relevant_fact_checks, unmatched_fact_checks = (
            filter_relevant_fact_checks(
                claim,
                all_fact_checks
            )
        )

        historical_assessment = assess_fact_checks(
            relevant_fact_checks
        )

    except Exception as error:

        errors.append(
            f"Google Fact Check: {error}"
        )

        unmatched_fact_checks = []

    # -----------------------------------------------------
    # 2. CURRENT GEMINI VERIFICATION
    # -----------------------------------------------------

    try:

        current_assessment = verify_current_claim(claim)

    except Exception as error:

        errors.append(
            f"Gemini Current Verification: {error}"
        )

        current_assessment = {
            "verdict": "Unverified",
            "summary": (
                "Current verification could not be completed. "
                "Check your Gemini API key and service availability."
            ),
            "reasoning": str(error),
            "as_of_date": "",
            "sources": [],
            "search_queries": [],
            "error": str(error)
        }

    # Gemini is the primary current-verification engine.
    verdict = current_assessment.get(
        "verdict",
        "Unverified"
    )

    explanation = current_assessment.get(
        "summary",
        "No conclusive current assessment was returned."
    )

    # -----------------------------------------------------
    # 3. HISTORICAL CONFLICT NOTICE
    # -----------------------------------------------------

    historical_verdict = historical_assessment.get(
        "verdict",
        "Unverified"
    )

    historical_conflict = (
        verdict in {"Supported", "Contradicted"}
        and historical_verdict in {"Supported", "Contradicted"}
        and verdict != historical_verdict
    )

    if historical_conflict:

        explanation += (
            " A relevant historical fact-check has a different "
            "rating. The historical review may concern a different "
            "timeframe or context, so the discrepancy should be "
            "examined before treating the current assessment as final."
        )

    # -----------------------------------------------------
    # 4. GDELT RELATED NEWS
    # -----------------------------------------------------

    try:

        related_news = search_news(
            claim,
            max_records=max_news
        )

    except Exception as error:

        errors.append(
            f"GDELT: {error}"
        )

    # -----------------------------------------------------
    # 5. RETURN
    # -----------------------------------------------------

    return {
        "claim": claim,

        # Final current assessment.
        "verdict": verdict,
        "explanation": explanation,

        # Current Gemini evidence.
        "current_assessment": current_assessment,
        "current_sources": current_assessment.get(
            "sources",
            []
        ),

        # Historical Google Fact Check evidence.
        "historical_assessment": historical_assessment,
        "historical_conflict": historical_conflict,
        "ratings": historical_assessment.get(
            "ratings",
            []
        ),
        "fact_checks": relevant_fact_checks,

        # Search context.
        "related_news": related_news,
        "search_result_count": len(all_fact_checks),
        "relevant_fact_check_count": len(
            relevant_fact_checks
        ),
        "unmatched_fact_check_count": len(
            unmatched_fact_checks
        ),

        "errors": errors
    }
