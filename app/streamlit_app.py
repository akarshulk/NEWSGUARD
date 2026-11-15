
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# =========================================================
# PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NEWSGUARD",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# NEWSGUARD — DEEP VINTAGE THEME
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --ng-deep-teal: #244855;
        --ng-coral: #E64833;
        --ng-earth: #874F41;
        --ng-muted-teal: #90AEAD;
        --ng-cream: #874F41;
        --ng-ink: #FBE9D0;
        --ng-paper: #FFF9F1;
        --ng-input-text: #244855;
    }

    /* Overall page */
    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--ng-cream);
        color: var(--ng-ink);
    }
    [data-testid="stHeader"] {
        background: rgba(135, 79, 65, 0.94);
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Typography */
    h1, h2, h3, h4,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: var(--ng-ink) !important;
        letter-spacing: -0.025em;
    }

    /* Align the native NEWSGUARD title with the form's left edge. */
    [data-testid="stMainBlockContainer"] h1 {
        margin-left: -0.75rem !important;
    }
    p, label, li, [data-testid="stCaptionContainer"] {
        color: var(--ng-ink);
    }
    [data-testid="stCaptionContainer"] {
        opacity: 0.78;
    }

    /* Text inputs: keep both controls cream with dark, readable text
       in every state (idle, focused, typing, and browser autofill). */
    [data-testid="stTextInput"] div[data-baseweb="input"] > div,
    [data-testid="stTextArea"] div[data-baseweb="textarea"] > div,
    .stTextInput div[data-baseweb="input"] > div,
    .stTextArea div[data-baseweb="textarea"] > div {
        background: var(--ng-paper) !important;
        background-color: var(--ng-paper) !important;
        border: 1px solid rgba(36, 72, 85, 0.35) !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    .stTextInput input,
    .stTextArea textarea,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input:focus,
    div[data-baseweb="textarea"] textarea:focus,
    .stTextInput input:focus,
    .stTextArea textarea:focus {
        background: var(--ng-paper) !important;
        background-color: var(--ng-paper) !important;
        color: var(--ng-input-text) !important;
        -webkit-text-fill-color: var(--ng-input-text) !important;
        caret-color: var(--ng-deep-teal) !important;
        border-radius: 9px !important;
        opacity: 1 !important;
    }

    /* Prevent Chrome/Edge autofill from reintroducing a dark field. */
    [data-testid="stTextInput"] input:-webkit-autofill,
    [data-testid="stTextArea"] textarea:-webkit-autofill,
    .stTextInput input:-webkit-autofill,
    .stTextArea textarea:-webkit-autofill {
        -webkit-text-fill-color: var(--ng-input-text) !important;
        -webkit-box-shadow: 0 0 0 1000px var(--ng-paper) inset !important;
        box-shadow: 0 0 0 1000px var(--ng-paper) inset !important;
        caret-color: var(--ng-deep-teal) !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder,
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: rgba(36, 72, 85, 0.62) !important;
        -webkit-text-fill-color: rgba(36, 72, 85, 0.62) !important;
        opacity: 1 !important;
    }

    [data-testid="stTextInput"]:focus-within div[data-baseweb="input"] > div,
    [data-testid="stTextArea"]:focus-within div[data-baseweb="textarea"] > div,
    div[data-baseweb="input"]:focus-within > div,
    div[data-baseweb="textarea"]:focus-within > div {
        background: var(--ng-paper) !important;
        background-color: var(--ng-paper) !important;
        border-color: var(--ng-muted-teal) !important;
        box-shadow: 0 0 0 2px rgba(144, 174, 173, 0.25) !important;
    }

    /* Buttons */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: var(--ng-coral) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--ng-coral) !important;
        border-radius: 10px !important;
        font-weight: 650 !important;
        transition: background 150ms ease, transform 150ms ease;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #CF3D2B !important;
        border-color: #CF3D2B !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }

    /* Primary Analyze News button: deep teal */
    .stButton > button[kind="primary"] {
        background: #244855 !important;
        background-color: #244855 !important;
        color: #FBE9D0 !important;
        border: 1px solid #244855 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #193641 !important;
        background-color: #193641 !important;
        border-color: #193641 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 3px rgba(36, 72, 85, 0.30) !important;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(230, 72, 51, 0.22) !important;
    }

    /* Checkbox and radio accents */
    [data-testid="stCheckbox"] label,
    [data-testid="stRadio"] label {
        color: var(--ng-ink) !important;
    }
    [data-testid="stCheckbox"] input:checked + div,
    [data-testid="stRadio"] input:checked + div {
        background-color: var(--ng-coral) !important;
        border-color: var(--ng-coral) !important;
    }

    /* Metrics: cream cards must always use deep-teal text. */
    [data-testid="stMetric"] {
        background: #FBE9D0 !important;
        border: 1px solid rgba(36, 72, 85, 0.22) !important;
        border-radius: 12px !important;
        padding: 0.9rem !important;
    }
    [data-testid="stMetric"] *,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: #244855 !important;
        -webkit-text-fill-color: #244855 !important;
    }
    [data-testid="stMetricValue"] { font-weight: 700 !important; }

    /* Expanders: warm cream surface with readable teal text. */
    [data-testid="stExpander"] {
        background: #FFF9F1 !important;
        border: 1px solid #90AEAD !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * {
        color: #244855 !important;
        -webkit-text-fill-color: #244855 !important;
    }
    [data-testid="stExpander"] summary { font-weight: 650 !important; }

    /* Alerts: replace Streamlit's default dark blocks with warm cream. */
    [data-testid="stAlert"],
    [data-testid="stAlert"] > div,
    div[role="alert"] {
        background: #FBE9D0 !important;
        color: #244855 !important;
        border: 1px solid #90AEAD !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    [data-testid="stAlert"] *,
    div[role="alert"] * {
        color: #244855 !important;
        -webkit-text-fill-color: #244855 !important;
    }
    [data-testid="stAlert"] svg,
    div[role="alert"] svg { color: #244855 !important; fill: #244855 !important; }

    /* Native HTML tables use NEWSGUARD's editorial palette. */
    [data-testid="stTable"] {
        background: #FFF9F1 !important;
        border: 1px solid #90AEAD !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    [data-testid="stTable"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        background: #FFF9F1 !important;
    }
    [data-testid="stTable"] th {
        background: #244855 !important;
        color: #FBE9D0 !important;
        -webkit-text-fill-color: #FBE9D0 !important;
        border: 1px solid #52727B !important;
    }
    [data-testid="stTable"] td {
        background: #FFF9F1 !important;
        color: #244855 !important;
        -webkit-text-fill-color: #244855 !important;
        border: 1px solid rgba(36, 72, 85, 0.18) !important;
    }

    /* Dividers and links: high contrast on the earth-brown page. */
    hr {
        border-color: rgba(251, 233, 208, 0.32) !important;
    }
    a,
    a:link,
    a:visited,
    [data-testid="stMarkdownContainer"] a,
    [data-testid="stMarkdownContainer"] a:visited {
        color: #FBE9D0 !important;
        -webkit-text-fill-color: #FBE9D0 !important;
        text-decoration: underline !important;
        text-decoration-color: #E64833 !important;
        text-decoration-thickness: 2px !important;
        text-underline-offset: 3px !important;
        font-weight: 650 !important;
    }
    a:hover,
    a:focus-visible,
    [data-testid="stMarkdownContainer"] a:hover {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        text-decoration-color: #FFFFFF !important;
        text-decoration-thickness: 2px !important;
    }

    /* Spinner */
    [data-testid="stSpinner"] > div {
        border-top-color: var(--ng-coral) !important;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# LOAD LOCAL BIAS MODELS
# =========================================================

@st.cache_resource
def load_bias_resources():

    from bias_detection.predict_bias import (
        load_models,
        load_biased_terms
    )

    bias_model, opinion_model = load_models()
    biased_terms = load_biased_terms()

    return (
        bias_model,
        opinion_model,
        biased_terms
    )

# =========================================================
# DISPLAY HELPERS
# =========================================================

def render_model_result(
    title,
    result,
    positive_label,
    negative_label
):
    """
    Display a confidence-aware binary classification.

    An uncertain prediction is never presented as a
    definitive positive or negative classification.
    """

    prediction = str(
        result.get("prediction", "Needs review")
    )

    confidence = result.get("confidence")

    st.markdown(f"**{title}**")

    if prediction == "Inconclusive":

        st.info(
            "Inconclusive — the model score is below "
            "the display threshold."
        )

    elif prediction == "Needs review":

        st.info(
            "Needs review — this model does not provide "
            "probability estimates."
        )

    elif prediction == positive_label:

        st.warning(
            f"⚠️ {positive_label.upper()}"
        )

    elif prediction == negative_label:

        st.success(
            f"✅ {negative_label.upper()}"
        )

    else:

        st.info(f"Prediction: {prediction}")

    if confidence is not None:

        st.metric(
            "Model Probability",
            f"{float(confidence):.2f}%"
        )

    if result.get("raw_prediction"):

        st.caption(
            f"Underlying model prediction: "
            f"{result['raw_prediction']}"
        )

def display_styled_table(dataframe):
    """Render a table using NEWSGUARD's cream-and-teal palette."""
    styled = (
        dataframe.reset_index(drop=True)
        .style
        .hide(axis="index")
        .set_table_styles([
            {"selector": "table", "props": [
                ("width", "100%"), ("border-collapse", "collapse"),
                ("background-color", "#FFF9F1"),
                ("border", "1px solid #90AEAD"),
            ]},
            {"selector": "th", "props": [
                ("background-color", "#244855"), ("color", "#FBE9D0"),
                ("font-weight", "650"), ("text-align", "left"),
                ("padding", "10px 12px"), ("border", "1px solid #52727B"),
            ]},
            {"selector": "td", "props": [
                ("background-color", "#FFF9F1"), ("color", "#244855"),
                ("padding", "9px 12px"),
                ("border", "1px solid rgba(36, 72, 85, 0.18)"),
            ]},
            {"selector": "tbody tr:nth-child(even) td", "props": [
                ("background-color", "#FBE9D0"),
            ]},
        ], overwrite=True)
    )
    st.table(styled)


def display_sentence_results(
    results,
    title,
    empty_message
):
    """Display individual sentence predictions."""

    st.markdown(f"**{title}**")

    if not results:

        st.info(empty_message)
        return

    dataframe = pd.DataFrame(results)

    display_styled_table(dataframe)

# =========================================================
# HEADER
# =========================================================

st.title("🛡️ NEWSGUARD")
st.caption("Claim Verification & News Bias Analysis")

# =========================================================
# NEWS INPUT
# =========================================================

st.subheader("Submit News for Analysis")

headline = st.text_input(
    "News Headline *",
    placeholder="Enter the news headline..."
)

article = st.text_area(
    "News Article / Content (Optional)",
    height=220,
    placeholder="Paste the full news article here (optional)..."
)

st.caption(
    "A headline is sufficient to start verification. "
    "The article is optional."
)

send_to_biasscanner = st.checkbox(
    "We suggest using BiasScanner too (because our in-house bias detector is currently rough around the edges 🤓)",
    value=True,
    help=(
        "BiasScanner provides an additional sentence-level analysis. "
        "When enabled, the headline and article are sent to the external service."
    )
)

analyze_button = st.button(
    "🔍 Analyze News",
    type="primary",
    use_container_width=True
)

# =========================================================
# RUN ANALYSIS
# =========================================================

if analyze_button:

    if not headline.strip():

        st.warning(
            "Please enter a news headline."
        )

        st.stop()

    claim = headline.strip()

    news_text = "\n\n".join(
        part.strip()
        for part in [headline, article]
        if part.strip()
    )

    headline_only = not bool(article.strip())

    # =====================================================
    # 1. CLAIM VERIFICATION
    # =====================================================

    st.divider()
    st.header("1. Claim Verification")

    verification_completed = False
    verdict = "Unverified"

    try:

        from fact_verification.verify import verify_claim

        with st.spinner(
            "Searching current web evidence and historical fact-checks..."
        ):

            verification = verify_claim(claim)

        verdict = verification.get(
            "verdict",
            "Unverified"
        )

        explanation = verification.get(
            "explanation",
            "No conclusive evidence was found."
        )

        verification_completed = True

        display_verdicts = {
            "Supported": "Evidence Supports Claim",
            "Contradicted": "Potentially False",
            "Unverified": "Insufficient Evidence"
        }

        display_verdict = display_verdicts.get(
            verdict,
            "Insufficient Evidence"
        )

        if verdict == "Supported":

            st.success(
                f"Assessment: {display_verdict}"
            )

        elif verdict == "Contradicted":

            st.error(
                f"Assessment: {display_verdict}"
            )

        else:

            st.warning(
                f"Assessment: {display_verdict}"
            )

        st.write(explanation)

        # -------------------------------------------------
        # CURRENT WEB VERIFICATION
        # -------------------------------------------------

        st.subheader("Current Web Verification")

        current = verification.get(
            "current_assessment",
            {}
        )

        current_summary = current.get(
            "summary",
            "Current verification information is unavailable."
        )

        st.write(current_summary)

        if current.get("as_of_date"):

            st.caption(
                f"Assessment date: {current['as_of_date']}"
            )

        reasoning = current.get("reasoning")

        if reasoning:

            with st.expander(
                "View Evidence Evaluation"
            ):

                st.write(reasoning)

        current_sources = verification.get(
            "current_sources",
            []
        )

        st.markdown("**Current Evidence Sources**")

        if current_sources:

            for index, source in enumerate(
                current_sources,
                start=1
            ):

                title = source.get(
                    "title",
                    source.get("domain", "Source")
                )

                url = source.get("url", "")

                if url:

                    st.markdown(
                        f"{index}. **[{title}]({url})**"
                    )

                else:

                    st.markdown(
                        f"{index}. **{title}**"
                    )

                details = " · ".join(
                    str(value)
                    for value in (
                        source.get("domain"),
                        source.get("source_type"),
                        source.get("published_date")
                    )
                    if value
                )

                if details:
                    st.caption(details)

                if source.get("finding"):
                    st.write(source["finding"])

        else:

            st.warning(
                "No usable current evidence sources were returned."
            )

        # -------------------------------------------------
        # HISTORICAL FACT-CHECKS
        # -------------------------------------------------

        st.subheader(
            "Historical Published Fact-Check Reviews"
        )

        fact_checks = verification.get(
            "fact_checks",
            []
        )

        if fact_checks:

            for item in fact_checks:

                with st.container(border=True):

                    st.markdown(
                        "**Claim reviewed:** "
                        + str(item.get("claim") or "Not provided")
                    )

                    if item.get("claimant"):

                        st.caption(
                            f"Claimant: {item['claimant']}"
                        )

                    for review in item.get("reviews", []):

                        st.markdown(
                            f"**{review.get('publisher', 'Publisher')}**"
                        )

                        st.write(
                            "Historical rating: "
                            + str(
                                review.get("rating")
                                or "Not available"
                            )
                        )

                        if review.get("title"):
                            st.write(review["title"])

                        if review.get("url"):

                            st.markdown(
                                f"[Read Original Fact-Check]"
                                f"({review['url']})"
                            )

                        if review.get("review_date"):

                            st.caption(
                                f"Review date: {review['review_date']}"
                            )

        else:

            st.info(
                "No closely matching historical fact-check reviews "
                "were found. This does not mean the claim is false."
            )

        # -------------------------------------------------
        # HISTORICAL CONFLICT NOTICE
        # -------------------------------------------------

        if verification.get("historical_conflict"):

            st.warning(
                "A relevant historical fact-check has a different "
                "rating from the current assessment. Review the "
                "dates and context of both sources."
            )

    except Exception as error:

        st.error(
            f"Claim verification could not be completed: {error}"
        )

        st.stop()

    # =====================================================
    # VERIFICATION DECISION GATE
    # =====================================================

    if not verification_completed or verdict != "Supported":

        if verdict == "Contradicted":

            st.error(
                "Bias and opinion analysis skipped because "
                "the current evidence contradicts the claim."
            )

        else:

            st.warning(
                "Bias and opinion analysis skipped because "
                "there is insufficient current supporting evidence."
            )

        st.stop()

    # =====================================================
    # 2. BIAS AND OPINION ANALYSIS
    # =====================================================

    st.divider()
    st.header("2. Bias and Opinion Analysis")

    if headline_only:

        st.info(
            "Headline-only mode: only the headline is analysed. "
            "A complete article provides a more representative "
            "basis for sentence-level analysis."
        )

    else:

        st.caption(
            "The headline and article are divided into sentences "
            "before being passed to the MBIC-trained models."
        )

    # =====================================================
    # 2A. EXISTING MBIC MODELS
    # =====================================================

    st.subheader(
        "NEWSGUARD — Sentence-Level Model Analysis"
    )

    try:

        with st.spinner(
            "Analysing individual sentences..."
        ):

            (
                bias_model,
                opinion_model,
                biased_terms
            ) = load_bias_resources()

            from bias_detection.predict_bias import (
                analyze_text,
                MIN_DISPLAY_CONFIDENCE
            )

            bias_result = analyze_text(
                news_text,
                bias_model,
                opinion_model,
                biased_terms
            )

        bias_info = bias_result.get(
            "bias_analysis",
            {}
        )

        opinion_info = bias_result.get(
            "opinion_analysis",
            {}
        )

        # -------------------------------------------------
        # BIAS CLASSIFICATION
        # -------------------------------------------------

        st.markdown("### Linguistic Bias Result")

        bias_prediction = bias_info.get(
            "prediction",
            "Needs review"
        )

        col1, col2 = st.columns(2)

        with col1:

            render_model_result(
                title="Overall Bias Classification",
                result=bias_info,
                positive_label="Biased",
                negative_label="Unbiased"
            )

        with col2:

            st.metric(
                "Sentences Analysed",
                bias_info.get("sentence_count", 0)
            )

            st.metric(
                "Confidence Threshold",
                f"{MIN_DISPLAY_CONFIDENCE:.0f}%"
            )

        bias_distribution = bias_info.get(
            "class_probabilities",
            {}
        )

        if bias_distribution:

            st.markdown("**Average Class Probability Distribution**")

            distribution_df = pd.DataFrame(
                list(bias_distribution.items()),
                columns=[
                    "Class",
                    "Average Probability (%)"
                ]
            )

            display_styled_table(distribution_df)

        # -------------------------------------------------
        # SENTENCE-LEVEL BIAS RESULTS
        # -------------------------------------------------

        display_sentence_results(
            bias_info.get("sentences", []),
            "Individual Sentence Bias Predictions",
            "No sentence-level bias predictions were generated."
        )

        # -------------------------------------------------
        # OPINION CLASSIFICATION
        # -------------------------------------------------

        st.divider()

        st.markdown("### Opinion Analysis")

        opinion_prediction = opinion_info.get(
            "prediction",
            "Needs review"
        )

        opinion_confidence = opinion_info.get(
            "confidence"
        )

        if opinion_prediction == "Inconclusive":

            st.info(
                "Inconclusive — the opinion model's average "
                "probability is below the display threshold."
            )

        elif opinion_prediction == "Needs review":

            st.info(
                "Needs review — probability estimates are unavailable."
            )

        else:

            st.write(
                f"**Overall opinion classification:** "
                f"{opinion_prediction}"
            )

        if opinion_confidence is not None:

            st.metric(
                "Opinion Model Probability",
                f"{float(opinion_confidence):.2f}%"
            )

        opinion_distribution = opinion_info.get(
            "class_probabilities",
            {}
        )

        if opinion_distribution:

            opinion_df = pd.DataFrame(
                list(opinion_distribution.items()),
                columns=[
                    "Opinion Category",
                    "Average Probability (%)"
                ]
            )

            display_styled_table(opinion_df)

        # -------------------------------------------------
        # SENTENCE-LEVEL OPINION RESULTS
        # -------------------------------------------------

        display_sentence_results(
            opinion_info.get("sentences", []),
            "Individual Sentence Opinion Predictions",
            "No sentence-level opinion predictions were generated."
        )

    except Exception as error:

        st.error(
            f"Local bias and opinion analysis failed: {error}"
        )

    # =====================================================
    # 2B. BIASSCANNER
    # =====================================================

    st.divider()

    st.subheader(
        "Additional Sentence-Level Analysis — BiasScanner"
    )

    if not send_to_biasscanner:

        st.info(
            "BiasScanner was not called. Enable the external-service "
            "checkbox before analysis to use it."
        )

    else:

        try:

            from bias_detection.biasscanner_client import (
                analyze_with_biasscanner
            )

            with st.spinner(
                "Analysing news using BiasScanner..."
            ):

                scanner_result = analyze_with_biasscanner(
                    news_text
                )

            scanner_sentences = scanner_result.get(
                "sentences",
                []
            )

            overall_bias = scanner_result.get(
                "overall_bias",
                {}
            )

            if not isinstance(overall_bias, dict):
                overall_bias = {}

            # -------------------------------------------------
            # SENTENCE-LEVEL RESULTS
            # -------------------------------------------------

            if scanner_sentences:

                sentence_rows = []

                for item in scanner_sentences:

                    if not isinstance(item, dict):
                        continue

                    strength = item.get(
                        "bias_strength"
                    )

                    try:

                        strength = (
                            round(float(strength), 3)
                            if strength is not None
                            else None
                        )

                    except (TypeError, ValueError):

                        strength = str(strength)

                    sentence_rows.append({
                        "Sentence": item.get(
                            "text",
                            ""
                        ),
                        "Bias Type": item.get(
                            "bias_type",
                            "Not provided"
                        ),
                        "Bias Strength (0–1)": strength,
                        "Explanation": item.get(
                            "bias_description",
                            "Not provided"
                        )
                    })

                if sentence_rows:

                    st.metric(
                        "Sentences Returned by BiasScanner",
                        len(sentence_rows)
                    )

                    display_styled_table(pd.DataFrame(sentence_rows))

            else:

                st.info(
                    "BiasScanner returned an overall assessment, "
                    "but no sentence-level classifications for this input."
                )

            # -------------------------------------------------
            # OVERALL BIAS ASSESSMENT
            # -------------------------------------------------

            conclusion = overall_bias.get(
                "conclusion"
            )

            if conclusion:

                st.markdown(
                    "**BiasScanner Overall Assessment**"
                )

                st.write(conclusion)

            elif not scanner_sentences:

                st.warning(
                    "BiasScanner did not return a usable "
                    "sentence-level or overall assessment."
                )

        except Exception as error:

            st.warning(
                "BiasScanner is currently unavailable or returned "
                f"an unexpected response: {error}"
            )

            st.info(
                "The local NEWSGUARD bias and opinion results "
                "remain available."
            )
