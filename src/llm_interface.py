"""
LLM-powered natural language interface for the heart disease risk model.

Design:
  - FEATURE_SPECS documents every feature the model needs (used both to
    ground the LLM's extraction prompt and to generate clarifying questions).
  - CRITICAL_FEATURES must be provided by the user or the system asks a
    clarifying question rather than guessing (avoids garbage predictions).
  - AUXILIARY_FEATURES may be filled from population medians
    (models/feature_defaults.json) when missing -- always disclosed to the
    user as an assumption, never silently substituted.
  - All LLM-calling functions accept an injectable `client` so the parsing
    and validation logic can be unit tested without hitting a real API.
"""
import os
import json
import re

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root, if present

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base")

FEATURE_SPECS = {
    "age":      {"label": "age", "type": "int", "range": (18, 100), "desc": "age in years"},
    "sex":      {"label": "sex", "type": "binary", "range": (0, 1), "desc": "biological sex (0 = female, 1 = male)"},
    "cp":       {"label": "chest pain type", "type": "int", "range": (1, 4),
                 "desc": "chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)"},
    "trestbps": {"label": "resting blood pressure", "type": "int", "range": (80, 220), "desc": "resting blood pressure in mm Hg"},
    "chol":     {"label": "serum cholesterol", "type": "int", "range": (100, 600), "desc": "serum cholesterol in mg/dl"},
    "fbs":      {"label": "fasting blood sugar", "type": "binary", "range": (0, 1),
                 "desc": "fasting blood sugar > 120 mg/dl (0 = no, 1 = yes)"},
    "restecg":  {"label": "resting ECG results", "type": "int", "range": (0, 2),
                 "desc": "resting electrocardiographic results (0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy)"},
    "thalach":  {"label": "max heart rate achieved", "type": "int", "range": (60, 220), "desc": "maximum heart rate achieved"},
    "exang":    {"label": "exercise-induced angina", "type": "binary", "range": (0, 1),
                 "desc": "exercise-induced angina (0 = no, 1 = yes)"},
    "oldpeak":  {"label": "ST depression", "type": "float", "range": (0.0, 7.0),
                 "desc": "ST depression induced by exercise relative to rest"},
    "slope":    {"label": "slope of peak exercise ST segment", "type": "int", "range": (1, 3),
                 "desc": "slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)"},
    "ca":       {"label": "number of major vessels", "type": "int", "range": (0, 3),
                 "desc": "number of major vessels (0-3) colored by fluoroscopy"},
    "thal":     {"label": "thalassemia", "type": "int", "range": (3, 7),
                 "desc": "thalassemia test result (3 = normal, 6 = fixed defect, 7 = reversible defect)"},
}

CRITICAL_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "oldpeak"]
AUXILIARY_FEATURES = ["fbs", "restecg", "slope", "ca", "thal"]

SYSTEM_PROMPT = """You are a strict information-extraction engine for a heart disease risk tool.
Extract clinical feature values from the user's message and respond with ONLY a JSON object,
no prose, no markdown fences. Schema:

{
  "in_scope": true or false,
  "features": {
    "age": number or null, "sex": 0 or 1 or null, "cp": 1-4 or null,
    "trestbps": number or null, "chol": number or null, "fbs": 0 or 1 or null,
    "restecg": 0-2 or null, "thalach": number or null, "exang": 0 or 1 or null,
    "oldpeak": number or null, "slope": 1-3 or null, "ca": 0-3 or null, "thal": 3/6/7 or null
  },
  "note": "short string, empty unless in_scope is false"
}

Set "in_scope" to false ONLY if the user is asking something this tool cannot do at all
(e.g. asking for a diagnosis of a different disease, medication advice, or general chit-chat
unrelated to heart disease risk from these features). A message that just provides partial
patient information is still in scope, even if many fields are missing -- extract what you can
and leave the rest null. Only include numbers you are confident about; never invent values.
"""


# --------------------------------------------------------------------------
# Feature extraction / validation (pure, network-free, unit-testable)
# --------------------------------------------------------------------------

def extract_features_from_llm_json(raw_llm_text):
    """
    Parse the LLM's raw text response into a structured dict.
    Tolerates markdown code fences around the JSON. Raises ValueError
    if the response cannot be parsed as the expected schema.
    """
    text = raw_llm_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM response as JSON: {e}\nRaw response: {raw_llm_text}")

    if "features" not in data or "in_scope" not in data:
        raise ValueError(f"LLM response missing required keys: {data}")

    # Only keep known feature keys, coerce to float where possible
    clean_features = {}
    for key in FEATURE_SPECS:
        val = data["features"].get(key)
        if val is None:
            clean_features[key] = None
        else:
            try:
                clean_features[key] = float(val)
            except (TypeError, ValueError):
                clean_features[key] = None

    return {
        "in_scope": bool(data.get("in_scope", True)),
        "features": clean_features,
        "note": data.get("note", "") or "",
    }


def check_missing_features(features_dict):
    """Return (missing_critical, missing_auxiliary) lists of feature keys."""
    missing_critical = [f for f in CRITICAL_FEATURES if features_dict.get(f) is None]
    missing_auxiliary = [f for f in AUXILIARY_FEATURES if features_dict.get(f) is None]
    return missing_critical, missing_auxiliary


def validate_feature_ranges(features_dict):
    """Return a list of human-readable warnings for out-of-range values."""
    warnings = []
    for key, val in features_dict.items():
        if val is None or key not in FEATURE_SPECS:
            continue
        lo, hi = FEATURE_SPECS[key]["range"]
        if not (lo <= val <= hi):
            warnings.append(
                f"{FEATURE_SPECS[key]['label']} = {val} is outside the expected range ({lo}-{hi})"
            )
    return warnings


def load_feature_defaults():
    path = os.path.join(MODEL_DIR, "feature_defaults.json")
    with open(path) as f:
        return json.load(f)


def apply_auxiliary_defaults(features_dict, missing_auxiliary, defaults):
    """Fill missing auxiliary features from population defaults. Returns (filled_dict, assumptions)."""
    filled = dict(features_dict)
    assumptions = []
    for key in missing_auxiliary:
        if key in defaults:
            filled[key] = defaults[key]
            assumptions.append(f"{FEATURE_SPECS[key]['label']} assumed to be {defaults[key]} (population typical value)")
    return filled, assumptions


def build_clarifying_message(missing_critical):
    """Deterministic, network-free clarifying question for missing required fields."""
    lines = [FEATURE_SPECS[f]["desc"] for f in missing_critical]
    bullet_list = "\n".join(f"- {line}" for line in lines)
    return (
        "I need a bit more information before I can estimate heart disease risk. "
        f"Could you provide:\n{bullet_list}"
    )


def build_out_of_scope_message(note):
    base = ("This tool only estimates heart disease risk from clinical measurements "
            "(age, blood pressure, cholesterol, chest pain type, and similar values). "
            "It can't provide a diagnosis, medication advice, or answer unrelated questions.")
    if note:
        return f"{base} {note}"
    return base


# --------------------------------------------------------------------------
# Model loading / inference
# --------------------------------------------------------------------------

def load_model_bundle():
    """Load the exported best model + feature column order."""
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    return model, feature_columns


def predict(model, feature_columns, features_dict):
    """Run inference. Returns {'prediction': 0/1, 'probability': float}."""
    row = pd.DataFrame([{col: features_dict[col] for col in feature_columns}])
    pred = int(model.predict(row)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        classes = list(model.classes_)
        prob = float(proba[classes.index(1)]) if 1 in classes else float(max(proba))
    return {"prediction": pred, "probability": prob}


def explain_top_factors(model, feature_columns, features_dict, feature_defaults, top_n=3):
    """
    Rank features by (importance * how far the patient's value is from the
    population median), as a lightweight, dependency-free explainability
    signal. Returns a list of {feature, label, value, direction} dicts.
    """
    # Find the underlying estimator with feature_importances_ (handles Pipeline too)
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("clf", model)

    if not hasattr(estimator, "feature_importances_"):
        return []

    importances = dict(zip(feature_columns, estimator.feature_importances_))

    scored = []
    for col in feature_columns:
        val = features_dict.get(col)
        default = feature_defaults.get(col)
        if val is None or default is None:
            continue
        deviation = abs(val - default)
        score = importances.get(col, 0) * deviation
        direction = "above typical" if val > default else ("below typical" if val < default else "typical")
        scored.append({
            "feature": col,
            "label": FEATURE_SPECS[col]["label"],
            "value": val,
            "direction": direction,
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# --------------------------------------------------------------------------
# Knowledge base (optional RAG grounding for the response-generation step)
# --------------------------------------------------------------------------

def load_knowledge_snippet(feature_key):
    """
    Read knowledge_base/<feature>.md if present, else fall back to the
    built-in FEATURE_SPECS description. Keeps the app functional even if
    the knowledge base files aren't in the repo yet.
    """
    path = os.path.join(KB_DIR, f"{feature_key}.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return FEATURE_SPECS.get(feature_key, {}).get("desc", "")


def load_overview_snippet():
    path = os.path.join(KB_DIR, "heart_disease_overview.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    return ("Heart disease risk is influenced by a combination of demographic, "
            "clinical, and diagnostic measurements.")


# --------------------------------------------------------------------------
# LLM calls (network-dependent, kept thin and mockable)
# --------------------------------------------------------------------------

def get_llm_client():
    """
    Build an OpenAI-compatible client pointed at Nebius Token Factory by default.
    Set LLM_PROVIDER=openai to use OpenAI's API instead -- the rest of the
    code is provider-agnostic since Nebius exposes an OpenAI-compatible API.
    """
    from openai import OpenAI

    provider = os.environ.get("LLM_PROVIDER", "nebius").lower()
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    return OpenAI(
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url="https://api.tokenfactory.nebius.com/v1/",
    )


def get_parse_model():
    """
    Model used for feature extraction (parsing). This is a cheap, mechanical
    task -- a small instruct model is plenty and keeps cost negligible.
    """
    provider = os.environ.get("LLM_PROVIDER", "nebius").lower()
    if provider == "openai":
        return os.environ.get("OPENAI_PARSE_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    return os.environ.get("NEBIUS_PARSE_MODEL", os.environ.get("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning"))


def get_explain_model():
    """
    Model used for the final natural-language explanation. Worth spending a
    bit more here for response quality, since this is what the user reads.
    """
    provider = os.environ.get("LLM_PROVIDER", "nebius").lower()
    if provider == "openai":
        return os.environ.get("OPENAI_EXPLAIN_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    return os.environ.get("NEBIUS_EXPLAIN_MODEL", "nvidia/Nemotron-3_5-Lightning")


def call_llm_parse(user_text, client=None, model=None):
    """Call the LLM to extract structured features from free text. Returns raw text."""
    if client is None:
        client = get_llm_client()
    if model is None:
        model = get_parse_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def call_llm_generate_response(prediction_result, top_factors, features_dict, assumptions,
                                overview_snippet, client=None, model=None):
    """Call the LLM to turn the raw prediction into a clear, contextual explanation."""
    if client is None:
        client = get_llm_client()
    if model is None:
        model = get_explain_model()

    factors_text = "\n".join(
        f"- {f['label']}: {f['value']} ({f['direction']} for this population)" for f in top_factors
    ) or "No dominant factors identified."

    assumptions_text = "\n".join(f"- {a}" for a in assumptions) or "None."

    prompt = f"""A heart disease risk model produced this result for a patient:
Prediction: {"heart disease likely" if prediction_result['prediction'] == 1 else "heart disease unlikely"}
Estimated probability of heart disease: {prediction_result['probability']:.0%}

Top contributing factors for this patient:
{factors_text}

Assumptions made due to missing information:
{assumptions_text}

Background context:
{overview_snippet}

Write a clear, warm, 3-5 sentence explanation of this result for the patient. Include the
probability, explain the top factors in plain language, mention any assumptions made, and
add a brief caveat that this is a statistical estimate from a machine learning model, not a
medical diagnosis, and they should consult a doctor for actual clinical decisions."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------
# End-to-end orchestration (used by app.py)
# --------------------------------------------------------------------------

def handle_query(user_text, model, feature_columns, feature_defaults, client=None,
                  parse_model=None, explain_model=None):
    """
    Full pipeline: parse -> validate -> (clarify | out-of-scope | predict+explain).
    Returns a dict with a 'type' key: 'clarify', 'out_of_scope', or 'result'.

    Uses a cheap model for parsing (mechanical extraction) and a stronger
    model for the final explanation (what the user actually reads), unless
    overridden via parse_model/explain_model.
    """
    raw = call_llm_parse(user_text, client=client, model=parse_model)
    parsed = extract_features_from_llm_json(raw)

    if not parsed["in_scope"]:
        return {"type": "out_of_scope", "message": build_out_of_scope_message(parsed["note"])}

    features = parsed["features"]
    missing_critical, missing_auxiliary = check_missing_features(features)

    if missing_critical:
        return {"type": "clarify", "message": build_clarifying_message(missing_critical),
                "missing": missing_critical}

    range_warnings = validate_feature_ranges(features)
    filled_features, assumptions = apply_auxiliary_defaults(features, missing_auxiliary, feature_defaults)

    prediction_result = predict(model, feature_columns, filled_features)
    top_factors = explain_top_factors(model, feature_columns, filled_features, feature_defaults)
    overview = load_overview_snippet()

    explanation = call_llm_generate_response(
        prediction_result, top_factors, filled_features, assumptions, overview,
        client=client, model=explain_model,
    )

    return {
        "type": "result",
        "prediction": prediction_result,
        "top_factors": top_factors,
        "assumptions": assumptions,
        "range_warnings": range_warnings,
        "explanation": explanation,
    }
