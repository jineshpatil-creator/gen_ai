"""InsightAI — Student Impact Intelligence.

A Flask front end for an *existing*, already-trained scikit-learn pipeline
(`logistic_regression_model.pkl`). This app never retrains, refits, or
replaces the model, and it never invents category encodings:

* feature names and their order come from ``preprocessor.feature_names_in_``
* every categorical option comes from the fitted ``OneHotEncoder.categories_``
* "typical student" defaults come from the fitted imputers' learned statistics

Run in production with:  gunicorn app:app
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request, url_for
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("insightai")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "logistic_regression_model.pkl"))

# --------------------------------------------------------------------------- #
# Presentation metadata (labels, grouping, input guard-rails)
#
# NOTE: min/max below are *input guard-rails* for the form, not facts learned
# by the model. Adjust them if your data uses different scales. Category lists
# are NOT defined here: they are always read from the fitted model.
# --------------------------------------------------------------------------- #
FEATURE_META: dict[str, dict[str, Any]] = {
    "Major_Category": dict(label="Major category", group="profile",
                           help="The student's field of study."),
    "Year_of_Study": dict(label="Year of study", group="profile",
                          help="Current stage of the degree."),
    "Institutional_Policy": dict(label="Institution's GenAI policy", group="profile",
                                 help="How the student's institution treats GenAI tools."),
    "Pre_Semester_GPA": dict(label="Pre-semester GPA", group="academic", unit="GPA",
                             min=0, max=4, step=0.01, help="Scale of 0.00 to 4.00."),
    "Post_Semester_GPA": dict(label="Post-semester GPA", group="academic", unit="GPA",
                              min=0, max=4, step=0.01, help="Scale of 0.00 to 4.00."),
    "Traditional_Study_Hours": dict(label="Traditional study hours", group="academic",
                                    unit="hrs/week", min=0, max=168, step=0.1,
                                    help="Hours per week studying without GenAI."),
    "Skill_Retention_Score": dict(label="Skill retention score", group="academic",
                                  unit="/ 100", min=0, max=100, step=0.1,
                                  help="Score from 0 to 100."),
    "Weekly_GenAI_Hours": dict(label="Weekly GenAI hours", group="usage", unit="hrs/week",
                               min=0, max=168, step=0.1,
                               help="Hours per week spent using GenAI tools."),
    "Tool_Diversity": dict(label="Tool diversity", group="usage", unit="tools",
                           min=0, max=20, step=1, integer=True,
                           help="Number of different GenAI tools used."),
    "Primary_Use_Case": dict(label="Primary use case", group="usage",
                             help="What the student mainly uses GenAI for."),
    "Prompt_Engineering_Skill": dict(label="Prompt-engineering skill", group="usage",
                                     help="Self-assessed skill at writing prompts."),
    "Paid_Subscription": dict(label="Paid GenAI subscription", group="usage",
                              help="Whether the student pays for a GenAI tool."),
    "Perceived_AI_Dependency": dict(label="Perceived AI dependency", group="mindset",
                                    min=1, max=10, step=1, integer=True, scale=True,
                                    help="1 is not dependent, 10 is fully dependent."),
    "Anxiety_Level_During_Exams": dict(label="Anxiety level during exams", group="mindset",
                                       min=1, max=10, step=1, integer=True, scale=True,
                                       help="1 is calm, 10 is severe anxiety."),
}

GROUPS = [
    dict(id="profile", title="Student profile", blurb="Who the student is and the rules they study under."),
    dict(id="academic", title="Academic performance", blurb="Grades, study time and how well skills stuck."),
    dict(id="usage", title="GenAI usage", blurb="How often, how broadly and how skillfully GenAI is used."),
    dict(id="mindset", title="Mindset", blurb="Self-reported reliance and exam stress."),
]

# Display order only. Values themselves always come from the fitted model.
DISPLAY_ORDER: dict[str, list[Any]] = {
    "Year_of_Study": ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
    "Prompt_Engineering_Skill": ["Beginner", "Intermediate", "Advanced"],
    "Institutional_Policy": ["Strict_Ban", "Allowed_With_Citation", "Actively_Encouraged"],
    "Paid_Subscription": [False, True],
}
CLASS_ORDER = ["Low", "Medium", "High"]
CLASS_SUMMARY = {
    "Low": "This profile most resembles students in the lowest burnout-risk group.",
    "Medium": "This profile most resembles students in the middle burnout-risk group.",
    "High": "This profile most resembles students in the highest burnout-risk group.",
}
MAX_WEEK_HOURS = 168


# --------------------------------------------------------------------------- #
# Model loading and schema introspection
# --------------------------------------------------------------------------- #
@dataclass
class Feature:
    name: str
    label: str
    kind: str                     # "numeric" | "categorical" | "boolean"
    group: str
    widget: str                   # "number" | "scale" | "select" | "segmented"
    help: str = ""
    unit: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    integer: bool = False
    typical: Any = None
    options: list[dict[str, str]] = field(default_factory=list)   # public: value + label
    raw_by_value: dict[str, Any] = field(default_factory=dict)    # private: value -> raw category

    def public(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "raw_by_value"}
        return d


@dataclass
class ModelBundle:
    pipeline: Pipeline
    features: list[Feature]                 # exact training order
    classes: list[str]
    class_display_order: list[str]

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.features]


def _py(value: Any) -> Any:
    """numpy scalar -> plain Python scalar."""
    return value.item() if isinstance(value, np.generic) else value


def _value_key(raw: Any) -> str:
    raw = _py(raw)
    return str(raw).lower() if isinstance(raw, bool) else str(raw)


def _label(raw: Any) -> str:
    raw = _py(raw)
    if isinstance(raw, bool):
        return "Yes" if raw else "No"
    return str(raw).replace("_", " ")


def _find_step(transformer: Any, kind: type) -> Any | None:
    steps = transformer.named_steps.values() if isinstance(transformer, Pipeline) else [transformer]
    return next((s for s in steps if isinstance(s, kind)), None)


def _build_features(pipe: Pipeline) -> list[Feature]:
    pre = pipe.named_steps["preprocessor"]
    if not isinstance(pre, ColumnTransformer) or not hasattr(pre, "feature_names_in_"):
        raise RuntimeError("Pipeline step 'preprocessor' is not a fitted ColumnTransformer.")

    order = [str(n) for n in pre.feature_names_in_]
    missing = [n for n in order if n not in FEATURE_META]
    extra = [n for n in FEATURE_META if n not in order]
    if missing or extra:
        raise RuntimeError(f"Model features differ from the UI schema. "
                           f"Missing metadata for {missing}; unused metadata for {extra}.")

    cat_info: dict[str, dict[str, Any]] = {}
    num_info: dict[str, Any] = {}
    for name, transformer, cols in pre.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        encoder = _find_step(transformer, OneHotEncoder)
        imputer = _find_step(transformer, SimpleImputer)
        stats = list(imputer.statistics_) if imputer is not None else [None] * len(cols)
        if encoder is not None:
            for col, cats, stat in zip(cols, encoder.categories_, stats):
                cat_info[col] = {"categories": [_py(c) for c in cats], "typical": _py(stat)}
        else:
            for col, stat in zip(cols, stats):
                num_info[col] = _py(stat)

    uncovered = [n for n in order if n not in cat_info and n not in num_info]
    if uncovered:
        raise RuntimeError(f"Features not handled by the preprocessor: {uncovered}")

    features: list[Feature] = []
    for name in order:
        meta = FEATURE_META[name]
        base = dict(name=name, label=meta["label"], group=meta["group"], help=meta.get("help", ""))
        if name in cat_info:
            cats = cat_info[name]["categories"]
            preferred = DISPLAY_ORDER.get(name, [])
            cats = sorted(cats, key=lambda c: preferred.index(c) if c in preferred else len(preferred))
            is_bool = all(isinstance(c, bool) for c in cats)
            f = Feature(kind="boolean" if is_bool else "categorical",
                        widget="segmented" if len(cats) <= 3 else "select",
                        typical=_value_key(cat_info[name]["typical"]) if cat_info[name]["typical"] is not None else None,
                        options=[{"value": _value_key(c), "label": _label(c)} for c in cats],
                        raw_by_value={_value_key(c): c for c in cats}, **base)
        else:
            lo, hi, step = meta["min"], meta["max"], meta["step"]
            integer = bool(meta.get("integer"))
            typical = num_info[name]
            if typical is not None:
                typical = min(max(float(typical), lo), hi)
                typical = int(round(typical)) if integer else round(typical, 2 if step < 1 else 1)
            f = Feature(kind="numeric", widget="scale" if meta.get("scale") else "number",
                        unit=meta.get("unit", ""), min=lo, max=hi, step=step,
                        integer=integer, typical=typical, **base)
        features.append(f)
    return features


def load_model(path: Path) -> tuple[ModelBundle | None, str | None]:
    """Load the existing pipeline. Returns (bundle, None) or (None, error message)."""
    try:
        if not path.is_file():
            return None, f"Model file not found at {path.name}."
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with open(path, "rb") as fh:
                pipe = pickle.load(fh)  # trusted, first-party artifact shipped with the app
        for w in caught:
            if issubclass(w.category, InconsistentVersionWarning):
                return None, (f"The model was saved with a different scikit-learn version than the "
                              f"one installed ({sklearn.__version__}). Install the version pinned "
                              f"in requirements.txt.")
            log.warning("While loading model: %s", w.message)
        if not isinstance(pipe, Pipeline) or not hasattr(pipe, "predict_proba"):
            return None, "The model file is not a scikit-learn Pipeline with predict_proba."
        features = _build_features(pipe)
        classes = [str(c) for c in pipe.classes_]
        ordered = [c for c in CLASS_ORDER if c in classes] + [c for c in classes if c not in CLASS_ORDER]
        log.info("Model loaded: %d features, classes=%s, scikit-learn %s",
                 len(features), classes, sklearn.__version__)
        return ModelBundle(pipe, features, classes, ordered), None
    except Exception as exc:  # noqa: BLE001 - surface any load problem gracefully
        log.exception("Could not load model")
        return None, f"The model could not be loaded: {exc}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _parse_number(raw: Any, f: Feature) -> tuple[float | None, str | None]:
    missing = (f"Choose a rating from {_fmt(f.min)} to {_fmt(f.max)}." if f.widget == "scale"
               else "Enter a value.")
    if raw is None or isinstance(raw, bool):
        return None, missing
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None, missing
        try:
            x = float(text)
        except ValueError:
            return None, "Enter a number."
    elif isinstance(raw, (int, float)):
        x = float(raw)
    else:
        return None, "Enter a number."
    if not math.isfinite(x):
        return None, "Enter a finite number."
    if f.integer and not x.is_integer():
        return None, "Enter a whole number."
    if x < f.min or x > f.max:
        return None, f"Must be between {_fmt(f.min)} and {_fmt(f.max)}."
    return (int(x) if f.integer else x), None


def _parse_choice(raw: Any, f: Feature) -> tuple[Any, str | None]:
    if raw is None or raw == "":
        return None, "Choose an option."
    key = _value_key(raw) if isinstance(raw, (str, bool)) else None
    if key is None or key not in f.raw_by_value:
        return None, "Choose one of the listed options."
    return f.raw_by_value[key], None


def validate(payload: Any, bundle: ModelBundle) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate every feature. Returns (row in training order, {field: message})."""
    row: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for f in bundle.features:
        raw = payload.get(f.name)
        if f.kind == "numeric":
            value, err = _parse_number(raw, f)
        else:
            value, err = _parse_choice(raw, f)
        if err:
            errors[f.name] = err
        else:
            row[f.name] = value

    a, b = row.get("Weekly_GenAI_Hours"), row.get("Traditional_Study_Hours")
    if a is not None and b is not None and a + b > MAX_WEEK_HOURS:
        errors["Weekly_GenAI_Hours"] = (f"GenAI and traditional study hours together exceed "
                                        f"{MAX_WEEK_HOURS} hours in a week.")
    return row, errors


# --------------------------------------------------------------------------- #
# Prediction + explanation
# --------------------------------------------------------------------------- #
def _drivers(bundle: ModelBundle, frame: pd.DataFrame, pred_idx: int) -> list[dict[str, Any]]:
    """Which inputs pushed this prediction toward (or away from) the predicted class.

    For a multinomial logistic regression the logit of class k is coef_k . x + b_k.
    Each input's influence is its share of (logit_predicted - mean logit_other_classes),
    summed over the one-hot columns of that input. Pure arithmetic on the fitted
    coefficients: nothing is refit.
    """
    pre: ColumnTransformer = bundle.pipeline.named_steps["preprocessor"]
    clf = bundle.pipeline.steps[-1][1]
    coef = getattr(clf, "coef_", None)
    if coef is None or coef.shape[0] != len(bundle.classes):
        return []
    x = pre.transform(frame)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    x = x[0]
    others = [i for i in range(coef.shape[0]) if i != pred_idx]
    delta = (coef[pred_idx] - coef[others].mean(axis=0)) * x

    names = [str(n).split("__", 1)[-1] for n in pre.get_feature_names_out()]
    by_feature: dict[str, float] = {n: 0.0 for n in bundle.names}
    z_by_feature: dict[str, float] = {}
    for col_name, d, xv in zip(names, delta, x):
        owner = max((n for n in bundle.names if col_name == n or col_name.startswith(n + "_")),
                    key=len, default=None)
        if owner is None:
            continue
        by_feature[owner] += float(d)
        if col_name == owner:
            z_by_feature[owner] = float(xv)

    out = []
    for f in bundle.features:
        c = by_feature[f.name]
        if f.kind == "numeric":
            z = z_by_feature.get(f.name, 0.0)
            note = "Above average" if z >= 0.25 else "Below average" if z <= -0.25 else "Near average"
        else:
            note = next(o["label"] for o in f.options if o["value"] == _value_key(frame.iloc[0][f.name]))
        out.append(dict(feature=f.name, label=f.label, note=note, contribution=c,
                        direction="toward" if c >= 0 else "away"))
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out[:5]


def predict(bundle: ModelBundle, row: dict[str, Any]) -> dict[str, Any]:
    # Columns in the exact training order; categorical values are the model's own
    # category objects (including real booleans for Paid_Subscription).
    frame = pd.DataFrame([[row[n] for n in bundle.names]], columns=bundle.names)
    pipe = bundle.pipeline
    proba = pipe.predict_proba(frame)[0]
    label = str(pipe.predict(frame)[0])
    pred_idx = bundle.classes.index(label)
    probs = [dict(label=c, value=float(proba[bundle.classes.index(c)]), predicted=(c == label))
             for c in bundle.class_display_order]
    return dict(prediction=label,
                probability=float(proba[pred_idx]),
                summary=CLASS_SUMMARY.get(label, ""),
                probabilities=probs,
                drivers=_drivers(bundle, frame, pred_idx))


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=16 * 1024, JSON_SORT_KEYS=False,
                  SEND_FILE_MAX_AGE_DEFAULT=3600)
app.json.sort_keys = False

BUNDLE, MODEL_ERROR = load_model(MODEL_PATH)


def _model_facts() -> dict[str, Any]:
    if BUNDLE is None:
        return dict(loaded=False)
    n_num = sum(f.kind == "numeric" for f in BUNDLE.features)
    return dict(loaded=True, algorithm="Logistic regression", sklearn=sklearn.__version__,
                n_features=len(BUNDLE.features), n_numeric=n_num,
                n_categorical=len(BUNDLE.features) - n_num,
                classes=BUNDLE.class_display_order)


@app.context_processor
def helpers():
    def asset(filename: str) -> str:
        p = Path(app.static_folder) / filename
        return url_for("static", filename=filename, v=int(p.stat().st_mtime) if p.exists() else 0)
    return dict(asset=asset)


@app.after_request
def secure(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
    if request.path.startswith("/api/") or request.path == "/health":
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    groups, typical, client_schema = [], {}, []
    if BUNDLE is not None:
        for g in GROUPS:
            fields = [f.public() for f in BUNDLE.features if f.group == g["id"]]
            groups.append({**g, "fields": fields})
        typical = {f.name: f.typical for f in BUNDLE.features if f.typical is not None}
        client_schema = [f.public() for f in BUNDLE.features]
    return render_template("index.html", groups=groups, typical=typical, schema=client_schema,
                           facts=_model_facts(), model_error=MODEL_ERROR)


@app.get("/api/schema")
def api_schema():
    if BUNDLE is None:
        return jsonify(error="model_unavailable", message=MODEL_ERROR), 503
    return jsonify(features=[f.public() for f in BUNDLE.features], classes=BUNDLE.class_display_order,
                   model=_model_facts())


@app.post("/api/predict")
def api_predict():
    if BUNDLE is None:
        return jsonify(error="model_unavailable", message=MODEL_ERROR), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="bad_request", message="Send a JSON object with the 14 input fields."), 400
    row, errors = validate(payload, BUNDLE)
    if errors:
        n = len(errors)
        return jsonify(error="validation_failed",
                       message=f"Fix {n} field{'s' if n != 1 else ''} and try again.",
                       fields=errors), 422
    try:
        return jsonify(predict(BUNDLE, row))
    except Exception:  # noqa: BLE001
        log.exception("Prediction failed")
        return jsonify(error="prediction_failed",
                       message="The model could not score this profile. Please try again."), 500


@app.get("/health")
def health():
    ok = BUNDLE is not None
    return jsonify(status="ok" if ok else "degraded", model_loaded=ok,
                   sklearn=sklearn.__version__, detail=MODEL_ERROR), (200 if ok else 503)


def _wants_json() -> bool:
    return request.path.startswith("/api/")


@app.errorhandler(404)
def not_found(_e):
    if _wants_json():
        return jsonify(error="not_found", message="No such endpoint."), 404
    return render_template("error.html", code=404, title="Page not found",
                           message="That page doesn't exist."), 404


@app.errorhandler(405)
def bad_method(_e):
    if _wants_json():
        return jsonify(error="method_not_allowed", message="Method not allowed."), 405
    return render_template("error.html", code=405, title="Not allowed",
                           message="That action isn't supported here."), 405


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error="payload_too_large", message="Request is too large."), 413


@app.errorhandler(500)
def server_error(_e):
    if _wants_json():
        return jsonify(error="server_error", message="Something went wrong on our side."), 500
    return render_template("error.html", code=500, title="Something went wrong",
                           message="Please try again in a moment."), 500


if __name__ == "__main__":
    # Local development only. Production uses:  gunicorn app:app
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
