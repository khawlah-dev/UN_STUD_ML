from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import json
import pickle

import pandas as pd
from flask import Flask, redirect, render_template, request, url_for


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "saved_best_model_pipeline.pkl"
RESULTS_PATH = PROJECT_DIR / "results" / "tree_model_details.json"
FEATURES_PATH = PROJECT_DIR / "results" / "best_tree_feature_importance.csv"

app = Flask(__name__)

NUMERIC_FIELDS = {
    "Age": {"label": "العمر", "min": 18, "max": 24, "step": 1, "placeholder": "مثال: 21", "hint": "من 18 إلى 24"},
    "CGPA": {"label": "المعدل التراكمي (CGPA)", "min": 1.5, "max": 4.0, "step": 0.01, "placeholder": "مثال: 3.20", "hint": "من 1.5 إلى 4.0"},
    "Sleep_Duration": {"label": "ساعات النوم يوميا", "min": 3, "max": 12, "step": 0.1, "placeholder": "مثال: 7", "hint": "من 3 إلى 12 ساعة"},
    "Study_Hours": {"label": "ساعات الدراسة يوميا", "min": 0, "max": 13, "step": 0.1, "placeholder": "مثال: 4", "hint": "من 0 إلى 13 ساعة"},
    "Social_Media_Hours": {"label": "ساعات السوشال ميديا يوميا", "min": 0, "max": 10, "step": 0.1, "placeholder": "مثال: 3", "hint": "من 0 إلى 10 ساعات"},
    "Physical_Activity": {"label": "النشاط البدني (دقيقة/أسبوع)", "min": 0, "max": 150, "step": 1, "placeholder": "مثال: 60", "hint": "من 0 إلى 150 دقيقة"},
    "Stress_Level": {"label": "مستوى التوتر", "min": 1, "max": 10, "step": 1, "placeholder": "مثال: 6", "hint": "من 1 منخفض إلى 10 مرتفع"},
}

GENDERS = [("Female", "أنثى"), ("Male", "ذكر")]
DEPARTMENTS = [
    ("Arts", "الآداب"),
    ("Business", "إدارة الأعمال"),
    ("Engineering", "الهندسة"),
    ("Medical", "الطب"),
    ("Science", "العلوم"),
]

DEFAULT_FORM = {
    "Age": 21,
    "Gender": "Female",
    "Department": "Science",
    "CGPA": 2.9,
    "Sleep_Duration": 7,
    "Study_Hours": 4.5,
    "Social_Media_Hours": 3.5,
    "Physical_Activity": 75,
    "Stress_Level": 4,
}

FEATURE_LABELS = {
    "Age": "العمر",
    "CGPA": "المعدل التراكمي (CGPA)",
    "Sleep_Duration": "ساعات النوم",
    "Study_Hours": "ساعات الدراسة",
    "Social_Media_Hours": "ساعات السوشال ميديا",
    "Physical_Activity": "النشاط البدني",
    "Stress_Level": "مستوى التوتر",
    "Gender": "الجنس",
    "Department": "القسم الدراسي",
}


def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "غير معروف"


def format_model_name(model_type):
    if model_type == "RandomForestClassifier":
        return "Random Forest Classifier"
    if model_type == "DecisionTreeClassifier":
        return "Decision Tree Classifier"
    return model_type or "غير معروف"


def load_feature_importance():
    if not FEATURES_PATH.exists():
        return []

    rows = pd.read_csv(FEATURES_PATH)
    grouped = {}

    for _, row in rows.iterrows():
        feature = row["feature"].replace("num__", "").replace("cat__", "")
        if feature.startswith("Gender_"):
            feature = "Gender"
        elif feature.startswith("Department_"):
            feature = "Department"

        grouped[feature] = grouped.get(feature, 0) + float(row["importance"])

    if not grouped:
        return []

    max_value = max(grouped.values())
    items = []
    for feature, importance in grouped.items():
        items.append(
            {
                "label": FEATURE_LABELS.get(feature, feature),
                "percent": round(importance * 100, 1),
                "width": round((importance / max_value) * 100, 1),
            }
        )

    return sorted(items, key=lambda item: item["percent"], reverse=True)


def load_model_info(trained_model):
    if not RESULTS_PATH.exists():
        return None

    with RESULTS_PATH.open(encoding="utf-8") as file:
        details = json.load(file)

    best_name = details.get("best_experiment")
    best_result = next(
        (item for item in details.get("results", []) if item.get("experiment") == best_name),
        {},
    )
    forest = trained_model.named_steps.get("model")
    matrix = best_result.get("test_confusion_matrix") or [[0, 0], [0, 0]]
    total_test = sum(sum(row) for row in matrix)

    return {
        "name": best_name or "غير معروف",
        "model_type": format_model_name(best_result.get("model_type")),
        "accuracy": best_result.get("test_accuracy"),
        "recall": best_result.get("test_recall_class_1"),
        "f1": best_result.get("test_f1_class_1"),
        "train_samples": best_result.get("train_samples"),
        "test_samples": total_test,
        "confusion_matrix": {
            "tn": matrix[0][0],
            "fp": matrix[0][1],
            "fn": matrix[1][0],
            "tp": matrix[1][1],
        },
        "library": f"scikit-learn {get_package_version('scikit-learn')}",
        "n_estimators": getattr(forest, "n_estimators", None),
        "max_depth": getattr(forest, "max_depth", None),
        "min_samples_leaf": getattr(forest, "min_samples_leaf", None),
        "class_weight": getattr(forest, "class_weight", None),
        "feature_importance": load_feature_importance(),
    }


model = load_model()
model_info = load_model_info(model)


def read_form_data():
    values = {}
    errors = []

    for field, config in NUMERIC_FIELDS.items():
        raw_value = request.form.get(field, "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"قيمة {config['label']} غير صحيحة.")
            continue

        if value < config["min"] or value > config["max"]:
            errors.append(
                f"{config['label']} لازم تكون بين {config['min']} و {config['max']}."
            )
        values[field] = value

    values["Gender"] = request.form.get("Gender", DEFAULT_FORM["Gender"])
    values["Department"] = request.form.get("Department", DEFAULT_FORM["Department"])

    if values["Gender"] not in dict(GENDERS):
        errors.append("اختيار الجنس غير صحيح.")
    if values["Department"] not in dict(DEPARTMENTS):
        errors.append("اختيار القسم غير صحيح.")

    return values, errors


def predict_student(values):
    row = pd.DataFrame([values])
    prediction = int(model.predict(row)[0])
    probability = None

    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            probability = float(model.predict_proba(row)[0][classes.index(1)])

    has_risk = prediction == 1
    return {
        "has_risk": has_risk,
        "title": "يوجد احتمال أعلى" if has_risk else "الاحتمال منخفض",
        "message": (
            "النموذج يتوقع وجود مؤشرات اكتئاب أعلى عند الطالب."
            if has_risk
            else "النموذج يتوقع أن مؤشرات الاكتئاب منخفضة عند الطالب."
        ),
        "probability": round(probability * 100, 1) if probability is not None else None,
    }


def render_predict_page(form=None, result=None, errors=None):
    return render_template(
        "index.html",
        fields=NUMERIC_FIELDS,
        genders=GENDERS,
        departments=DEPARTMENTS,
        form=form or DEFAULT_FORM,
        result=result,
        errors=errors or [],
    )


@app.get("/")
def index():
    return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_predict_page()

    values, errors = read_form_data()
    form = {**DEFAULT_FORM, **values}
    result = None if errors else predict_student(values)

    return render_predict_page(form=form, result=result, errors=errors)


@app.get("/about")
def about():
    return render_template("about.html", model_info=model_info)


if __name__ == "__main__":
    app.run(debug=True)
