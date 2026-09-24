from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).parent.parent / "lab2" / "Titanic-Dataset.csv"
TARGET = "Survived"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
N_SPLITS = 5
RANDOM_STATE = 42

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_preprocessor():
    """สร้าง preprocessing ใหม่สำหรับแต่ละโมเดลและแต่ละ fold"""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def make_models():
    """สร้างโมเดล Bayes, Decision Tree และ KNN"""
    return {
        "Bayes": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("classifier", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def calculate_metrics(actual, predicted):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "F-Measure": f1_score(actual, predicted, zero_division=0),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    cross_validator = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    models = make_models()
    fold_metrics = {model_name: [] for model_name in models}

    print(f"Dataset: Titanic-Dataset.csv ({len(df)} rows)")
    print(f"Cross-validation: {N_SPLITS} folds, test {100 / N_SPLITS:.0f}% per fold")
    print(f"Features: {', '.join(FEATURES)}")
    print("Class: Survived (0 = ไม่รอด, 1 = รอด)\n")

    for fold_number, (train_indices, test_indices) in enumerate(
        cross_validator.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        predictions = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions[model_name] = model.predict(X_test)
            fold_metrics[model_name].append(
                calculate_metrics(y_test, predictions[model_name])
            )

        # ตาราง Actual/Predicted ของ fold นี้ รวมผลจากทั้ง 3 วิธี
        actual_table = pd.DataFrame(
            {
                "Row": test_indices + 1,
                "Actual": y_test.to_numpy(),
                "Bayes": predictions["Bayes"],
                "Decision Tree": predictions["Decision Tree"],
                "KNN": predictions["KNN"],
            }
        ).sort_values("Row")

        print(f"{'=' * 78}\nFOLD {fold_number}: Actual / Predicted Table")
        print(actual_table.to_string(index=False))
        print("\nMetrics")
        for model_name in models:
            metrics = fold_metrics[model_name][-1]
            print(
                f"{model_name:15} | "
                f"Accuracy: {metrics['Accuracy']:.4f} | "
                f"Recall: {metrics['Recall']:.4f} | "
                f"Precision: {metrics['Precision']:.4f} | "
                f"F-Measure: {metrics['F-Measure']:.4f}"
            )

    summary_rows = []
    for model_name, metrics_list in fold_metrics.items():
        summary = {
            metric_name: sum(metrics[metric_name] for metrics in metrics_list) / N_SPLITS
            for metric_name in ["Accuracy", "Recall", "Precision", "F-Measure"]
        }
        summary["Method"] = model_name
        summary_rows.append(summary)

    summary_table = pd.DataFrame(summary_rows).set_index("Method")
    print(f"\n{'=' * 78}\nAVERAGE RESULTS FROM 5 FOLDS")
    print(summary_table[["Accuracy", "Recall", "Precision", "F-Measure"]].round(4).to_string())


if __name__ == "__main__":
    main()
