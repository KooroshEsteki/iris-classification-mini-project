import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

DATA_PATH = "iris.txt"

column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=column_names,
    sep=","
)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nClass distribution:")
print(df["class"].value_counts())

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = df["class"]

print("\nFeature matrix shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=7,
    stratify=y
)

print("\nTraining set:")
print(X_train.shape, y_train.shape)

print("\nTesting set:")
print(X_test.shape, y_test.shape)

models = [
    (
        "Logistic Regression",
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000
                ))
            ]
        )
    ),
    (
        "KNN Classification",
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5))
            ]
        )
    ),
    (
        "Decision Tree Classification",
        DecisionTreeClassifier(random_state=7)
    ),
    (
        "Random Forest Classification",
        RandomForestClassifier(
            n_estimators=100,
            random_state=7
        )
    ),
    (
        "Support Vector Machine",
        Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    gamma="scale",
                    random_state=7
                ))
            ]
        )
    ),
    (
        "Gaussian Naive Bayes",
        GaussianNB()
    )
]

results = []

for model_name, model in models:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy
        }
    )

    print("\n====================================================")
    print(model_name)
    print("====================================================")

    print("Accuracy:")
    print(round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

results_df = pd.DataFrame(results)

print("\n====================================================")
print("Final Accuracy Comparison")
print("====================================================")

print(results_df.sort_values(by="Accuracy", ascending=False))

best_model = results_df.loc[results_df["Accuracy"].idxmax()]

print("\nBest model:")
print(best_model)

plt.figure(figsize=(11, 6))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy")
plt.title("Algorithm Comparison")
plt.xticks(rotation=35, ha="right")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("algorithm_comparison.png", dpi=300)
plt.show()

print("\nAlgorithm comparison plot saved as:")
print("algorithm_comparison.png")
