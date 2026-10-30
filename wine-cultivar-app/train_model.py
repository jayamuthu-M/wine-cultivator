"""Train and evaluate a wine cultivar classifier."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42)
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=wine.target_names.astype(str)))

if __name__ == "__main__":
    main()
