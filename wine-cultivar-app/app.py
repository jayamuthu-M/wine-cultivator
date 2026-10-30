import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Wine Cultivar Classifier", page_icon="🍷", layout="wide")

@st.cache_resource
def train_model():
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, random_state=42)
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, wine, accuracy

model, wine, accuracy = train_model()
class_names = [f"Cultivar {i + 1}" for i in range(3)]

st.title("🍷 Wine Cultivar Classification")
st.write("Enter the chemical measurements of a wine sample to predict its cultivar class.")
st.caption("Demo project using scikit-learn's built-in Wine dataset. The cultivar labels are dataset classes, not commercial wine-variety recommendations.")

with st.sidebar:
    st.header("About the model")
    st.metric("Test accuracy", f"{accuracy:.1%}")
    st.write("Algorithm: Random Forest")
    st.write(f"Training samples: {int(len(wine.data) * 0.8)}")
    st.write(f"Input features: {wine.data.shape[1]}")

st.subheader("Sample measurements")
defaults = wine.data.median()
feature_descriptions = {
    "alcohol": "Alcohol",
    "malic_acid": "Malic acid",
    "ash": "Ash",
    "alcalinity_of_ash": "Alcalinity of ash",
    "magnesium": "Magnesium",
    "total_phenols": "Total phenols",
    "flavanoids": "Flavanoids",
    "nonflavanoid_phenols": "Nonflavanoid phenols",
    "proanthocyanins": "Proanthocyanins",
    "color_intensity": "Color intensity",
    "hue": "Hue",
    "od280/od315_of_diluted_wines": "OD280/OD315 of diluted wines",
    "proline": "Proline",
}

with st.form("prediction_form"):
    cols = st.columns(3)
    values = {}
    for i, feature in enumerate(wine.data.columns):
        with cols[i % 3]:
            low = float(wine.data[feature].min())
            high = float(wine.data[feature].max())
            default = float(defaults[feature])
            values[feature] = st.number_input(
                feature_descriptions.get(feature, feature.replace("_", " ").title()),
                min_value=low,
                max_value=high,
                value=default,
                help=f"Observed dataset range: {low:.3g} to {high:.3g}"
            )
    submitted = st.form_submit_button("Predict cultivar", type="primary")

if submitted:
    sample = pd.DataFrame([values], columns=wine.data.columns)
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    st.divider()
    st.subheader("Prediction result")
    st.success(f"Predicted class: {class_names[prediction]}")
    st.write("Model confidence estimates:")
    probability_df = pd.DataFrame({
        "Cultivar class": class_names,
        "Estimated probability": probabilities
    }).sort_values("Estimated probability", ascending=False)
    st.bar_chart(probability_df.set_index("Cultivar class"))
    st.dataframe(probability_df.style.format({"Estimated probability": "{:.1%}"}), use_container_width=True)

with st.expander("View dataset information"):
    st.write("The dataset contains 178 samples, 13 numeric chemical features, and 3 target classes.")
    st.dataframe(wine.data.head(), use_container_width=True)
    st.write("Feature names:", ", ".join(wine.data.columns))
