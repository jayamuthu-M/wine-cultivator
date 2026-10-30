# Wine Cultivar Classification App

A beginner-friendly machine-learning project that predicts one of three classes in scikit-learn's built-in Wine dataset from 13 chemical measurements.

## Requirements
- Python 3.10 or newer
- VS Code with the Python extension

## Run in VS Code

1. Extract the ZIP and open the `wine-cultivar-app` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create a virtual environment (recommended):

   **Windows**
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Start the web app:
   ```bash
   streamlit run app.py
   ```
6. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Train/evaluate from the terminal

```bash
python train_model.py
```

## Project structure

- `app.py` — Streamlit interface and interactive predictions.
- `train_model.py` — model training and evaluation script.
- `requirements.txt` — Python dependencies.

## Dataset and limitations

The app uses the Wine recognition dataset included with scikit-learn, so no separate CSV download is required. It has 178 samples, 13 numeric features, and three target classes. The class labels are shown as Cultivar 1–3 for clarity. The displayed test accuracy is measured on a held-out split; results may vary if the split or model changes. Predictions are educational demonstrations and should not be used for real-world wine authentication.
