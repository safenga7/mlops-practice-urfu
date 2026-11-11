import os
import pickle
import pytest
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

MODEL_PATH = "linear_regression_model.pkl"
QUALITY_THRESHOLD_R2 = 0.85
QUALITY_THRESHOLD_MSE = 2.0

@pytest.fixture(scope="module")
def model():
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

def evaluate_dataset(model, csv_path):
    assert os.path.exists(csv_path), f"Dataset file not found: {csv_path}"
    df = pd.read_csv(csv_path)
    X = df[['x']].values
    y = df['y'].values
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return r2, mse

def test_clean_data_1(model):
    """Тест качества на первом чистом датасете (train/baseline)."""
    r2, mse = evaluate_dataset(model, "data/clean_data_1.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, f"Clean 1 R2 below threshold: {r2:.4f} < {QUALITY_THRESHOLD_R2}"
    assert mse <= QUALITY_THRESHOLD_MSE, f"Clean 1 MSE above threshold: {mse:.4f} > {QUALITY_THRESHOLD_MSE}"

def test_clean_data_2(model):
    """Тест качества на втором чистом датасете (validation 1)."""
    r2, mse = evaluate_dataset(model, "data/clean_data_2.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, f"Clean 2 R2 below threshold: {r2:.4f} < {QUALITY_THRESHOLD_R2}"
    assert mse <= QUALITY_THRESHOLD_MSE, f"Clean 2 MSE above threshold: {mse:.4f} > {QUALITY_THRESHOLD_MSE}"

def test_clean_data_3(model):
    """Тест качества на третьем чистом датасете (validation 2)."""
    r2, mse = evaluate_dataset(model, "data/clean_data_3.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, f"Clean 3 R2 below threshold: {r2:.4f} < {QUALITY_THRESHOLD_R2}"
    assert mse <= QUALITY_THRESHOLD_MSE, f"Clean 3 MSE above threshold: {mse:.4f} > {QUALITY_THRESHOLD_MSE}"

def test_noised_data_quality_detection(model):
    """Тест обнаружения деградации качества на зашумленном датасете.
    На зашумленных данных метрики не должны удовлетворять SLA чистого датасета.
    """
    r2, mse = evaluate_dataset(model, "data/noised_data.csv")
    assert r2 >= QUALITY_THRESHOLD_R2, (
        f"DATA QUALITY ANOMALY DETECTED: R2={r2:.4f} < threshold ({QUALITY_THRESHOLD_R2}), "
        f"MSE={mse:.4f} > threshold ({QUALITY_THRESHOLD_MSE})"
    )
