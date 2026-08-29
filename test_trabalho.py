import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]


@pytest.fixture(scope="module")
def dataset_raw():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    return pd.read_csv(url, names=COLUMNS, sep=r",\s*", engine="python", na_values="?")


@pytest.fixture(scope="module")
def dataset_clean(dataset_raw):
    return dataset_raw.dropna().drop_duplicates()


@pytest.fixture(scope="module")
def model_data(dataset_clean):
    df = dataset_clean.copy()
    df["target"] = (df["income"] == ">50K").astype(int)

    features_num = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    features_cat = ["workclass", "marital_status", "occupation", "relationship", "race", "sex"]

    df_encoded = pd.get_dummies(df[features_num + features_cat], drop_first=True)

    X = df_encoded.values
    y = df["target"].values
    sex = df["sex"].values

    X_train, X_test, y_train, y_test, sex_train, sex_test = train_test_split(
        X, y, sex, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Modelo A: Regressão logística completa
    clf_a = LogisticRegression(max_iter=1000, random_state=42)
    clf_a.fit(X_train_s, y_train)
    pred_a = clf_a.predict(X_test_s)

    # Modelo B: Regressão logística simples (somente variáveis numéricas)
    X_train_num = X_train[:, :len(features_num)]
    X_test_num = X_test[:, :len(features_num)]
    scaler_num = StandardScaler()
    clf_b = LogisticRegression(max_iter=1000, random_state=42)
    clf_b.fit(scaler_num.fit_transform(X_train_num), y_train)
    pred_b = clf_b.predict(scaler_num.transform(X_test_num))

    return {
        "y_true": y_test,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "sex_test": sex_test,
    }


# Validadores de dados (Aula 3)

def validate_schema(df, expected_columns):
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    for col, expected_type in expected_columns.items():
        actual_type = str(df[col].dtype)
        if expected_type in ("object", "string", "str") and actual_type in ("object", "string", "str"):
            continue
        if actual_type != expected_type:
            raise ValueError(f"Coluna {col} com tipo {actual_type}, esperado {expected_type}")
    return True


def check_no_missing_values(df, columns):
    for col in columns:
        if df[col].isna().any():
            raise ValueError(f"Valores nulos encontrados na coluna {col}")
    return True


def check_range(df, column, lo, hi):
    if not df[column].dropna().between(lo, hi).all():
        raise ValueError(f"Coluna {column} com valores fora da faixa [{lo}, {hi}]")
    return True


def check_unique_ids(df, id_column):
    if df[id_column].duplicated().any():
        raise ValueError(f"Linhas duplicadas identificadas em {id_column}")
    return True


# Funções de testes estatísticos e fairness (Aula 4)

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (y_true == y_pred).mean()


def bootstrap_ci(y_true, y_pred, metric_fn=accuracy, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    metrics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        metrics.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = np.percentile(metrics, (1 - ci) / 2 * 100)
    hi = np.percentile(metrics, (1 + ci) / 2 * 100)
    return lo, hi


def compare_models(y_true, pred_a, pred_b, metric_fn=accuracy, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx]))
    lo = np.percentile(diffs, (1 - ci) / 2 * 100)
    hi = np.percentile(diffs, (1 + ci) / 2 * 100)
    return lo, hi


def positive_rate(y_pred, group):
    y_pred_arr = np.array(y_pred)
    group_arr = np.array(group)
    return {g: float(y_pred_arr[group_arr == g].mean()) for g in sorted(set(group))}


def check_demographic_parity(y_pred, group, max_gap=0.10):
    rates = positive_rate(y_pred, group)
    gap = max(rates.values()) - min(rates.values())
    if gap > max_gap:
        raise ValueError(f"Diferença de paridade demográfica ({gap:.4f}) excede limite ({max_gap}): {rates}")
    return True


# Testes pytests

def test_schema_validacao(dataset_clean):
    expected = {"age": "int64", "education_num": "int64", "hours_per_week": "int64", "sex": "object"}
    assert validate_schema(dataset_clean, expected) is True


def test_completude_dataset_bruto(dataset_raw):
    with pytest.raises(ValueError, match="Valores nulos"):
        check_no_missing_values(dataset_raw, ["workclass", "occupation", "native_country"])


def test_faixa_valores(dataset_clean):
    assert check_range(dataset_clean, "age", 17, 90) is True
    assert check_range(dataset_clean, "hours_per_week", 1, 99) is True


def test_intervalo_confianca_acuracia(model_data):
    y_true = model_data["y_true"]
    pred_a = model_data["pred_a"]
    acc = accuracy(y_true, pred_a)
    lo, hi = bootstrap_ci(y_true, pred_a, n_boot=1000)
    assert lo <= acc <= hi


def test_comparacao_modelos(model_data):
    y_true = model_data["y_true"]
    pred_a = model_data["pred_a"]
    pred_b = model_data["pred_b"]
    lo, hi = compare_models(y_true, pred_a, pred_b, n_boot=1000)
    assert lo > 0  # diferença estatisticamente significativa a favor do modelo completo


def test_paridade_demografica_genero(model_data):
    pred_a = model_data["pred_a"]
    sex_test = model_data["sex_test"]
    with pytest.raises(ValueError, match="paridade demográfica"):
        check_demographic_parity(pred_a, sex_test, max_gap=0.10)
