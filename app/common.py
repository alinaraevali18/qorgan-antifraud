"""Общие функции для страниц Streamlit: загрузка данных и стили."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA, MODELS = ROOT / "data", ROOT / "models"

ACCENT = "#0FA37F"  # зелёный в стиле кейса
WARN = "#E8A317"
DANGER = "#D64545"


def setup(title: str, icon: str = "🛡️"):
    st.set_page_config(page_title=f"Qorgan · {title}", page_icon=icon, layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.8rem;}
        .q-badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;
                  font-weight:600;color:white;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.markdown("### 🛡️ Qorgan")
        st.caption("Антифрод против социальной инженерии и сетей дропов")


def _ensure_built():
    if not (MODELS / "risk_model.joblib").exists() or not (DATA / "transactions.csv.gz").exists():
        with st.spinner("Первый запуск: генерируем данные и обучаем модель (~30 сек)…"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "build.py")], check=True)


@st.cache_resource
def load_model():
    _ensure_built()
    return joblib.load(MODELS / "risk_model.joblib")


@st.cache_data
def load_metrics() -> dict:
    _ensure_built()
    return json.loads((MODELS / "metrics.json").read_text())


@st.cache_data
def load_tx() -> pd.DataFrame:
    _ensure_built()
    return pd.read_csv(DATA / "transactions.csv.gz", parse_dates=["ts"])


@st.cache_data
def load_graph():
    _ensure_built()
    ledger = pd.read_csv(DATA / "ledger.csv.gz", parse_dates=["ts"])
    af = pd.read_csv(DATA / "account_features.csv.gz", index_col=0)
    rings = [set(r) for r in json.loads((DATA / "rings.json").read_text())]
    return ledger, af, rings


def fmt_kzt(x: float) -> str:
    return f"{x:,.0f} ₸".replace(",", " ")


def badge(decision: str) -> str:
    from qorgan.model import DECISION_RU, HOLD, PAUSE
    color = DANGER if decision == HOLD else WARN if decision == PAUSE else ACCENT
    return f'<span class="q-badge" style="background:{color}">{DECISION_RU[decision]}</span>'
