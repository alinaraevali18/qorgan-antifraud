"""Быстрые проверки пайплайна на уменьшенных данных."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qorgan.data_gen import GenConfig, generate  # noqa: E402
from qorgan.features import FEATURES, build_features  # noqa: E402
from qorgan.graph import account_features, evaluate_radar  # noqa: E402
from qorgan.model import ALLOW, HOLD, RiskModel, evaluate  # noqa: E402
from qorgan.policy import SCHEME_WARNINGS, TEXTS  # noqa: E402


@pytest.fixture(scope="module")
def built():
    cfg = GenConfig(seed=7)  # другой seed, чем в демо
    clients, accounts, tx, ledger = generate(cfg)
    df = build_features(tx, clients, accounts)
    split = pd.Timestamp(cfg.start) + pd.Timedelta(days=30)
    train, test = df[df.ts < split], df[df.ts >= split]
    model = RiskModel().fit(train)
    return dict(df=df, train=train, test=test, model=model, ledger=ledger, accounts=accounts)


def test_no_label_leakage():
    assert "is_fraud" not in FEATURES and "scheme" not in FEATURES
    # демографию в модель не берём
    assert not {"age", "city", "owner_age"} & set(FEATURES)


def test_features_are_causal(built):
    df = built["df"]
    first = df.sort_values("ts").groupby("client_id").head(1)
    assert (first.client_tx_1h == 0).all()


def test_model_beats_rules(built):
    m = evaluate(built["model"], built["test"])
    assert m["model"]["recall"] > m["rules"]["recall"]
    assert m["model"]["false_alarm_rate"] < 0.03


def test_decisions_monotonic(built):
    model = built["model"]
    d = model.decide(np.array([0.0, model.t_pause, model.t_hold, 1.0]))
    assert d[0] == ALLOW and d[-1] == HOLD


def test_explain_returns_shares(built):
    row = built["test"][built["test"].is_fraud].iloc[0]
    reasons = built["model"].explain(row)
    assert reasons and all(0 < s <= 1 for _, s in reasons)


def test_radar_finds_mules(built):
    tx_risk = pd.Series(built["model"].score(built["df"]), index=built["df"].tx_id)
    af = account_features(built["ledger"], tx_risk, built["accounts"], built["model"].t_pause)
    r = evaluate_radar(af)
    assert r["recall"] > 0.8 and r["precision"] > 0.6


def test_texts_bilingual():
    assert TEXTS["ru"].keys() == TEXTS["kz"].keys()
    for w in SCHEME_WARNINGS.values():
        assert w.keys() == {"ru", "kz"}
