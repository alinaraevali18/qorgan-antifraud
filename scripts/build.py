"""Генерирует данные, обучает модель, считает метрики и сохраняет всё для демо.

Запуск:  python scripts/build.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qorgan.data_gen import GenConfig, generate  # noqa: E402
from qorgan.features import build_features  # noqa: E402
from qorgan.graph import account_features, evaluate_radar, find_rings  # noqa: E402
from qorgan.model import RiskModel, evaluate  # noqa: E402

DATA, MODELS = ROOT / "data", ROOT / "models"


def main():
    DATA.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    cfg = GenConfig()
    clients, accounts, tx, ledger = generate(cfg)
    df = build_features(tx, clients, accounts)

    # разбиение по времени: учимся на первых 30 днях, проверяем на последних 15
    split = pd.Timestamp(cfg.start) + pd.Timedelta(days=30)
    train, test = df[df.ts < split], df[df.ts >= split]
    model = RiskModel().fit(train)
    metrics = evaluate(model, test)
    metrics["thresholds"] = dict(pause=model.t_pause, hold=model.t_hold)
    metrics["split"] = str(split.date())
    metrics["n_train"], metrics["n_train_fraud"] = len(train), int(train.is_fraud.sum())

    df["risk"] = model.score(df)
    df["decision"] = model.decide(df.risk.to_numpy())

    af = account_features(ledger, df.set_index("tx_id").risk, accounts, model.t_pause)
    rings = find_rings(ledger, af)
    metrics["radar"] = evaluate_radar(af)
    metrics["radar"]["rings_found"] = len(rings)

    clients.to_csv(DATA / "clients.csv.gz", index=False)
    accounts.to_csv(DATA / "accounts.csv.gz", index=False)
    df.to_csv(DATA / "transactions.csv.gz", index=False)
    ledger.to_csv(DATA / "ledger.csv.gz", index=False)
    af.to_csv(DATA / "account_features.csv.gz")
    (DATA / "rings.json").write_text(json.dumps([sorted(r) for r in rings], ensure_ascii=False))
    joblib.dump(model, MODELS / "risk_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    m, r = metrics["model"], metrics["rules"]
    print(f"Переводов: {len(df)}, мошеннических: {int(df.is_fraud.sum())}")
    print(f"ROC-AUC {metrics['roc_auc']:.3f}  PR-AUC {metrics['pr_auc']:.3f}")
    print(f"Qorgan: recall {m['recall']:.1%}, precision {m['precision']:.1%}, "
          f"ложных тревог {m['false_alarm_rate']:.2%}, спасено {m['saved_share']:.1%} суммы")
    print(f"Правила: recall {r['recall']:.1%}, precision {r['precision']:.1%}, "
          f"ложных тревог {r['false_alarm_rate']:.2%}, спасено {r['saved_share']:.1%} суммы")
    print("По схемам:", metrics["by_scheme"])
    print("Правила по схемам:", metrics["rules_by_scheme"])
    print("Дроп-радар:", metrics["radar"])


if __name__ == "__main__":
    main()
