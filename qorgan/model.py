"""Модель риска перевода, базовая линия на правилах и объяснения решений."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .features import FEATURES, FEATURE_NAMES_RU

# Решения антифрода
ALLOW = "allow"
PAUSE = "smart_pause"  # «умная пауза»: предупреждение + вопросы + задержка
HOLD = "hold"          # перевод удержан, звонок из банка по официальному номеру

DECISION_RU = {ALLOW: "Пропустить", PAUSE: "Умная пауза", HOLD: "Удержать и перезвонить"}


def rules_baseline(df: pd.DataFrame) -> np.ndarray:
    """Типичные «классические» правила: крупная сумма новому получателю или новое устройство."""
    big_new = (df.amount_to_avg > 10) & (df.is_new_recipient == 1)
    new_dev_big = (df.is_new_device == 1) & (df.amount_to_avg > 5)
    return (big_new | new_dev_big).to_numpy().astype(int)


@dataclass
class RiskModel:
    clf: HistGradientBoostingClassifier = None
    t_pause: float = 0.3
    t_hold: float = 0.8
    reference: dict = field(default_factory=dict)  # «типичные» значения признаков у честных переводов

    def fit(self, df: pd.DataFrame, max_fpr_pause=0.01, max_fpr_hold=0.0015):
        X, y = df[FEATURES], df.is_fraud.astype(int)
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight={0: 1, 1: 20}, random_state=0,
        ).fit(X, y)
        self.reference = df.loc[~df.is_fraud.astype(bool), FEATURES].median().to_dict()
        # пороги по допустимой доле «ложных тревог» среди честных переводов
        s = self.score(df)[~y.astype(bool).to_numpy()]
        self.t_pause = float(np.quantile(s, 1 - max_fpr_pause))
        self.t_hold = float(max(np.quantile(s, 1 - max_fpr_hold), self.t_pause + 0.05))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[FEATURES])[:, 1]

    def decide(self, scores: np.ndarray) -> np.ndarray:
        return np.where(scores >= self.t_hold, HOLD, np.where(scores >= self.t_pause, PAUSE, ALLOW))

    def explain(self, row, top: int = 5, n_perm: int = 64) -> list[tuple[str, float]]:
        """Вклад признаков в риск по значениям Шепли (как в SHAP), в шкале log-odds.

        Точка отсчёта — «типичный честный перевод». Признаки по одному в случайном
        порядке заменяются на значения текущего перевода; средний прирост log-odds
        и есть вклад признака. Возвращаем доли от суммарного роста риска.
        """
        x = np.array([float(row[f]) for f in FEATURES])
        ref = np.array([float(self.reference[f]) for f in FEATURES])
        rng = np.random.default_rng(0)
        n_f = len(FEATURES)
        states, orders = [], []
        for _ in range(n_perm):
            order = rng.permutation(n_f)
            cur = ref.copy()
            states.append(cur.copy())
            for j in order:
                cur[j] = x[j]
                states.append(cur.copy())
            orders.append(order)
        p = np.clip(self.clf.predict_proba(pd.DataFrame(states, columns=FEATURES))[:, 1], 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p)).reshape(n_perm, n_f + 1)
        phi = np.zeros(n_f)
        for k, order in enumerate(orders):
            phi[order] += np.diff(logit[k])
        phi /= n_perm
        pos = phi.clip(min=0)
        if pos.sum() <= 0:
            return []
        share = pos / pos.sum()
        out = [(FEATURE_NAMES_RU[f], float(v)) for f, v in zip(FEATURES, share) if v >= 0.03]
        return sorted(out, key=lambda t: -t[1])[:top]


def evaluate(model: RiskModel, test: pd.DataFrame) -> dict:
    y = test.is_fraud.astype(int).to_numpy()
    s = model.score(test)
    d = model.decide(s)
    flagged = d != ALLOW
    rules = rules_baseline(test)
    fraud_amt = test.amount[y == 1].sum()

    def pack(flag):
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        return dict(
            recall=tp / max(y.sum(), 1),
            precision=tp / max(tp + fp, 1),
            false_alarms=fp,
            false_alarm_rate=fp / max((y == 0).sum(), 1),
            saved_amount=float(test.amount[flag & (y == 1)].sum()),
            saved_share=float(test.amount[flag & (y == 1)].sum() / max(fraud_amt, 1)),
        )

    prec, rec, thr = precision_recall_curve(y, s)
    return dict(
        n=len(test), n_fraud=int(y.sum()), fraud_amount=float(fraud_amt),
        roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
        model=pack(flagged), rules=pack(rules.astype(bool)),
        hold=pack(d == HOLD), pause=pack(d == PAUSE),
        by_scheme=(test.assign(caught=flagged)[test.is_fraud.astype(bool)]
                   .groupby("scheme").caught.mean().round(3).to_dict()),
        rules_by_scheme=(test.assign(caught=rules.astype(bool))[test.is_fraud.astype(bool)]
                         .groupby("scheme").caught.mean().round(3).to_dict()),
        pr_curve=dict(precision=prec[::max(len(prec) // 300, 1)].tolist(),
                      recall=rec[::max(len(rec) // 300, 1)].tolist()),
        importance=None,
    )
