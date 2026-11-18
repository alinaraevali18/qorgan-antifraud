"""Дроп-радар: поиск счетов-дропов и их сетей по графу переводов."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def account_features(ledger: pd.DataFrame, tx_risk: pd.Series, accounts: pd.DataFrame,
                     risk_threshold: float) -> pd.DataFrame:
    """Поведенческий профиль каждого счёта-получателя.

    tx_risk — риск-скор переводов клиентов (индекс = tx_id).
    """
    led = ledger.copy()
    led["risk"] = led.tx_id.map(tx_risk)
    led["risky"] = led.risk >= risk_threshold

    inc = led.groupby("dst").agg(
        in_count=("amount", "size"), in_amount=("amount", "sum"),
        in_unique_senders=("src", "nunique"), risky_in=("risky", "sum"),
        first_in=("ts", "min"),
    )
    out = led.groupby("src").agg(out_count=("amount", "size"), out_amount=("amount", "sum"),
                                 out_unique=("dst", "nunique"))
    f = inc.join(out, how="left").fillna({"out_count": 0, "out_amount": 0, "out_unique": 0})
    f.index.name = "account_id"

    # «быстрый транзит»: какая доля входящих денег ушла дальше в течение 24 часов
    led_sorted = led.sort_values("ts")
    outs = (led_sorted[["src", "ts", "amount"]]
            .rename(columns={"src": "account_id", "ts": "out_ts", "amount": "out_amt"}))
    ins = (led_sorted[["dst", "ts", "amount"]]
           .rename(columns={"dst": "account_id", "ts": "in_ts", "amount": "in_amt"}))
    m = pd.merge_asof(ins, outs, left_on="in_ts", right_on="out_ts", by="account_id",
                      direction="forward", tolerance=pd.Timedelta(hours=24))
    m["fwd"] = np.minimum(m.in_amt, m.out_amt.fillna(0))
    m["hold_h"] = (m.out_ts - m.in_ts).dt.total_seconds() / 3600
    agg = m.groupby("account_id").agg(fwd=("fwd", "sum"), median_hold_h=("hold_h", "median"))
    f["fast_forward_share"] = (agg.fwd.reindex(f.index).fillna(0) / f.in_amount).clip(upper=1)
    f["median_hold_h"] = agg.median_hold_h.reindex(f.index)
    f["risky_in_share"] = f.risky_in / f.in_count

    meta = accounts.set_index("account_id")
    f["account_age_days"] = meta.account_age_days.reindex(f.index)
    f["owner_age"] = meta.owner_age.reindex(f.index)
    f["kind"] = meta.kind.reindex(f.index).fillna("cashout")
    f["ring"] = meta["ring"].reindex(f.index) if "ring" in meta else np.nan

    senders = (f.in_unique_senders / 8).clip(upper=1)
    # транзит важен, только когда деньги приходят от многих разных людей
    many = (f.in_unique_senders >= 3).astype(float)
    young = (f.account_age_days.fillna(9999) < 180).astype(float)
    s1 = (0.35 * f.risky_in_share.clip(upper=1) + 0.35 * f.fast_forward_share * many
          + 0.20 * senders + 0.10 * young)

    # второй проход: доля входящих денег от уже подозрительных счетов
    # (так находим сборщиков, которые получают деньги не от жертв, а от дропов)
    sus = set(s1.index[s1 >= 0.5])
    from_sus = led[led.src.isin(sus)].groupby("dst").amount.sum()
    f["from_suspicious_share"] = (from_sus.reindex(f.index).fillna(0) / f.in_amount).clip(upper=1)
    f["mule_score"] = (s1 + 0.35 * f.from_suspicious_share).clip(upper=1).round(3)
    # обнальные узлы — не счета клиентов, их не скорим
    f.loc[f.index.str.startswith("CASHOUT"), "mule_score"] = np.nan
    return f.sort_values("mule_score", ascending=False)


def find_rings(ledger: pd.DataFrame, af: pd.DataFrame, threshold: float = 0.5) -> list[set]:
    """Сети дропов: связные компоненты среди подозрительных счетов и обнальных узлов."""
    sus = set(af.index[af.mule_score >= threshold])
    edges = ledger[ledger.src.isin(sus) | ledger.dst.isin(sus)]
    g = nx.Graph()
    for s, d in zip(edges.src, edges.dst):
        if (s in sus or s.startswith("CASHOUT")) and (d in sus or d.startswith("CASHOUT")):
            g.add_edge(s, d)
    # обнальные узлы общие для всех — не даём им склеивать разные сети
    g.remove_nodes_from([n for n in list(g) if n.startswith("CASHOUT")])
    rings = [c for c in nx.connected_components(g) if len(c) >= 2]
    return sorted(rings, key=len, reverse=True)


def ring_subgraph(ledger: pd.DataFrame, ring: set) -> pd.DataFrame:
    """Все переводы в сеть, внутри сети и из неё (для визуализации)."""
    e = ledger[ledger.src.isin(ring) | ledger.dst.isin(ring)]
    return e.groupby(["src", "dst"], as_index=False).agg(amount=("amount", "sum"), n=("amount", "size"))


def evaluate_radar(af: pd.DataFrame, threshold: float = 0.5) -> dict:
    scored = af.dropna(subset=["mule_score"])
    truth = scored.kind.isin(["mule", "collector"])
    pred = scored.mule_score >= threshold
    tp = int((truth & pred).sum())
    return dict(
        mules_total=int(truth.sum()), flagged=int(pred.sum()), true_found=tp,
        precision=tp / max(int(pred.sum()), 1), recall=tp / max(int(truth.sum()), 1),
    )
