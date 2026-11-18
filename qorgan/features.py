"""Признаки для скоринга перевода.

Все признаки считаются «причинно» — только по данным, доступным
на момент перевода (без заглядывания в будущее).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

FEATURES = [
    "log_amount",
    "amount_to_avg",
    "amount_to_balance",
    "is_new_recipient",
    "recipient_pasted",
    "is_new_device",
    "call_active",
    "call_from_unknown",
    "call_duration_min",
    "remote_access_app",
    "loan_issued_24h",
    "session_duration_sec",
    "is_night",
    "client_tx_1h",
    "client_amount_24h_to_avg",
    "recipient_account_age_days",
    "recipient_unique_senders_7d",
]

FEATURE_NAMES_RU = {
    "log_amount": "Сумма перевода",
    "amount_to_avg": "Сумма относительно обычной для клиента",
    "amount_to_balance": "Доля от остатка на счёте",
    "is_new_recipient": "Новый получатель",
    "recipient_pasted": "Реквизиты вставлены из буфера",
    "is_new_device": "Новое устройство",
    "call_active": "Идёт телефонный разговор",
    "call_from_unknown": "Звонок с неизвестного номера",
    "call_duration_min": "Длительность звонка",
    "remote_access_app": "Запущено приложение удалённого доступа",
    "loan_issued_24h": "Кредит оформлен за последние 24 ч",
    "session_duration_sec": "Длительность сессии",
    "is_night": "Ночное время",
    "client_tx_1h": "Переводов за последний час",
    "client_amount_24h_to_avg": "Сумма переводов за 24 ч относительно нормы",
    "recipient_account_age_days": "Возраст счёта получателя",
    "recipient_unique_senders_7d": "Разных отправителей у получателя за 7 дней",
}


def _recipient_unique_senders(tx: pd.DataFrame, window=pd.Timedelta(days=7)) -> np.ndarray:
    """Сколько разных клиентов переводили этому получателю за 7 дней ДО текущего перевода."""
    hist: dict[str, deque] = defaultdict(deque)
    out = np.zeros(len(tx), dtype=np.int32)
    for i, (ts, src, dst) in enumerate(zip(tx.ts.to_numpy(), tx.client_id.to_numpy(), tx.recipient_id.to_numpy())):
        q = hist[dst]
        while q and ts - q[0][0] > window:
            q.popleft()
        out[i] = len({s for _, s in q if s != src})
        q.append((ts, src))
    return out


def build_features(tx: pd.DataFrame, clients: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = tx.sort_values("ts").reset_index(drop=True).copy()
    prof = clients.set_index("client_id")
    df["avg_amount"] = df.client_id.map(prof.avg_amount)
    df["balance"] = df.client_id.map(prof.balance)
    df["log_amount"] = np.log1p(df.amount)
    df["amount_to_avg"] = (df.amount / df.avg_amount).round(3)
    df["amount_to_balance"] = (df.amount / df.balance).clip(upper=5).round(3)
    df["is_night"] = df.ts.dt.hour.between(0, 5)

    # скользящие окна по клиенту: сортируем по (клиент, время), тогда
    # результат groupby().rolling() идёт ровно в этом же порядке
    by_client = df.sort_values(["client_id", "ts"], kind="stable")
    r = by_client.set_index("ts").groupby("client_id", sort=True)["amount"]
    by_client = by_client.assign(
        cnt=r.rolling("1h").count().to_numpy(),
        s24=r.rolling("24h").sum().to_numpy(),
    )
    df = df.merge(by_client[["tx_id", "cnt", "s24"]], on="tx_id", how="left")
    df["client_tx_1h"] = df.cnt - 1  # без текущего перевода
    df["client_amount_24h_to_avg"] = (df.s24 / df.avg_amount).round(3)
    df = df.drop(columns=["cnt", "s24"])

    acc_age = accounts.set_index("account_id").account_age_days
    df["recipient_account_age_days"] = df.recipient_id.map(acc_age).fillna(365)
    df["recipient_unique_senders_7d"] = _recipient_unique_senders(df)

    for c in FEATURES:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
    return df
