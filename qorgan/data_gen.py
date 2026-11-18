"""Генератор синтетических банковских данных с мошенническими сценариями.

Реальных данных банка у нас нет, поэтому мы моделируем поведение клиентов
и схемы мошенничества, которые описаны в публичных отчётах регуляторов и банков:

* «Служба безопасности банка» / «безопасный счёт»
* «Кредит под диктовку» (мошенник уговаривает оформить кредит и перевести деньги)
* Удалённый доступ (жертва ставит AnyDesk/RustDesk по просьбе «сотрудника»)
* Фейковые инвестиции
* Предоплата на маркетплейсе / доске объявлений

Деньги жертв уходят на счета дропов (часто это студенты, которые «продали» карту),
а дропы пересылают их сборщикам, которые обналичивают средства.

Все параметры распределений вынесены в константы, чтобы их было легко
пересмотреть, когда появятся реальные данные банка.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CITIES = [
    "Алматы", "Астана", "Шымкент", "Тараз", "Караганда", "Актобе",
    "Павлодар", "Усть-Каменогорск", "Семей", "Атырау", "Костанай", "Кызылорда",
]
CITY_P = np.array([20, 16, 11, 9, 8, 6, 5, 5, 4, 5, 5, 6], dtype=float)
CITY_P /= CITY_P.sum()

SCHEMES = {
    "security_call": "Звонок «службы безопасности» / «безопасный счёт»",
    "loan_dictation": "Кредит под диктовку",
    "remote_access": "Удалённый доступ (AnyDesk/RustDesk)",
    "fake_invest": "Фейковые инвестиции",
    "marketplace": "Предоплата на маркетплейсе",
}
# доля каждой схемы среди мошеннических эпизодов
SCHEME_P = {
    "security_call": 0.34,
    "loan_dictation": 0.18,
    "remote_access": 0.14,
    "fake_invest": 0.16,
    "marketplace": 0.18,
}


@dataclass
class GenConfig:
    n_clients: int = 4000
    n_days: int = 45
    start: str = "2026-09-01"
    n_mule_rings: int = 8
    mules_per_ring: tuple[int, int] = (4, 8)
    victim_share: float = 0.07  # доля клиентов, ставших жертвами за период
    seed: int = 42


def _clients(cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_clients
    age = np.clip(rng.normal(41, 14, n), 18, 85).round().astype(int)
    df = pd.DataFrame(
        {
            "client_id": [f"C{i:05d}" for i in range(n)],
            "age": age,
            "city": rng.choice(CITIES, n, p=CITY_P),
            "account_age_days": rng.integers(60, 4000, n),
            # медианный перевод клиента, ₸
            "avg_amount": np.round(rng.lognormal(np.log(15000), 0.7, n), -2),
            "balance": np.round(rng.lognormal(np.log(350000), 1.0, n), -2),
            "tx_rate": rng.gamma(2.0, 0.3, n),  # переводов в день
        }
    )
    return df


def _mule_network(cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Кольца дропов: дропы 1-го уровня -> сборщики -> обнал."""
    rows = []
    for r in range(cfg.n_mule_rings):
        n_m = rng.integers(*cfg.mules_per_ring, endpoint=True)
        n_c = 1 if n_m < 7 else 2
        for c in range(n_c):
            rows.append(
                dict(account_id=f"X{r:02d}{c}", ring=r, role="collector",
                     owner_age=int(rng.integers(22, 40)),
                     account_age_days=int(rng.integers(5, 900)),
                     active_from=0, active_days=cfg.n_days)
            )
        for m in range(n_m):
            rows.append(
                dict(account_id=f"M{r:02d}{m:02d}", ring=r, role="mule",
                     # дропы — чаще всего молодые люди, часто студенты
                     owner_age=int(np.clip(rng.normal(20.5, 2.5), 17, 35)),
                     # часть дропов — новые счета, часть — «проданные» старые
                     account_age_days=int(rng.integers(3, 120) if rng.random() < 0.55
                                          else rng.integers(120, 2500)),
                     active_from=int(rng.integers(0, cfg.n_days - 5)),
                     active_days=int(rng.integers(5, 15)))
            )
    return pd.DataFrame(rows)


def _hours(rng, n, night=False):
    if night:
        p = np.ones(24)
    else:
        p = np.array([1, 0.5, 0.3, 0.2, 0.2, 0.4, 1, 3, 5, 6, 7, 7,
                      8, 8, 7, 7, 7, 8, 9, 9, 8, 6, 4, 2], dtype=float)
    p /= p.sum()
    return rng.choice(24, n, p=p)


def generate(cfg: GenConfig | None = None):
    """Возвращает (clients, accounts, transactions, ledger).

    transactions — переводы, инициированные клиентами банка (то, что скорит антифрод);
    ledger — все переводы, включая движение денег между дропами (для графа).
    """
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    clients = _clients(cfg, rng)
    mules = _mule_network(cfg, rng)
    start = pd.Timestamp(cfg.start)

    n_ext = 6000
    ext_ids = np.array([f"E{i:05d}" for i in range(n_ext)])
    # популярные легальные получатели (ИП, магазины, сборы) — много разных отправителей
    popular_ids = ext_ids[:150]
    client_ids = clients["client_id"].to_numpy()

    # «Записная книжка» получателей: друзья/родные/коммуналка
    contacts = {}
    for cid in client_ids:
        k = rng.integers(3, 12)
        pool = np.concatenate([rng.choice(client_ids, k // 2), rng.choice(ext_ids, k - k // 2)])
        contacts[cid] = list(pool[pool != cid])

    # ------------------------------------------------------------------ норма
    recs = []
    for c in clients.itertuples(index=False):
        n_tx = rng.poisson(c.tx_rate * cfg.n_days)
        if n_tx == 0:
            continue
        days = rng.integers(0, cfg.n_days, n_tx)
        hours = _hours(rng, n_tx)
        mins = rng.integers(0, 60, n_tx)
        new_rec = rng.random(n_tx) < 0.14
        known = rng.choice(contacts[c.client_id], n_tx)
        u = rng.random(n_tx)
        new = np.where(u < 0.25, rng.choice(popular_ids, n_tx),
                       np.where(u < 0.6, rng.choice(ext_ids, n_tx), rng.choice(client_ids, n_tx)))
        amount = np.round(c.avg_amount * rng.lognormal(0, 0.8, n_tx), -2).clip(200)
        # редкие крупные легитимные переводы (аренда, покупка авто, помощь родным)
        big = rng.random(n_tx) < 0.02
        amount = np.where(big, amount * rng.uniform(5, 20, n_tx), amount)
        call = rng.random(n_tx) < 0.05  # «скинь мне, я на телефоне» — нормальная ситуация
        call_unknown = call & (rng.random(n_tx) < 0.15)
        for i in range(n_tx):
            recs.append(dict(
                ts=start + pd.Timedelta(days=int(days[i]), hours=int(hours[i]), minutes=int(mins[i])),
                client_id=c.client_id,
                recipient_id=new[i] if new_rec[i] else known[i],
                amount=float(amount[i]),
                is_new_recipient=bool(new_rec[i]),
                is_new_device=bool(rng.random() < 0.02),
                call_active=bool(call[i]),
                call_from_unknown=bool(call_unknown[i]),
                call_duration_min=float(rng.exponential(3)) if call[i] else 0.0,
                remote_access_app=bool(rng.random() < 0.002),
                recipient_pasted=bool(new_rec[i] and rng.random() < 0.35),
                session_duration_sec=float(rng.lognormal(np.log(70), 0.5)),
                loan_issued_24h=bool(rng.random() < 0.008),
                is_fraud=False,
                scheme="",
            ))

    # ---------------------------------------------------------- мошенничество
    first_mules = mules[mules.role == "mule"]
    n_victims = int(cfg.victim_share * cfg.n_clients)
    # пожилые чаще становятся жертвами «службы безопасности»
    w = np.where(clients.age >= 55, 2.5, 1.0)
    victims = clients.sample(n_victims, weights=w, random_state=cfg.seed)
    schemes = list(SCHEME_P)
    for v in victims.itertuples(index=False):
        scheme = rng.choice(schemes, p=list(SCHEME_P.values()))
        if scheme == "security_call" and v.age < 30 and rng.random() < 0.5:
            scheme = "fake_invest"
        day0 = int(rng.integers(0, cfg.n_days))
        t0 = start + pd.Timedelta(days=day0, hours=int(_hours(rng, 1)[0]), minutes=int(rng.integers(0, 60)))
        # мошенники используют дропов, которые активны в этот момент
        active = first_mules[(first_mules.active_from <= day0)
                             & (day0 < first_mules.active_from + first_mules.active_days)]
        if active.empty:
            active = first_mules
        ring = rng.choice(active.ring.unique())
        ring_mules = active[active.ring == ring].account_id.to_numpy()
        balance = v.balance
        n_tx = {"security_call": rng.integers(1, 4), "loan_dictation": rng.integers(1, 3),
                "remote_access": rng.integers(1, 4), "fake_invest": rng.integers(2, 6),
                "marketplace": 1}[scheme]
        loan = 0.0
        if scheme == "loan_dictation":
            loan = float(np.round(rng.uniform(300_000, 3_000_000), -3))
            balance += loan
        call_len = rng.uniform(15, 90)
        for k in range(n_tx):
            if scheme == "fake_invest":
                ts = t0 + pd.Timedelta(days=int(k * rng.integers(1, 5)))
                amt = v.avg_amount * rng.uniform(3, 15) * (1.4 ** k)
            elif scheme == "marketplace":
                ts = t0
                amt = rng.uniform(8_000, 120_000)
            else:
                ts = t0 + pd.Timedelta(minutes=int(k * rng.integers(3, 25)))
                amt = balance * rng.uniform(0.4, 0.95)
            amt = float(np.round(min(amt, max(balance, 5000)), -2))
            balance = max(balance - amt, 0)
            is_call_scheme = scheme in ("security_call", "loan_dictation", "remote_access")
            recs.append(dict(
                ts=ts, client_id=v.client_id,
                recipient_id=(rng.choice(ring_mules) if scheme != "marketplace" or rng.random() < 0.6
                              else rng.choice(ext_ids)),
                amount=amt,
                is_new_recipient=bool(k == 0 or rng.random() < 0.6),
                is_new_device=bool(rng.random() < (0.55 if scheme == "remote_access" else 0.06)),
                call_active=bool(rng.random() < (0.88 if is_call_scheme else 0.2)),
                call_from_unknown=bool(rng.random() < (0.85 if is_call_scheme else 0.1)),
                call_duration_min=float(call_len + 5 * k) if is_call_scheme else float(rng.exponential(4)),
                remote_access_app=bool(rng.random() < (0.9 if scheme == "remote_access" else 0.03)),
                recipient_pasted=bool(rng.random() < 0.8),
                session_duration_sec=float(rng.lognormal(np.log(220 if is_call_scheme else 110), 0.5)),
                loan_issued_24h=bool(scheme == "loan_dictation" or rng.random() < 0.01),
                is_fraud=True, scheme=scheme,
            ))
    # для call_from_unknown без звонка — нет смысла
    tx = pd.DataFrame(recs)
    tx.loc[~tx.call_active, ["call_from_unknown"]] = False
    tx.loc[~tx.call_active, "call_duration_min"] = 0.0
    tx = tx[tx.ts < start + pd.Timedelta(days=cfg.n_days)]

    # дропы иногда получают «чистые» переводы для маскировки
    mule_ids = first_mules.account_id.to_numpy()
    cover_idx = tx[~tx.is_fraud].sample(frac=0.002, random_state=1).index
    tx.loc[cover_idx, "recipient_id"] = rng.choice(mule_ids, len(cover_idx))

    tx = tx.sort_values("ts").reset_index(drop=True)
    tx.insert(0, "tx_id", [f"T{i:07d}" for i in range(len(tx))])
    tx["amount"] = tx["amount"].round(0)

    # ------------------------------------------------- движение денег дропов
    ledger_rows = [tx[["tx_id", "ts", "client_id", "recipient_id", "amount"]]
                   .rename(columns={"client_id": "src", "recipient_id": "dst"})]
    inflow = tx[tx.recipient_id.isin(mule_ids)]
    collectors = mules[mules.role == "collector"].set_index("ring").account_id
    mule_ring = mules.set_index("account_id").ring
    fw = []
    for r in inflow.itertuples(index=False):
        ring = mule_ring[r.recipient_id]
        col = collectors.loc[[ring]].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        delay = pd.Timedelta(minutes=int(rng.integers(5, 240)))
        fee = rng.uniform(0.05, 0.15)  # «зарплата» дропа
        fw.append(dict(ts=r.ts + delay, src=r.recipient_id, dst=col, amount=round(r.amount * (1 - fee))))
        fw.append(dict(ts=r.ts + delay + pd.Timedelta(minutes=int(rng.integers(10, 600))),
                       src=col, dst=f"CASHOUT_{rng.choice(['ATM', 'CRYPTO', 'P2P'])}",
                       amount=round(r.amount * (1 - fee) * 0.97)))
    fwd = pd.DataFrame(fw)
    fwd.insert(0, "tx_id", [f"F{i:07d}" for i in range(len(fwd))])
    # зарплаты и другие поступления клиентам (2 раза в месяц от работодателей)
    sal = []
    employers = [f"EMP{i:03d}" for i in range(200)]
    for c in clients.itertuples(index=False):
        emp = employers[int(rng.integers(len(employers)))]
        pay = round(c.tx_rate * 15 * c.avg_amount * rng.uniform(1.1, 1.6), -3)
        for d in range(5, cfg.n_days, 15):
            sal.append(dict(ts=start + pd.Timedelta(days=d, hours=int(rng.integers(9, 18))),
                            src=emp, dst=c.client_id, amount=max(pay, 50_000)))
    sal = pd.DataFrame(sal)
    sal.insert(0, "tx_id", [f"S{i:07d}" for i in range(len(sal))])
    ledger = pd.concat(ledger_rows + [fwd, sal], ignore_index=True).sort_values("ts").reset_index(drop=True)

    # справочник счетов-получателей
    acc = pd.concat([
        pd.DataFrame({"account_id": client_ids, "kind": "client",
                      "account_age_days": clients.account_age_days, "owner_age": clients.age}),
        pd.DataFrame({"account_id": ext_ids, "kind": "external",
                      "account_age_days": rng.integers(30, 4000, n_ext),
                      "owner_age": np.clip(rng.normal(40, 14, n_ext), 18, 85).astype(int)}),
        mules.assign(kind=mules.role)[["account_id", "kind", "account_age_days", "owner_age", "ring"]],
    ], ignore_index=True)
    return clients, acc, tx, ledger
