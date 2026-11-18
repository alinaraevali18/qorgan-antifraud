import numpy as np
import pandas as pd
import streamlit as st

from common import ACCENT, DANGER, WARN, badge, fmt_kzt, load_model, setup
from qorgan.model import ALLOW, HOLD, PAUSE
from qorgan.policy import SCHEME_WARNINGS, TEXTS, guess_scheme

setup("Клиент", "📱")
model = load_model()

CLIENT = dict(name="Айгуль Сейтжановна, 62 года, Тараз", avg_amount=20_000, balance=450_000)

PRESETS = {
    "👧 Перевод дочери": dict(amount=25_000, is_new_recipient=False, recipient_pasted=False, is_new_device=False,
                             call_active=False, call_from_unknown=False, call_duration_min=0, remote_access_app=False,
                             loan_issued_24h=False, recipient_account_age_days=2500, recipient_unique_senders_7d=1,
                             session_duration_sec=60, prev_24h=0, client_tx_1h=0, is_night=False),
    "📞 Друг звонит: «скинь 15 000»": dict(amount=15_000, is_new_recipient=False, recipient_pasted=False,
                                           is_new_device=False, call_active=True, call_from_unknown=False,
                                           call_duration_min=3, remote_access_app=False, loan_issued_24h=False,
                                           recipient_account_age_days=1800, recipient_unique_senders_7d=0,
                                           session_duration_sec=50, prev_24h=0, client_tx_1h=0, is_night=False),
    "🚨 «Служба безопасности банка»": dict(amount=380_000, is_new_recipient=True, recipient_pasted=True,
                                          is_new_device=False, call_active=True, call_from_unknown=True,
                                          call_duration_min=35, remote_access_app=False, loan_issued_24h=False,
                                          recipient_account_age_days=40, recipient_unique_senders_7d=6,
                                          session_duration_sec=260, prev_24h=0, client_tx_1h=0, is_night=False),
    "💳 Кредит под диктовку": dict(amount=1_200_000, is_new_recipient=True, recipient_pasted=True,
                                   is_new_device=False, call_active=True, call_from_unknown=True,
                                   call_duration_min=50, remote_access_app=False, loan_issued_24h=True,
                                   recipient_account_age_days=900, recipient_unique_senders_7d=4,
                                   session_duration_sec=300, prev_24h=0, client_tx_1h=0, is_night=False),
    "🖥️ Удалённый доступ": dict(amount=300_000, is_new_recipient=True, recipient_pasted=True, is_new_device=True,
                               call_active=True, call_from_unknown=True, call_duration_min=20, remote_access_app=True,
                               loan_issued_24h=False, recipient_account_age_days=60, recipient_unique_senders_7d=3,
                               session_duration_sec=200, prev_24h=0, client_tx_1h=0, is_night=False),
    "🛒 Предоплата продавцу": dict(amount=60_000, is_new_recipient=True, recipient_pasted=True,
                                   is_new_device=False, call_active=False, call_from_unknown=False,
                                   call_duration_min=0, remote_access_app=False, loan_issued_24h=False,
                                   recipient_account_age_days=1500, recipient_unique_senders_7d=1,
                                   session_duration_sec=120, prev_24h=0, client_tx_1h=0, is_night=False),
    "📈 «Инвестиции» без звонка": dict(amount=150_000, is_new_recipient=True, recipient_pasted=True,
                                      is_new_device=False, call_active=False, call_from_unknown=False,
                                      call_duration_min=0, remote_access_app=False, loan_issued_24h=False,
                                      recipient_account_age_days=1500, recipient_unique_senders_7d=5,
                                      session_duration_sec=140, prev_24h=90_000, client_tx_1h=1, is_night=False),
}

if "preset" not in st.session_state:
    st.session_state.preset = "🚨 «Служба безопасности банка»"


def apply_preset(name):
    st.session_state.preset = name
    for k, v in PRESETS[name].items():
        st.session_state[f"in_{k}"] = v


if "in_amount" not in st.session_state:
    apply_preset(st.session_state.preset)

st.title("📱 Глазами клиента")
st.caption(f"Клиент: {CLIENT['name']}. Обычный перевод — {fmt_kzt(CLIENT['avg_amount'])}, "
           f"на счёте — {fmt_kzt(CLIENT['balance'])}.")

cols = st.columns(4) + st.columns(4)
for col, name in zip(cols, PRESETS):
    col.button(name, on_click=apply_preset, args=(name,), use_container_width=True,
               type="primary" if st.session_state.preset == name else "secondary")

ctrl, phone, why = st.columns([1.1, 0.9, 1.1], gap="large")

with ctrl:
    st.subheader("Что происходит")
    amount = st.number_input("Сумма перевода, ₸", 100, 5_000_000, step=5_000, key="in_amount")
    st.markdown("**Получатель**")
    new_rec = st.toggle("Новый получатель", key="in_is_new_recipient")
    pasted = st.toggle("Реквизиты вставлены из буфера", key="in_recipient_pasted")
    rec_age = st.slider("Возраст счёта получателя, дней", 1, 4000, key="in_recipient_account_age_days")
    rec_senders = st.slider("Разных отправителей у получателя за 7 дней", 0, 15,
                            key="in_recipient_unique_senders_7d")
    st.markdown("**Контекст сессии (сигналы из приложения)**")
    call = st.toggle("Идёт телефонный разговор", key="in_call_active")
    unknown = st.toggle("Номер не из контактов", key="in_call_from_unknown", disabled=not call)
    call_min = st.slider("Разговор длится, мин", 0, 90, key="in_call_duration_min", disabled=not call)
    remote = st.toggle("Запущен AnyDesk / RustDesk", key="in_remote_access_app")
    new_dev = st.toggle("Новое устройство", key="in_is_new_device")
    loan = st.toggle("Кредит оформлен за последние 24 ч", key="in_loan_issued_24h")
    with st.expander("Ещё сигналы"):
        sess = st.slider("Длительность сессии, сек", 10, 600, key="in_session_duration_sec")
        prev = st.number_input("Уже переведено за 24 ч, ₸", 0, 5_000_000, step=10_000, key="in_prev_24h")
        tx1h = st.slider("Переводов за последний час", 0, 10, key="in_client_tx_1h")
        night = st.toggle("Ночь (00:00–06:00)", key="in_is_night")

row = dict(
    log_amount=np.log1p(amount), amount_to_avg=amount / CLIENT["avg_amount"],
    amount_to_balance=min(amount / CLIENT["balance"], 5),
    is_new_recipient=int(new_rec), recipient_pasted=int(pasted), is_new_device=int(new_dev),
    call_active=int(call), call_from_unknown=int(call and unknown), call_duration_min=call_min if call else 0,
    remote_access_app=int(remote), loan_issued_24h=int(loan), session_duration_sec=sess,
    is_night=int(night), client_tx_1h=tx1h,
    client_amount_24h_to_avg=(amount + prev) / CLIENT["avg_amount"],
    recipient_account_age_days=rec_age, recipient_unique_senders_7d=rec_senders,
)
risk = float(model.score(pd.DataFrame([row]))[0])
decision = model.decide(np.array([risk]))[0]
scheme = guess_scheme(row)

with phone:
    lang = st.segmented_control("Язык", ["ru", "kz"], default="ru",
                                format_func=lambda x: "Русский" if x == "ru" else "Қазақша") or "ru"
    T = TEXTS[lang]
    if decision == ALLOW:
        color, icon, title, body = ACCENT, "✅", T["ok_title"], T["ok_body"]
        extra = ""
    elif decision == PAUSE:
        color, icon, title = WARN, "✋", T["pause_title"]
        body = SCHEME_WARNINGS[scheme][lang]
        extra = "".join(f'<div class="q">☐ {q}</div>' for q in T["questions"])
        extra += (f'<div class="btn" style="background:{DANGER}">{T["cancel"]}</div>'
                  f'<div class="btn ghost">{T["continue"]}</div>'
                  f'<div class="btn ghost">👤 {T["trusted"]}</div>')
    else:
        color, icon, title = DANGER, "🛡️", T["hold_title"]
        body = SCHEME_WARNINGS[scheme][lang] + "<br><br>" + T["hold_body"]
        extra = f'<div class="btn ghost">👤 {T["trusted"]}</div>'
    st.markdown(
        f"""
        <style>
        .phone {{width:300px;margin:8px auto;border:10px solid #111;border-radius:36px;background:#fff;
                 color:#111;padding:18px 16px 22px;box-shadow:0 10px 30px rgba(0,0,0,.25);
                 font-family:Inter,system-ui,sans-serif;min-height:520px}}
        .phone .top {{font-size:.75rem;color:#888;text-align:center;margin-bottom:10px}}
        .phone .amt {{font-size:1.6rem;font-weight:700;text-align:center;margin:6px 0 14px}}
        .phone .card {{border-radius:16px;padding:14px;background:{color}18;border:2px solid {color}}}
        .phone .ttl {{font-weight:700;font-size:1.05rem;color:{color};margin-bottom:6px}}
        .phone .body {{font-size:.88rem;line-height:1.35}}
        .phone .q {{font-size:.85rem;margin:8px 0 0}}
        .phone .btn {{margin-top:10px;border-radius:12px;padding:9px;text-align:center;color:#fff;
                      font-weight:600;font-size:.85rem}}
        .phone .ghost {{background:#f1f3f5;color:#111}}
        </style>
        <div class="phone">
          <div class="top">{T["header"]}</div>
          <div class="amt">{fmt_kzt(amount)}</div>
          <div class="card"><div class="ttl">{icon} {title}</div><div class="body">{body}</div>{extra}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with why:
    st.subheader("Решение антифрода")
    st.markdown(badge(decision), unsafe_allow_html=True)
    st.progress(min(risk, 1.0), text=f"Риск: {risk:.0%}")
    st.caption(f"Порог «умной паузы» — {model.t_pause:.0%}, удержания — {model.t_hold:.0%}.")
    reasons = model.explain(row)
    if reasons and decision != ALLOW:
        st.markdown("**Из чего сложился риск**")
        for name, d in reasons:
            st.progress(min(d, 1.0), text=f"{name} — {d:.0%}")
    else:
        st.markdown("Поведение типично для клиента — перевод проходит без задержки.")
    st.divider()
    st.markdown(
        "**Почему пауза, а не блокировка?** Жертву торопят и держат на телефоне. "
        "Вопрос на экране и 10 минут тишины часто разрушают сценарий мошенника, "
        "а честный клиент теряет лишь несколько секунд."
    )
