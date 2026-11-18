import pandas as pd
import plotly.express as px
import streamlit as st

from common import ACCENT, DANGER, WARN, badge, fmt_kzt, load_metrics, load_model, load_tx, setup
from qorgan.data_gen import SCHEMES
from qorgan.model import ALLOW, DECISION_RU, HOLD, PAUSE
from qorgan.policy import mask

setup("Мониторинг", "🛡️")
model, m, tx = load_model(), load_metrics(), load_tx()
test = tx[tx.ts >= pd.Timestamp(m["split"])].copy()

st.title("🛡️ Кабинет аналитика антифрода")
st.caption(f"Тестовый период: {test.ts.min():%d.%m} – {test.ts.max():%d.%m.%Y}. "
           "Модель не видела эти переводы при обучении.")

alerts = test[test.decision != ALLOW]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Переводов за период", f"{len(test):,}".replace(",", " "))
c2.metric("Умных пауз", f"{(test.decision == PAUSE).sum():,}".replace(",", " "),
          help=f"{(test.decision == PAUSE).mean():.2%} всех переводов")
c3.metric("Удержаний", int((test.decision == HOLD).sum()))
c4.metric("Сохранено клиентам", fmt_kzt(m["model"]["saved_amount"]))

daily = (test.assign(day=test.ts.dt.date, Решение=test.decision.map(DECISION_RU))
         [test.decision != ALLOW].groupby(["day", "Решение"]).size().reset_index(name="n"))
fig = px.bar(daily, x="day", y="n", color="Решение",
             color_discrete_map={DECISION_RU[PAUSE]: WARN, DECISION_RU[HOLD]: DANGER},
             labels={"day": "", "n": "Срабатываний"})
fig.update_xaxes(tickformat="%d.%m")
fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.15))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Очередь срабатываний")
f1, f2, f3 = st.columns([1, 1, 2])
dec = f1.multiselect("Решение", [HOLD, PAUSE], default=[HOLD, PAUSE], format_func=DECISION_RU.get)
min_amt = f2.number_input("Сумма от, ₸", 0, 10_000_000, 0, step=10_000)
only_sig = f3.multiselect("Сигнал", ["Звонок с неизвестного", "Удалённый доступ", "Кредит за 24 ч", "Новое устройство"])

q = alerts[alerts.decision.isin(dec) & (alerts.amount >= min_amt)]
sig_map = {"Звонок с неизвестного": "call_from_unknown", "Удалённый доступ": "remote_access_app",
           "Кредит за 24 ч": "loan_issued_24h", "Новое устройство": "is_new_device"}
for s in only_sig:
    q = q[q[sig_map[s]] == 1]
q = q.sort_values("risk", ascending=False)

view = pd.DataFrame({
    "Время": q.ts.dt.strftime("%d.%m %H:%M"),
    "Клиент": q.client_id.map(mask),
    "Получатель": q.recipient_id.map(mask),
    "Сумма": q.amount.map(fmt_kzt),
    "Риск": q.risk,
    "Решение": q.decision.map(DECISION_RU),
    "Звонок": q.call_active.map({1: "📞", 0: ""}),
    "AnyDesk": q.remote_access_app.map({1: "🖥️", 0: ""}),
    "Кредит 24ч": q.loan_issued_24h.map({1: "💳", 0: ""}),
})
sel = st.dataframe(
    view, hide_index=True, use_container_width=True, height=330, on_select="rerun", selection_mode="single-row",
    column_config={"Риск": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)},
)
st.caption(f"Показано {len(q)} из {len(alerts)}. Выберите строку, чтобы увидеть объяснение.")

rows = sel.selection.rows if sel and sel.selection else []
if rows:
    r = q.iloc[rows[0]]
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown(f"### {fmt_kzt(r.amount)}  " + badge(r.decision), unsafe_allow_html=True)
        st.markdown(f"**Клиент** {mask(r.client_id)} → **получатель** {mask(r.recipient_id)}")
        st.markdown(f"Риск **{r.risk:.0%}** · {r.ts:%d.%m.%Y %H:%M}")
        with st.expander("Разметка (известна только в синтетических данных)"):
            st.write("Мошенничество:", "да — " + SCHEMES.get(r.scheme, "") if r.is_fraud else "нет")
    with b:
        st.markdown("**Из чего сложился риск** (вклад сигналов по Шепли)")
        for name, d in model.explain(r):
            st.progress(min(d, 1.0), text=f"{name} — {d:.0%}")
