import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ACCENT, load_metrics, setup
from qorgan.data_gen import SCHEMES
from qorgan.features import FEATURE_NAMES_RU, FEATURES

setup("Модель", "📊")
m = load_metrics()

st.title("📊 Модель и методология")

st.subheader("Qorgan против классических правил")
rows = []
for key, name in [("rules", "Правила: крупная сумма новому получателю / новое устройство"),
                  ("model", "Qorgan (умная пауза + удержание)")]:
    x = m[key]
    rows.append({"Подход": name, "Поймано переводов": x["recall"], "Сохранено суммы": x["saved_share"],
                 "Точность срабатываний": x["precision"], "Лишних срабатываний": x["false_alarm_rate"]})
_cfg = {c: st.column_config.NumberColumn(format="percent", width="small") for c in
        ["Поймано переводов", "Сохранено суммы", "Точность срабатываний", "Лишних срабатываний"]}
_cfg["Подход"] = st.column_config.TextColumn(width="medium")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config=_cfg)

c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
c2.metric("PR-AUC", f"{m['pr_auc']:.3f}", help="Главная метрика при редком мошенничестве (~0.5% переводов)")
c3.metric("Удержаний: точность", f"{m['hold']['precision']:.0%}",
          help="Доля мошеннических среди удержанных переводов")

a, b = st.columns(2)
with a:
    fig = go.Figure(go.Scatter(x=m["pr_curve"]["recall"], y=m["pr_curve"]["precision"], mode="lines",
                               line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[m["rules"]["recall"]], y=[m["rules"]["precision"]], mode="markers+text",
                             text=["Правила"], textposition="top center",
                             marker=dict(size=12, color="#9AA5B1")))
    fig.update_layout(title="Precision–Recall", xaxis_title="Полнота (recall)", yaxis_title="Точность",
                      height=360, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
with b:
    df = pd.DataFrame({"Схема": [SCHEMES[k] for k in m["by_scheme"]],
                       "Правила": list(m["rules_by_scheme"].values()),
                       "Qorgan": list(m["by_scheme"].values())})
    st.markdown("**Доля пойманных по схемам**")
    cfg = {c: st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1, width="small")
           for c in ["Правила", "Qorgan"]}
    cfg["Схема"] = st.column_config.TextColumn(width="medium")
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=cfg)
    st.caption("«Предоплата на маркетплейсе» ловится хуже всего: ни звонка, ни крупной суммы. "
               "Здесь нужен дроп-радар и обмен данными о счетах мошенников между банками.")

st.divider()
st.subheader("Как это устроено")
st.markdown(
    f"""
**Модель.** Градиентный бустинг (`HistGradientBoostingClassifier`, scikit-learn) на {len(FEATURES)} признаках.
Обучение — на первых 30 днях, проверка — на последних 15 (**разбиение по времени**, как в реальной
эксплуатации). Пороги подобраны так, чтобы «умная пауза» срабатывала не более чем на 1% честных
переводов, а удержание — не более чем на 0,15%.

**Объяснимость.** Для каждого решения считаем вклад сигналов по значениям Шепли (идея SHAP):
насколько каждый признак увеличил риск по сравнению с типичным честным переводом. Это объяснение
видит аналитик, а клиент — понятный текст предупреждения на русском или казахском.

**Этика.** Возраст, пол и город клиента **не используются** в модели — чтобы не дискриминировать.
Пожилые клиенты чаще становятся жертвами, но модель реагирует на ситуацию (звонок, AnyDesk, кредит), а не на человека.

**Дроп-радар.** Скор счёта = доля входящих переводов с высоким риском + доля денег, пересланных
дальше в течение 24 часов (при ≥3 разных отправителях) + число отправителей + молодость счёта.
Второй проход добавляет долю денег от уже подозрительных счетов — так находятся сборщики.
Сети — связные компоненты подозрительных счетов.
"""
)
with st.expander("Все признаки модели"):
    st.dataframe(pd.DataFrame({"Признак": FEATURES, "Описание": [FEATURE_NAMES_RU[f] for f in FEATURES]}),
                 hide_index=True, use_container_width=True)

st.warning(
    "**Ограничения.** Данные синтетические: мы смоделировали поведение клиентов и 5 схем мошенничества "
    "по публичным описаниям. Метрики показывают, что подход работает, но на реальных данных они будут "
    "ниже. Следующий шаг — пилот на обезличенных данных банка и A/B-тест текстов умной паузы."
)
