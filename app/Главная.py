import plotly.graph_objects as go
import streamlit as st

from common import ACCENT, load_metrics, setup

setup("Обзор")
m = load_metrics()
SCHEMES = __import__("qorgan.data_gen", fromlist=["SCHEMES"]).SCHEMES

st.title("🛡️ Qorgan")
st.markdown(
    "#### Антифрод, который замечает, что клиента **уговаривают** перевести деньги, "
    "и находит **сети дропов**, через которые эти деньги уходят."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Мошеннических переводов поймано", f"{m['model']['recall']:.0%}",
          f"{(m['model']['recall'] - m['rules']['recall']) * 100:+.0f} п.п. к правилам")
c2.metric("Сумма, сохранённая клиентам", f"{m['model']['saved_share']:.1%}",
          f"{(m['model']['saved_share'] - m['rules']['saved_share']) * 100:+.0f} п.п. к правилам")
c3.metric("Честных переводов с паузой", f"{m['model']['false_alarm_rate']:.1%}",
          help="Без блокировки: клиент видит вопрос и может продолжить")
c4.metric("Сетей дропов найдено", m["radar"]["rings_found"],
          help=f"Точность {m['radar']['precision']:.0%}, полнота {m['radar']['recall']:.0%}")

st.divider()
left, right = st.columns([1.1, 1])
with left:
    st.subheader("Проблема")
    st.markdown(
        """
- Основной ущерб сегодня — не взлом банка, а **социальная инженерия**: клиент сам,
  своими руками, переводит деньги «сотруднику службы безопасности».
- Классические правила смотрят только на сумму и получателя. Для них такой перевод
  выглядит как обычный: правильный пароль, своё устройство, сам клиент.
- Украденное за минуты уходит через **дропов** — часто студентов, «продавших» карту
  за 20–50 тыс. ₸, — и обналичивается.
"""
    )
    st.subheader("Решение — три слоя")
    st.markdown(
        """
1. **Сигналы сессии.** Приложение банка видит контекст, а не только сумму: идёт ли звонок
   с неизвестного номера, запущен ли AnyDesk, вставлены ли реквизиты из буфера,
   оформлен ли только что кредит.
2. **Умная пауза вместо блокировки.** При среднем риске клиент видит понятное предупреждение
   на родном языке и 3 вопроса — это разрывает «гипноз» мошенника. При высоком риске —
   удержание и звонок из банка с официального номера.
3. **Дроп-радар.** Граф переводов находит счета, которые быстро пересылают деньги от
   многих незнакомых людей, и целые сети вместе со сборщиками.
"""
    )
with right:
    st.subheader("Какие схемы ловим")
    names = [SCHEMES[k] for k in m["by_scheme"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=[m["rules_by_scheme"][k] for k in m["by_scheme"]], name="Классические правила",
                orientation="h", marker_color="#9AA5B1")
    fig.add_bar(y=names, x=list(m["by_scheme"].values()), name="Qorgan", orientation="h",
                marker_color=ACCENT)
    fig.update_layout(barmode="group", height=380, xaxis_tickformat=".0%",
                      margin=dict(l=0, r=10, t=10, b=0), legend=dict(orientation="h", y=-0.15),
                      xaxis_title="Доля пойманных переводов")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Тестовый период: {m['n']:,} переводов после {m['split']}, из них {m['n_fraud']} мошеннических. "
        "Данные синтетические — см. страницу «Модель»."
        .replace(",", " ")
    )

st.info("👈 Начните с **📱 Клиент** — попробуйте сами пройти сценарий «звонок из службы безопасности».")
