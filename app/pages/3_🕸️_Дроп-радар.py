import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import DANGER, fmt_kzt, load_graph, load_metrics, load_tx, setup
from qorgan.graph import ring_subgraph
from qorgan.policy import mask

setup("Дроп-радар", "🕸️")
ledger, af, rings = load_graph()
m, tx = load_metrics(), load_tx()
THR = 0.5

st.title("🕸️ Дроп-радар")
st.markdown(
    "Дроп получает деньги **от многих незнакомых людей** и почти сразу **пересылает их дальше**. "
    "Радар ищет такие счета по графу всех переводов, а затем — сборщиков, к которым стекаются деньги."
)
r = m["radar"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Подозрительных счетов", r["flagged"])
c2.metric("Сетей найдено", r["rings_found"])
c3.metric("Точность", f"{r['precision']:.0%}")
c4.metric("Полнота", f"{r['recall']:.0%}")

ring_info = []
for i, ring in enumerate(rings):
    victims = ledger[ledger.dst.isin(ring) & ledger.src.str.startswith("C")]
    ring_info.append(dict(i=i, size=len(ring), victims=victims.src.nunique(), amount=victims.amount.sum()))
ri = pd.DataFrame(ring_info)

left, right = st.columns([1, 2.2], gap="large")
with left:
    st.subheader("Сети")
    choice = st.radio(
        "Выберите сеть", ri.i,
        format_func=lambda i: f"Сеть #{i + 1}: {ri.loc[i, 'size']} счетов · {ri.loc[i, 'victims']} жертв · "
                              f"{fmt_kzt(ri.loc[i, 'amount'])}",
        label_visibility="collapsed",
    )
    ring = rings[choice]
    members = af.loc[sorted(ring)].sort_values("mule_score", ascending=False)
    st.markdown("**Счета сети**")
    st.dataframe(pd.DataFrame({
        "Счёт": members.index.map(mask),
        "Скор": members.mule_score,
        "Отправителей": members.in_unique_senders.astype(int),
        "Транзит 24ч": members.fast_forward_share,
        "Возраст счёта, дн": members.account_age_days.astype(int),
    }), hide_index=True, use_container_width=True,
        column_config={"Скор": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                       "Транзит 24ч": st.column_config.NumberColumn(format="percent")})

with right:
    e = ring_subgraph(ledger, ring)
    g = nx.DiGraph()
    for s, d, amt in zip(e.src, e.dst, e.amount):
        g.add_edge(s, d, amount=amt)

    def role(n):
        if n.startswith("CASHOUT"):
            return "cashout"
        if n in ring:
            return "collector" if af.loc[n, "from_suspicious_share"] > 0.5 else "mule"
        return "victim"

    # раскладка по слоям: жертвы → дропы → сборщики → обнал
    layer = {"victim": 0, "mule": 1, "collector": 2, "cashout": 3}
    for n in g:
        g.nodes[n]["layer"] = layer[role(n)]
    # каждый слой равномерно растягиваем по высоте
    pos = {}
    for lv in range(4):
        nodes = sorted(n for n in g if g.nodes[n]["layer"] == lv)
        for k, n in enumerate(nodes):
            pos[n] = (lv, 0.5 if len(nodes) == 1 else k / (len(nodes) - 1))
    style = {"victim": ("#4C8BF5", 10, "Жертва"), "mule": (DANGER, 16, "Дроп"),
             "collector": ("#7A1F1F", 22, "Сборщик"), "cashout": ("#222", 22, "Обнал")}
    fig = go.Figure()
    for s, d in g.edges:
        fig.add_trace(go.Scatter(x=[pos[s][0], pos[d][0]], y=[pos[s][1], pos[d][1]], mode="lines",
                                 line=dict(width=0.7, color="rgba(120,120,120,.45)"), hoverinfo="skip",
                                 showlegend=False))
    for rl, (color, size, label) in style.items():
        nodes = [n for n in g if role(n) == rl]
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers", name=label,
            marker=dict(color=color, size=size, line=dict(width=1, color="white")),
            text=[f"{mask(n)}<br>вход: {fmt_kzt(e[e.dst == n].amount.sum())}" for n in nodes],
            hoverinfo="text"))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, use_container_width=True)

    victims = ledger[ledger.dst.isin(ring) & ledger.src.str.startswith("C")]
    st.markdown(
        f"**Что делать банку:** заблокировать {len(ring)} счетов сети, передать данные в АФМ / полицию и "
        f"**проактивно позвонить {victims.src.nunique()} клиентам**, которые переводили в эту сеть, — "
        "часть из них, возможно, ещё не поняли, что их обманули."
    )
