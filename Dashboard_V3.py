# Dashboard_V2.py  (je kunt het bestand ook app.py noemen)
# Streamlit-dashboard voor Kattenburg – Leefbaarheidsscore Klimaatadaptatie
# Vereisten (requirements.txt):
#   streamlit>=1.32
#   pandas>=2.0
#   numpy>=1.24
#   plotly>=5.18

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------- Basisinstellingen ----------------------------
st.set_page_config(page_title="Kattenburg – Leefbaarheid Klimaatadaptatie", layout="wide")

CRITERIA = [
    "hitte", "wateroverlast", "droogte", "biodiversiteit",
    "sociale_cohesie", "toegankelijkheid", "beleving"
]
CONSTRAINTS = ["ruimteclaim", "kosten", "beheerlast"]

WEIGHT_DEFAULTS = {
    "hitte": 0.22, "wateroverlast": 0.20, "droogte": 0.18,
    "biodiversiteit": 0.12, "sociale_cohesie": 0.12,
    "toegankelijkheid": 0.08, "beleving": 0.08
}
PENALTY_DEFAULTS = {"ruimteclaim": 0.40, "kosten": 0.40, "beheerlast": 0.20}

BASE_MEASURES = [
    {"id": "groene_daken", "naam": "Groene daken",
     "hitte": 4.0, "wateroverlast": 4.5, "droogte": 3.5, "biodiversiteit": 4.0,
     "sociale_cohesie": 2.5, "toegankelijkheid": 3.0, "beleving": 3.5,
     "ruimteclaim": 1.0, "kosten": 3.5, "beheerlast": 2.5},
    {"id": "gevelgroen", "naam": "Gevelgroen",
     "hitte": 3.0, "wateroverlast": 1.5, "droogte": 1.5, "biodiversiteit": 3.0,
     "sociale_cohesie": 2.0, "toegankelijkheid": 3.5, "beleving": 4.0,
     "ruimteclaim": 0.5, "kosten": 2.0, "beheerlast": 2.0},
    {"id": "wadi", "naam": "Wadi’s & infiltratie",
     "hitte": 3.5, "wateroverlast": 4.5, "droogte": 4.0, "biodiversiteit": 4.0,
     "sociale_cohesie": 3.0, "toegankelijkheid": 2.5, "beleving": 3.5,
     "ruimteclaim": 3.5, "kosten": 3.0, "beheerlast": 2.5},
    {"id": "doorlatend", "naam": "Waterdoorlatende bestrating",
     "hitte": 2.5, "wateroverlast": 3.5, "droogte": 3.0, "biodiversiteit": 1.5,
     "sociale_cohesie": 1.0, "toegankelijkheid": 3.5, "beleving": 2.5,
     "ruimteclaim": 1.0, "kosten": 2.0, "beheerlast": 2.0},
    {"id": "regenwater", "naam": "Regenwateropvang & hergebruik",
     "hitte": 1.5, "wateroverlast": 3.5, "droogte": 4.0, "biodiversiteit": 1.0,
     "sociale_cohesie": 2.0, "toegankelijkheid": 4.0, "beleving": 2.0,
     "ruimteclaim": 1.0, "kosten": 2.5, "beheerlast": 2.0},
    {"id": "bomen", "naam": "Schaduwrijke beplanting & bomen",
     "hitte": 5.0, "wateroverlast": 2.5, "droogte": 2.0, "biodiversiteit": 4.0,
     "sociale_cohesie": 3.5, "toegankelijkheid": 2.5, "beleving": 4.5,
     "ruimteclaim": 3.0, "kosten": 3.0, "beheerlast": 3.0},
    {"id": "pocketparks", "naam": "Groene pleinen & pocketparks",
     "hitte": 4.0, "wateroverlast": 3.0, "droogte": 2.5, "biodiversiteit": 3.5,
     "sociale_cohesie": 4.5, "toegankelijkheid": 3.0, "beleving": 4.5,
     "ruimteclaim": 4.0, "kosten": 3.5, "beheerlast": 3.0},
    {"id": "verhoogde_trottoirs", "naam": "Verhoogde trottoirs & tijdelijke waterberging",
     "hitte": 1.5, "wateroverlast": 4.0, "droogte": 1.0, "biodiversiteit": 1.0,
     "sociale_cohesie": 1.0, "toegankelijkheid": 2.0, "beleving": 1.5,
     "ruimteclaim": 1.5, "kosten": 2.5, "beheerlast": 2.0},
    {"id": "onttegel", "naam": "Verminderen verhard oppervlak (onttegelen)",
     "hitte": 3.5, "wateroverlast": 3.5, "droogte": 3.0, "biodiversiteit": 3.5,
     "sociale_cohesie": 2.5, "toegankelijkheid": 3.0, "beleving": 3.0,
     "ruimteclaim": 2.0, "kosten": 1.0, "beheerlast": 2.0},
    {"id": "adaptief_bouwen", "naam": "Adaptief bouwen & ontwerpen",
     "hitte": 3.5, "wateroverlast": 3.5, "droogte": 3.0, "biodiversiteit": 2.5,
     "sociale_cohesie": 2.0, "toegankelijkheid": 3.5, "beleving": 3.0,
     "ruimteclaim": 2.0, "kosten": 4.0, "beheerlast": 2.5},
]

# ---------------------------- Hulpfuncties ----------------------------
def normalize_weights(w: dict) -> dict:
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}

def compute_scores(df: pd.DataFrame, active_ids, w_norm: dict, penalties: dict, lam: float = 0.5):
    df = df.copy()
    df["active"] = df["id"].isin(active_ids)
    A = df[df["active"]].copy()
    if A.empty:
        return A, 0.0

    # cast & clip
    for c in CRITERIA + CONSTRAINTS:
        A[c] = pd.to_numeric(A[c], errors="coerce").fillna(0.0).clip(0, 5)

    # componenten (benefit)
    for c in CRITERIA:
        A[f"w_{c}"] = w_norm.get(c, 0.0) * (A[c] / 5.0)
    A["benefit"] = A[[f"w_{c}" for c in CRITERIA]].sum(axis=1)

    # penalty (genormaliseerd)
    pen_sum = sum(penalties.values()) or 1.0
    A["penalty"] = sum(penalties[p] * (A[p] / 5.0) for p in CONSTRAINTS) / pen_sum

    # totaalscore
    A["score"] = (A["benefit"] - lam * A["penalty"]).clip(0, 1)

    A = A.sort_values("score", ascending=False)
    avg_score = float(A["score"].mean()) if not A.empty else 0.0
    return A, avg_score

def pareto_front(df: pd.DataFrame, x="penalty", y="benefit"):
    if df.empty:
        return df
    pts = df[[x, y]].to_numpy()
    keep = []
    for i in range(len(pts)):
        xi, yi = pts[i]
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            xj, yj = pts[j]
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep]

# ---------------------------- Data inladen ----------------------------
st.title("Kattenburg – Leefbaarheidsscore Klimaatadaptatie (Streamlit)")
st.caption("Zet maatregelen aan/uit, pas wegingen en strafpunten aan, en bekijk live de effecten en onderbouwing.")

uploaded = st.sidebar.file_uploader("Laad scoringsmatrix (CSV)", type=["csv"])
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        # minimale kolomcheck
        needed = set(["id", "naam"] + CRITERIA + CONSTRAINTS)
        if not needed.issubset(df_raw.columns):
            st.warning("CSV mist verplichte kolommen. Vallen terug op voorbeelddata.")
            df_raw = pd.DataFrame(BASE_MEASURES)
    except Exception:
        st.warning("Kon CSV niet lezen. Vallen terug op voorbeelddata.")
        df_raw = pd.DataFrame(BASE_MEASURES)
else:
    df_raw = pd.DataFrame(BASE_MEASURES)

# selectie maatregelen
st.sidebar.subheader("Maatregelen")
all_ids = df_raw["id"].tolist()
active_ids = st.sidebar.multiselect(
    "Actieve maatregelen", options=all_ids, default=all_ids,
    format_func=lambda x: df_raw.loc[df_raw["id"] == x, "naam"].values[0]
)

# ---------------- SIDEBAR: wegingen, penalties, presets (met callback) ----------------
if "weights" not in st.session_state:
    st.session_state.weights = WEIGHT_DEFAULTS.copy()
if "penalties" not in st.session_state:
    st.session_state.penalties = PENALTY_DEFAULTS.copy()

st.sidebar.subheader("Wegingen (effecten)")
for k in CRITERIA:
    st.session_state.weights[k] = st.sidebar.slider(
        k.capitalize(), 0.0, 1.0, float(st.session_state.weights[k]), 0.01, key=f"w_{k}"
    )

st.sidebar.subheader("Strafpunten (constraints)")
for k in CONSTRAINTS:
    st.session_state.penalties[k] = st.sidebar.slider(
        k.capitalize(), 0.0, 1.0, float(st.session_state.penalties[k]), 0.01, key=f"p_{k}"
    )
lambda_pen = st.sidebar.slider("Penalty-factor λ", 0.0, 1.0, 0.5, 0.01, key="lambda")

def apply_preset(new_w: dict):
    st.session_state.weights.update(new_w)
    for kk, vv in new_w.items():
        st.session_state[f"w_{kk}"] = float(vv)

st.sidebar.subheader("Scenario presets")
preset = st.sidebar.selectbox("Kies scenario", ["Gebalanceerd", "Hitte-focus", "Water-focus", "Droogte-focus"])
if preset == "Hitte-focus":
    PRESET_W = {"hitte":0.50,"wateroverlast":0.15,"droogte":0.10,"biodiversiteit":0.10,"sociale_cohesie":0.05,"toegankelijkheid":0.05,"beleving":0.05}
elif preset == "Water-focus":
    PRESET_W = {"hitte":0.15,"wateroverlast":0.50,"droogte":0.15,"biodiversiteit":0.08,"sociale_cohesie":0.05,"toegankelijkheid":0.04,"beleving":0.03}
elif preset == "Droogte-focus":
    PRESET_W = {"hitte":0.15,"wateroverlast":0.15,"droogte":0.50,"biodiversiteit":0.08,"sociale_cohesie":0.05,"toegankelijkheid":0.04,"beleving":0.03}
else:
    PRESET_W = WEIGHT_DEFAULTS.copy()
st.sidebar.button("Toepassen preset", on_click=apply_preset, args=(PRESET_W,))

weights_norm = normalize_weights(st.session_state.weights)

# ---------------------------- Berekening & KPI's ----------------------------
df_rank, avg_score = compute_scores(df_raw, active_ids, weights_norm, st.session_state.penalties, lam=lambda_pen)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Gemiddelde totaalscore (actief)", f"{round(avg_score*100):.0f}/100")
kpi2.metric("Aantal actieve maatregelen", len(df_rank))
kpi3.metric("Beste maatregel", df_rank.iloc[0]["naam"] if not df_rank.empty else "—")

st.markdown("---")

# ---------------------------- Tabs ----------------------------
tabs = st.tabs(["Ranking & tabel", "Uitleg per maatregel", "Effecten-overzicht", "Pareto & selectie", "Scenario-vergelijking"])

# 1) Ranking + Top-5 bar
with tabs[0]:
    col1, col2 = st.columns([2, 1])
    with col1:
        if df_rank.empty:
            st.info("Geen maatregelen actief.")
        else:
            show_cols = ["naam", "benefit", "penalty", "score"] + CRITERIA + CONSTRAINTS
            st.dataframe(df_rank[show_cols].reset_index(drop=True), use_container_width=True, height=420)
    with col2:
        if not df_rank.empty:
            top5 = df_rank.head(5)[["naam", "score"]].copy()
            top5["score_pct"] = (top5["score"] * 100).round(0)
            fig_bar = px.bar(top5, x="naam", y="score_pct", text="score_pct", range_y=[0, 100])
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(yaxis_title="Score (0–100)", xaxis_title="", margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)

# 2) Uitleg per maatregel (waterfall + radar)
with tabs[1]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        sel_name = st.selectbox("Kies maatregel", df_rank["naam"])
        row = df_rank[df_rank["naam"] == sel_name].iloc[0]

        contrib = [row[f"w_{c}"] for c in CRITERIA]
        pen_sum = sum(st.session_state.penalties.values()) or 1.0
        pens = [(st.session_state.penalties[c] * (row[c] / 5.0)) / pen_sum for c in CONSTRAINTS]

        wf = go.Figure(go.Waterfall(
            name="score",
            orientation="v",
            measure=["relative"] * len(CRITERIA) + ["relative"] * len(CONSTRAINTS) + ["total"],
            x=[c.capitalize() for c in CRITERIA] + [f"- {c}" for c in CONSTRAINTS] + ["Totaal"],
            textposition="outside",
            y=contrib + [-lambda_pen * p for p in pens] + [0.0],
        ))
        wf.update_layout(title=f"Scoresamenstelling – {row['naam']}", showlegend=False, margin=dict(t=30, b=20, l=10, r=10))
        st.plotly_chart(wf, use_container_width=True)

        radar_df = pd.DataFrame({"criterium": [c.capitalize() for c in CRITERIA], "waarde": [row[c] for c in CRITERIA]})
        fig_radar = px.line_polar(radar_df, r="waarde", theta="criterium", line_close=True)
        fig_radar.update_traces(fill="toself")
        fig_radar.update_polars(radialaxis=dict(range=[0, 5]))
        st.plotly_chart(fig_radar, use_container_width=True)

# 3) Effecten-overzicht (stacked + heatmap)
with tabs[2]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        topN = st.slider("Toon top-N", 3, min(10, len(df_rank)), 5, 1)
        stack_df = df_rank.head(topN).copy()
        for c in CRITERIA:
            stack_df[c.capitalize()] = stack_df[f"w_{c}"]
        melted = stack_df.melt(id_vars=["naam"], value_vars=[c.capitalize() for c in CRITERIA],
                               var_name="criterium", value_name="bijdrage")
        fig_stack = px.bar(melted, x="naam", y="bijdrage", color="criterium", title="Bijdrage per criterium aan benefit (gewogen)")
        st.plotly_chart(fig_stack, use_container_width=True)

        hm = df_rank.set_index("naam")[CRITERIA].rename(columns={c: c.capitalize() for c in CRITERIA})
        fig_hm = px.imshow(hm, aspect="auto", text_auto=True, title="Heatmap – criteria (0–5)")
        st.plotly_chart(fig_hm, use_container_width=True)

# 4) Pareto & selectie
with tabs[3]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        size_by = st.selectbox("Grootte bubbel", ["kosten", "ruimteclaim", "beheerlast"])
        fig_sc = px.scatter(df_rank, x="penalty", y="benefit", size=size_by, hover_name="naam",
                            title="Benefit vs Penalty – efficiënte maatregelen")
        fig_sc.update_layout(xaxis_title="Penalty (lager is beter)", yaxis_title="Benefit (hoger is beter)")
        st.plotly_chart(fig_sc, use_container_width=True)

        front = pareto_front(df_rank)
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=df_rank["penalty"], y=df_rank["benefit"], mode="markers",
                                    name="Alle maatregelen", text=df_rank["naam"]))
        if not front.empty:
            fig_pf.add_trace(go.Scatter(x=front["penalty"], y=front["benefit"], mode="markers+lines",
                                        name="Pareto-front", text=front["naam"]))
        fig_pf.update_layout(xaxis_title="Penalty (lager is beter)", yaxis_title="Benefit (hoger is beter)",
                             title="Pareto-front – efficiënte keuzes")
        st.plotly_chart(fig_pf, use_container_width=True)

# 5) Scenario-vergelijking
with tabs[4]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        colA, colB, colC = st.columns(3)
        presets = {
            "Gebalanceerd": WEIGHT_DEFAULTS,
            "Hitte-focus": {"hitte":0.5, "wateroverlast":0.15, "droogte":0.1, "biodiversiteit":0.1, "sociale_cohesie":0.05, "toegankelijkheid":0.05, "beleving":0.05},
            "Water-focus": {"hitte":0.15,"wateroverlast":0.5,"droogte":0.15,"biodiversiteit":0.08,"sociale_cohesie":0.05,"toegankelijkheid":0.04,"beleving":0.03},
        }
        for col, (name, w) in zip([colA, colB, colC], presets.items()):
            with col:
                W = normalize_weights(w)
                d, avg = compute_scores(df_raw, active_ids, W, st.session_state.penalties, lam=lambda_pen)
                st.markdown(f"**{name}** — Gemiddelde: **{round(avg*100):.0f}/100**")
                st.dataframe(d[["naam", "score"]].head(5).reset_index(drop=True), use_container_width=True, height=240)

# ---------------------------- Exports & uitleg ----------------------------
st.markdown("---")
colx, coly = st.columns(2)
with colx:
    csv_current = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button("Download huidige scoringsmatrix (CSV)", csv_current, file_name="kattenburg_scoring.csv", mime="text/csv")
with coly:
    if not df_rank.empty:
        csv_rank = df_rank.to_csv(index=False).encode("utf-8")
        st.download_button("Download ranking/analyses (CSV)", csv_rank, file_name="kattenburg_ranking.csv", mime="text/csv")

