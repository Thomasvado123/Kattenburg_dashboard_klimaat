import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Kattenburg – Leefbaarheid Klimaatadaptatie", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
CRITERIA = ["hitte", "wateroverlast", "droogte", "biodiversiteit",
            "sociale_cohesie", "toegankelijkheid", "beleving"]
CONSTRAINTS = ["ruimteclaim", "kosten", "beheerlast"]

WEIGHT_DEFAULTS = {"hitte":0.22,"wateroverlast":0.20,"droogte":0.18,
                   "biodiversiteit":0.12,"sociale_cohesie":0.12,
                   "toegankelijkheid":0.08,"beleving":0.08}
PENALTY_DEFAULTS = {"ruimteclaim":0.40,"kosten":0.40,"beheerlast":0.20}

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

def normalize_weights(w: dict) -> dict:
    s = sum(w.values()) or 1.0
    return {k: v/s for k, v in w.items()}

def compute_scores(df: pd.DataFrame, active_ids, w_norm: dict, penalties: dict, lam: float = 0.5):
    df = df.copy()
    df["active"] = df["id"].isin(active_ids)
    A = df[df["active"]].copy()
    if A.empty:
        return A, 0.0

    # Ensure numeric and clipped
    for c in CRITERIA + CONSTRAINTS:
        A[c] = pd.to_numeric(A[c], errors="coerce").fillna(0.0).clip(0,5)

    # Benefit components (weighted)
    for c in CRITERIA:
        A[f"w_{c}"] = w_norm.get(c, 0) * (A[c]/5.0)

    A["benefit"] = A[[f"w_{c}" for c in CRITERIA]].sum(axis=1)

    # Penalty (normalized by weight sum of penalties)
    pen_sum = sum(penalties.values()) or 1.0
    A["penalty"] = sum(penalties[c]*(A[c]/5.0) for c in CONSTRAINTS)/pen_sum

    # Final score
    A["score"] = (A["benefit"] - lam*A["penalty"]).clip(0,1)

    A = A.sort_values("score", ascending=False)
    avg_score = float(A["score"].mean()) if not A.empty else 0.0
    return A, avg_score

def pareto_front(df: pd.DataFrame, x="penalty", y="benefit"):
    # Lower x (penalty) and higher y (benefit) is better
    pts = df[[x, y, "naam", "id"]].values
    idx = []
    for i in range(len(pts)):
        xi, yi, _, _ = pts[i]
        dominated = False
        for j in range(len(pts)):
            if i==j: continue
            xj, yj, _, _ = pts[j]
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            idx.append(i)
    return df.iloc[idx]

# ----------------------------
# UI – Sidebar
# ----------------------------
st.title("Kattenburg – Leefbaarheidsscore Klimaatadaptatie (Streamlit)")
st.caption("Zet maatregelen aan/uit, pas wegingen en strafpunten aan, en bekijk live de effecten en onderbouwing.")

uploaded = st.sidebar.file_uploader("Laad scoringsmatrix (CSV)", type=["csv"])
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.DataFrame(BASE_MEASURES)

st.sidebar.subheader("Maatregelen")
all_ids = df_raw["id"].tolist()
default_on = all_ids
active_ids = st.sidebar.multiselect(
    "Actieve maatregelen", options=all_ids, default=default_on,
    format_func=lambda x: df_raw.loc[df_raw["id"]==x, "naam"].values[0]
)

st.sidebar.subheader("Wegingen (effecten)")
if "weights" not in st.session_state:
    st.session_state.weights = WEIGHT_DEFAULTS.copy()
if "penalties" not in st.session_state:
    st.session_state.penalties = PENALTY_DEFAULTS.copy()

for k in CRITERIA:
    st.session_state.weights[k] = st.sidebar.slider(
        k.capitalize(), 0.0, 1.0, float(st.session_state.weights[k]), 0.01, key=f"w_{k}"
    )
weights_norm = normalize_weights(st.session_state.weights)

st.sidebar.subheader("Strafpunten (constraints)")
for k in CONSTRAINTS:
    st.session_state.penalties[k] = st.sidebar.slider(
        k.capitalize(), 0.0, 1.0, float(st.session_state.penalties[k]), 0.01, key=f"p_{k}"
    )

lambda_pen = st.sidebar.slider("Penalty-factor λ", 0.0, 1.0, 0.5, 0.01, key="lambda")

# Presets

# ---------------- SIDEBAR: wegingen, penalties, presets (met callback) ----------------

# 1) Init session state
if "weights" not in st.session_state:
    st.session_state.weights = {
        "hitte": 0.22, "wateroverlast": 0.20, "droogte": 0.18,
        "biodiversiteit": 0.12, "sociale_cohesie": 0.12,
        "toegankelijkheid": 0.08, "beleving": 0.08
    }
if "penalties" not in st.session_state:
    st.session_state.penalties = {"ruimteclaim": 0.40, "kosten": 0.40, "beheerlast": 0.20}

# 2) Sliders – wegingen (met vaste keys zodat we ze kunnen updaten vanuit presets)
st.sidebar.subheader("Wegingen (effecten)")
for k in CRITERIA:
    st.session_state.weights[k] = st.sidebar.slider(
        k.capitalize(), 0.0, 1.0,
        float(st.session_state.weights[k]), 0.01,
        key=f"w_{k}"
    )

# 3) Sliders – strafpunten en λ
st.sidebar.subheader("Strafpunten (constraints)")
for k in CONSTRAINTS:
    st.session_state.penalties[k] = st.sidebar.slider(
        k.capitalize(), 0.0, 1.0,
        float(st.session_state.penalties[k]), 0.01,
        key=f"p_{k}"
    )
lambda_pen = st.sidebar.slider("Penalty-factor λ", 0.0, 1.0, 0.5, 0.01, key="lambda")

# 4) Preset callback + UI
def apply_preset(new_w: dict):
    """Update zowel de onderliggende wegingen als de zichtbare sliderwaarden."""
    st.session_state.weights.update(new_w)
    for kk, vv in new_w.items():
        st.session_state[f"w_{kk}"] = float(vv)  # update slider key

st.sidebar.subheader("Scenario presets")
preset = st.sidebar.selectbox(
    "Kies scenario",
    ["Gebalanceerd", "Hitte-focus", "Water-focus", "Droogte-focus"]
)

if preset == "Hitte-focus":
    PRESET_W = {"hitte":0.50,"wateroverlast":0.15,"droogte":0.10,
                "biodiversiteit":0.10,"sociale_cohesie":0.05,
                "toegankelijkheid":0.05,"beleving":0.05}
elif preset == "Water-focus":
    PRESET_W = {"hitte":0.15,"wateroverlast":0.50,"droogte":0.15,
                "biodiversiteit":0.08,"sociale_cohesie":0.05,
                "toegankelijkheid":0.04,"beleving":0.03}
elif preset == "Droogte-focus":
    PRESET_W = {"hitte":0.15,"wateroverlast":0.15,"droogte":0.50,
                "biodiversiteit":0.08,"sociale_cohesie":0.05,
                "toegankelijkheid":0.04,"beleving":0.03}
else:
    PRESET_W = {"hitte":0.22,"wateroverlast":0.20,"droogte":0.18,
                "biodiversiteit":0.12,"sociale_cohesie":0.12,
                "toegankelijkheid":0.08,"beleving":0.08}

st.sidebar.button("Toepassen preset", on_click=apply_preset, args=(PRESET_W,))

# 5) Genormaliseerde wegingen (na eventuele preset/slider-wijziging)
weights_norm = normalize_weights(st.session_state.weights)


# ----------------------------
# Compute & KPIs
# ----------------------------
df_rank, avg_score = compute_scores(df_raw, active_ids, weights_norm, st.session_state.penalties, lam=lambda_pen)

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Gemiddelde totaalscore (actief)", f"{round(avg_score*100):.0f}/100")
kpi2.metric("Aantal actieve maatregelen", len(df_rank))
kpi3.metric("Beste maatregel", df_rank.iloc[0]["naam"] if not df_rank.empty else "—")

st.markdown("---")

# ----------------------------
# Tabs for analysis
# ----------------------------
tabs = st.tabs(["Ranking & tabel", "Uitleg per maatregel", "Effecten-overzicht", "Pareto & selectie", "Scenario-vergelijking"])

# 1) Ranking table & Top-5 bar
with tabs[0]:
    col1, col2 = st.columns([2,1])
    with col1:
        if df_rank.empty:
            st.info("Geen maatregelen actief.")
        else:
            show_cols = ["naam", "benefit", "penalty", "score"] + CRITERIA + CONSTRAINTS
            st.dataframe(df_rank[show_cols].reset_index(drop=True), use_container_width=True, height=420)
    with col2:
        if not df_rank.empty:
            top5 = df_rank.head(5)[["naam", "score"]].copy()
            top5["score_pct"] = (top5["score"]*100).round(0)
            fig_bar = px.bar(top5, x="naam", y="score_pct", text="score_pct", range_y=[0,100])
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(yaxis_title="Score (0–100)", xaxis_title="", margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig_bar, use_container_width=True)

# 2) Waterfall & radar for a selected measure
with tabs[1]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        sel_name = st.selectbox("Kies maatregel voor uitleg", df_rank["naam"])
        row = df_rank[df_rank["naam"]==sel_name].iloc[0]

        # Waterfall: contributions + penalties -> final
        contrib = [row[f"w_{c}"] for c in CRITERIA]
        pen_norm = st.session_state.penalties
        pen_sum = sum(pen_norm.values()) or 1.0
        pens = [ (pen_norm[c] * (row[c]/5.0)) / pen_sum for c in CONSTRAINTS ]

        wf = go.Figure(go.Waterfall(
            name = "score",
            orientation = "v",
            measure = ["relative"]*len(CRITERIA) + ["relative"]*len(CONSTRAINTS) + ["total"],
            x = [c.capitalize() for c in CRITERIA] + [f"- {c}" for c in CONSTRAINTS] + ["Totaal"],
            textposition = "outside",
            y = contrib + [ -lambda_pen * p for p in pens ] + [0.0],
        ))
        wf.update_layout(title=f"Scoresamenstelling – {row['naam']}", showlegend=False, margin=dict(t=30,b=20,l=10,r=10))
        st.plotly_chart(wf, use_container_width=True)

        # Radar of raw 0–5 per criterion
        radar_df = pd.DataFrame({"criterium": [c.capitalize() for c in CRITERIA],
                                 "waarde": [row[c] for c in CRITERIA]})
        fig_radar = px.line_polar(radar_df, r="waarde", theta="criterium", line_close=True)
        fig_radar.update_traces(fill="toself")
        fig_radar.update_polars(radialaxis=dict(range=[0,5]))
        st.plotly_chart(fig_radar, use_container_width=True)

# 3) Effects overview: stacked contributions + heatmap
with tabs[2]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        topN = st.slider("Toon top-N voor grafieken", 3, min(10, len(df_rank)), 5, 1)
        # Stacked contributions to benefit (per criterion)
        stack_df = df_rank.head(topN).copy()
        for c in CRITERIA:
            stack_df[c.capitalize()] = stack_df[f"w_{c}"]
        melted = stack_df.melt(id_vars=["naam"], value_vars=[c.capitalize() for c in CRITERIA],
                               var_name="criterium", value_name="bijdrage")
        fig_stack = px.bar(melted, x="naam", y="bijdrage", color="criterium", title="Bijdrage per criterium aan 'benefit' (gewogen)")
        st.plotly_chart(fig_stack, use_container_width=True)

        # Heatmap of raw 0-5 scores per criterion
        hm = df_rank.set_index("naam")[CRITERIA].rename(columns={c:c.capitalize() for c in CRITERIA})
        fig_hm = px.imshow(hm, aspect="auto", text_auto=True, title="Heatmap – criteria (0–5)")
        st.plotly_chart(fig_hm, use_container_width=True)

# 4) Pareto frontier & scatter
with tabs[3]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        # Scatter: benefit vs penalty, bubble size by costs or space
        size_by = st.selectbox("Grootte bubbel", ["kosten", "ruimteclaim", "beheerlast"])
        fig_sc = px.scatter(df_rank, x="penalty", y="benefit", size=size_by, hover_name="naam",
                            title="Benefit vs Penalty – selecteer efficiënte maatregelen")
        st.plotly_chart(fig_sc, use_container_width=True)

        # Pareto front highlight
        front = pareto_front(df_rank)
        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(x=df_rank["penalty"], y=df_rank["benefit"],
                                    mode="markers", name="Alle maatregelen",
                                    text=df_rank["naam"]))
        fig_pf.add_trace(go.Scatter(x=front["penalty"], y=front["benefit"],
                                    mode="markers+lines", name="Pareto-front",
                                    text=front["naam"]))
        fig_pf.update_layout(xaxis_title="Penalty (lager is beter)",
                             yaxis_title="Benefit (hoger is beter)",
                             title="Pareto-front – efficiënte keuzes")
        st.plotly_chart(fig_pf, use_container_width=True)

# 5) Scenario comparison (side-by-side)
with tabs[4]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")
    else:
        colA, colB, colC = st.columns(3)
        presets = {
            "Gebalanceerd": WEIGHT_DEFAULTS,
            "Hitte-focus": {"hitte":0.5, "wateroverlast":0.15, "droogte":0.1, "biodiversiteit":0.1, "sociale_cohesie":0.05, "toegankelijkheid":0.05, "beleving":0.05},
            "Water-focus": {"hitte":0.15,"wateroverlast":0.5,"droogte":0.15,"biodiversiteit":0.08,"sociale_cohesie":0.05,"toegankelijkheid":0.04,"beleving":0.03},
            "Droogte-focus": {"hitte":0.15,"wateroverlast":0.15,"droogte":0.5,"biodiversiteit":0.08,"sociale_cohesie":0.05,"toegankelijkheid":0.04,"beleving":0.03},
        }
        for col, (name, w) in zip([colA,colB,colC], list(presets.items())[:3]):
            with col:
                W = normalize_weights(w)
                d, avg = compute_scores(df_raw, active_ids, W, st.session_state.penalties, lam=lambda_pen)
                st.markdown(f"**{name}** — Gemiddelde: **{round(avg*100):.0f}/100**")
                st.dataframe(d[["naam","score"]].head(5).reset_index(drop=True), use_container_width=True, height=240)

# ----------------------------
# Export buttons
# ----------------------------
st.markdown("---")
colx, coly = st.columns([1,1])
with colx:
    csv_current = df_raw.to_csv(index=False).encode("utf-8")
    st.download_button("Download huidige scoringsmatrix (CSV)", csv_current, file_name="kattenburg_scoring.csv", mime="text/csv")
with coly:
    if not df_rank.empty:
        csv_rank = df_rank.to_csv(index=False).encode("utf-8")
        st.download_button("Download ranking/analyses (CSV)", csv_rank, file_name="kattenburg_ranking.csv", mime="text/csv")
