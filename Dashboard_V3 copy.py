"""Streamlit-dashboard dat ik gebruik om klimaatadaptieve maatregelen voor Kattenburg te vergelijken."""

import streamlit as st  # Streamlit levert de webinterface
import pandas as pd  # Pandas beheert de scoringsmatrix en filters
import plotly.express as px  # Plotly Express gebruik ik voor de meeste grafieken
import plotly.graph_objects as go  # Graph Objects dekt onder andere de waterfall en pareto grafieken

st.set_page_config(page_title="Kattenburg - Leefbaarheid Klimaatadaptatie", layout="wide")  # breed canvas zodat alles past

CRITERIA = [
    "hitte",
    "wateroverlast",
    "droogte",
    "biodiversiteit",
    "sociale_cohesie",
    "toegankelijkheid",
    "beleving",
]  # hoofdeffecten die ik wil verbeteren
CONSTRAINTS = ["ruimteclaim", "kosten", "beheerlast"]  # beperkende factoren die meetellen

WEIGHT_DEFAULTS = {
    "hitte": 0.22,
    "wateroverlast": 0.20,
    "droogte": 0.18,
    "biodiversiteit": 0.12,
    "sociale_cohesie": 0.12,
    "toegankelijkheid": 0.08,
    "beleving": 0.08,
}  # startwegingen waarmee ik meestal begin
PENALTY_DEFAULTS = {"ruimteclaim": 0.40, "kosten": 0.40, "beheerlast": 0.20}  # standaard strafpuntenmix

BASE_MEASURES = [
    {
        "id": "groene_daken",
        "naam": "Groene daken",
        "hitte": 4.0,
        "wateroverlast": 4.5,
        "droogte": 3.5,
        "biodiversiteit": 4.0,
        "sociale_cohesie": 2.5,
        "toegankelijkheid": 3.0,
        "beleving": 3.5,
        "ruimteclaim": 1.0,
        "kosten": 3.5,
        "beheerlast": 2.5,
    },
    {
        "id": "gevelgroen",
        "naam": "Gevelgroen",
        "hitte": 3.0,
        "wateroverlast": 1.5,
        "droogte": 1.5,
        "biodiversiteit": 3.0,
        "sociale_cohesie": 2.0,
        "toegankelijkheid": 3.5,
        "beleving": 4.0,
        "ruimteclaim": 0.5,
        "kosten": 2.0,
        "beheerlast": 2.0,
    },
    {
        "id": "wadi",
        "naam": "Wadi's & infiltratie",
        "hitte": 3.5,
        "wateroverlast": 4.5,
        "droogte": 4.0,
        "biodiversiteit": 4.0,
        "sociale_cohesie": 3.0,
        "toegankelijkheid": 2.5,
        "beleving": 3.5,
        "ruimteclaim": 3.5,
        "kosten": 3.0,
        "beheerlast": 2.5,
    },
    {
        "id": "doorlatend",
        "naam": "Waterdoorlatende bestrating",
        "hitte": 2.5,
        "wateroverlast": 3.5,
        "droogte": 3.0,
        "biodiversiteit": 1.5,
        "sociale_cohesie": 1.0,
        "toegankelijkheid": 3.5,
        "beleving": 2.5,
        "ruimteclaim": 1.0,
        "kosten": 2.0,
        "beheerlast": 2.0,
    },
    {
        "id": "regenwater",
        "naam": "Regenwateropvang & hergebruik",
        "hitte": 1.5,
        "wateroverlast": 3.5,
        "droogte": 4.0,
        "biodiversiteit": 1.0,
        "sociale_cohesie": 2.0,
        "toegankelijkheid": 4.0,
        "beleving": 2.0,
        "ruimteclaim": 1.0,
        "kosten": 2.5,
        "beheerlast": 2.0,
    },
    {
        "id": "bomen",
        "naam": "Schaduwrijke beplanting & bomen",
        "hitte": 5.0,
        "wateroverlast": 2.5,
        "droogte": 2.0,
        "biodiversiteit": 4.0,
        "sociale_cohesie": 3.5,
        "toegankelijkheid": 2.5,
        "beleving": 4.5,
        "ruimteclaim": 3.0,
        "kosten": 3.0,
        "beheerlast": 3.0,
    },
    {
        "id": "pocketparks",
        "naam": "Groene pleinen & pocketparks",
        "hitte": 4.0,
        "wateroverlast": 3.0,
        "droogte": 2.5,
        "biodiversiteit": 3.5,
        "sociale_cohesie": 4.5,
        "toegankelijkheid": 3.0,
        "beleving": 4.5,
        "ruimteclaim": 4.0,
        "kosten": 3.5,
        "beheerlast": 3.0,
    },
    {
        "id": "verhoogde_trottoirs",
        "naam": "Verhoogde trottoirs & tijdelijke waterberging",
        "hitte": 1.5,
        "wateroverlast": 4.0,
        "droogte": 1.0,
        "biodiversiteit": 1.0,
        "sociale_cohesie": 1.0,
        "toegankelijkheid": 2.0,
        "beleving": 1.5,
        "ruimteclaim": 1.5,
        "kosten": 2.5,
        "beheerlast": 2.0,
    },
    {
        "id": "onttegel",
        "naam": "Verminderen verhard oppervlak (onttegelen)",
        "hitte": 3.5,
        "wateroverlast": 3.5,
        "droogte": 3.0,
        "biodiversiteit": 3.5,
        "sociale_cohesie": 2.5,
        "toegankelijkheid": 3.0,
        "beleving": 3.0,
        "ruimteclaim": 2.0,
        "kosten": 1.0,
        "beheerlast": 2.0,
    },
    {
        "id": "adaptief_bouwen",
        "naam": "Adaptief bouwen & ontwerpen",
        "hitte": 3.5,
        "wateroverlast": 3.5,
        "droogte": 3.0,
        "biodiversiteit": 2.5,
        "sociale_cohesie": 2.0,
        "toegankelijkheid": 3.5,
        "beleving": 3.0,
        "ruimteclaim": 2.0,
        "kosten": 4.0,
        "beheerlast": 2.5,
    },
]  # referentieset als ik geen CSV upload


def normalize_weights(w: dict) -> dict:
    """Normaliseer de gewichten zodat ze samen precies 1 vormen."""
    total = sum(w.values()) or 1.0  # vang het geval af waarin alles nul is
    return {k: v / total for k, v in w.items()}  # schaal elk gewicht relatief op


def compute_scores(
    df: pd.DataFrame,
    active_ids,
    w_norm: dict,
    penalties: dict,
    lam: float = 0.5,
):
    """Bereken bijdrage, penalty en totaalscore voor de geselecteerde maatregelen."""
    df = df.copy()  # raak de oorspronkelijke dataframe niet direct aan
    df["active"] = df["id"].isin(active_ids)  # markeer maatregelen die aangezet zijn
    active_df = df[df["active"]].copy()  # filter alleen de rijen die mee mogen doen
    if active_df.empty:
        return active_df, 0.0  # zonder selectie is er ook geen score

    for col in CRITERIA + CONSTRAINTS:
        active_df[col] = (
            pd.to_numeric(active_df[col], errors="coerce").fillna(0.0).clip(0, 5)
        )  # zorg dat alles numeriek is en binnen 0-5 blijft

    for col in CRITERIA:
        active_df[f"w_{col}"] = w_norm.get(col, 0.0) * (active_df[col] / 5.0)  # gewogen bijdrage per criterium
    active_df["benefit"] = active_df[[f"w_{c}" for c in CRITERIA]].sum(axis=1)  # sommeer alle pluspunten

    penalty_sum = sum(penalties.values()) or 1.0  # normaliseer de strafpunten
    active_df["penalty"] = (
        sum(penalties[p] * (active_df[p] / 5.0) for p in CONSTRAINTS) / penalty_sum
    )  # gemiddeld nadeel per maatregel

    active_df["score"] = (active_df["benefit"] - lam * active_df["penalty"]).clip(0.0, 1.0)  # eindscore tussen 0 en 1
    active_df = active_df.sort_values("score", ascending=False)  # hoogste score bovenaan
    avg_score = float(active_df["score"].mean()) if not active_df.empty else 0.0  # totale indicatie voor KPI
    return active_df, avg_score  # geef de gescoorde set en de gemiddelde score terug


def pareto_front(df: pd.DataFrame, x: str = "penalty", y: str = "benefit"):
    """Zoek de niet-gedomineerde opties zodat ik de pareto-lijn kan tekenen."""
    if df.empty:
        return df  # als er niets actief is valt er niets te plotten
    points = df[[x, y]].to_numpy()  # pak de relevante assen als matrix
    keep = []  # hier verzamel ik de indexen die niet gedomineerd worden
    for i in range(len(points)):
        xi, yi = points[i]  # neem de huidige kandidaat
        dominated = False  # start vanuit het idee dat hij behouden blijft
        for j in range(len(points)):
            if i == j:
                continue  # sla dezelfde rij over
            xj, yj = points[j]  # vergelijk met een andere maatregel
            if (xj <= xi and yj >= yi) and (xj < xi or yj > yi):
                dominated = True  # deze maatregel is gedomineerd op beide assen
                break
        if not dominated:
            keep.append(i)  # alleen de niet-gedomineerde indexen bewaar ik
    return df.iloc[keep]  # terug naar een dataframe zodat plotly ermee overweg kan


st.title("Kattenburg - Leefbaarheidsscore Klimaatadaptatie")  # hoofdtitel in de app
st.caption("Ik test hier maatregelen, pas de wegingen aan en volg live welke combinatie het meest oplevert voor de leefbaarheid.")  # korte uitleg waarom dit dashboard bestaat

uploaded = st.sidebar.file_uploader("Laad een eigen scoringsmatrix (CSV)", type=["csv"])  # optie om mijn eigen data in te lezen
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)  # lees de aangeleverde csv in pandas
        required_columns = set(["id", "naam"] + CRITERIA + CONSTRAINTS)  # kolommen die ik minimaal nodig heb
        if not required_columns.issubset(df_raw.columns):
            st.warning("De CSV mist verplichte kolommen, dus ik val terug op de voorbeelddata.")  # melding als de structuur afwijkt
            df_raw = pd.DataFrame(BASE_MEASURES)  # terug naar de ingebakken dataset
    except Exception:
        st.warning("De CSV kon niet worden gelezen, dus ik gebruik de voorbeelddata.")  # melding bij leesfout
        df_raw = pd.DataFrame(BASE_MEASURES)  # fallback naar standaarddata
else:
    df_raw = pd.DataFrame(BASE_MEASURES)  # standaard dataset als er niets is geupload

st.sidebar.subheader("Maatregelen")  # sectie voor de selectie van maatregelen
all_ids = df_raw["id"].tolist()  # volgorde van maatregelen zoals ze in de dataset staan
active_ids = st.sidebar.multiselect(
    "Actieve maatregelen",
    options=all_ids,
    default=all_ids,
    format_func=lambda x: df_raw.loc[df_raw["id"] == x, "naam"].values[0],
)  # lijst waarmee ik maatregelen aan of uit zet

if "weights" not in st.session_state:
    st.session_state.weights = WEIGHT_DEFAULTS.copy()  # start met de standaardwegingen
if "penalties" not in st.session_state:
    st.session_state.penalties = PENALTY_DEFAULTS.copy()  # idem voor penalties

st.sidebar.subheader("Wegingen (effecten)")  # blok met de effect-schuiven
for criterion in CRITERIA:
    st.session_state.weights[criterion] = st.sidebar.slider(
        criterion.capitalize(),
        0.0,
        1.0,
        float(st.session_state.weights[criterion]),
        0.01,
        key=f"w_{criterion}",
    )  # per criterium stel ik eenvoudig het gewicht bij

st.sidebar.subheader("Strafpunten (constraints)")  # blok met constraints
for constraint in CONSTRAINTS:
    st.session_state.penalties[constraint] = st.sidebar.slider(
        constraint.capitalize(),
        0.0,
        1.0,
        float(st.session_state.penalties[constraint]),
        0.01,
        key=f"p_{constraint}",
    )  # hiermee geef ik aan hoe zwaar elke constraint moet wegen

lambda_penalty = st.sidebar.slider("Penalty-factor (lambda)", 0.0, 1.0, 0.5, 0.01, key="lambda")  # globale factor om penalties mee te dempen of te versterken


def apply_preset(new_weights: dict):
    """Schrijf een preset terug naar de sliders zodat ik snel scenario's kan testen."""
    st.session_state.weights.update(new_weights)  # update het interne profiel
    for key, value in new_weights.items():
        st.session_state[f"w_{key}"] = float(value)  # zet ook de sliderwaarden terug


st.sidebar.subheader("Scenario presets")  # blok waarmee ik favoriete profielen oproep
PRESETS = {
    "Gebalanceerd": WEIGHT_DEFAULTS.copy(),
    "Hitte-focus": {"hitte": 0.50, "wateroverlast": 0.15, "droogte": 0.10, "biodiversiteit": 0.10, "sociale_cohesie": 0.05, "toegankelijkheid": 0.05, "beleving": 0.05},
    "Water-focus": {"hitte": 0.15, "wateroverlast": 0.50, "droogte": 0.15, "biodiversiteit": 0.08, "sociale_cohesie": 0.05, "toegankelijkheid": 0.04, "beleving": 0.03},
    "Droogte-focus": {"hitte": 0.15, "wateroverlast": 0.15, "droogte": 0.50, "biodiversiteit": 0.08, "sociale_cohesie": 0.05, "toegankelijkheid": 0.04, "beleving": 0.03},
}  # presets waarmee ik snel kan wisselen van invalshoek
preset_choice = st.sidebar.selectbox("Kies scenario", list(PRESETS.keys()))  # combobox om een preset te kiezen
st.sidebar.button("Preset toepassen", on_click=apply_preset, args=(PRESETS[preset_choice],))  # knop die de preset naar de sliders schrijft

weights_norm = normalize_weights(st.session_state.weights)  # normaliseer de gekozen wegingen voor de berekening

df_rank, avg_score = compute_scores(
    df_raw,
    active_ids,
    weights_norm,
    st.session_state.penalties,
    lam=lambda_penalty,
)  # bereken direct de benefit, penalty en totaalscore

kpi1, kpi2, kpi3 = st.columns(3)  # drie KPI-tegels bovenaan
kpi1.metric("Gemiddelde totaalscore (actief)", f"{round(avg_score * 100):.0f}/100")  # geeft een globale indicatie
kpi2.metric("Aantal actieve maatregelen", len(df_rank))  # houdt bij hoeveel maatregelen nu meedoen
best_measure = df_rank.iloc[0]["naam"] if not df_rank.empty else "-"  # pak de beste naam voor de derde KPI
kpi3.metric("Beste maatregel", best_measure)  # toon meteen de huidige topper

st.markdown("---")  # visuele scheiding voor de tabs
tabs = st.tabs(
    ["Ranking & tabel", "Uitleg per maatregel", "Effecten-overzicht", "Pareto & selectie", "Scenario-vergelijking"]
)  # tabstructuur zodat ik de analyses kan scheiden

with tabs[0]:
    col_table, col_chart = st.columns([2, 1])  # combineer een tabel met een bar chart
    with col_table:
        if df_rank.empty:
            st.info("Geen maatregelen actief.")  # melding als er niets geselecteerd is
        else:
            show_cols = ["naam", "benefit", "penalty", "score"] + CRITERIA + CONSTRAINTS  # kolommen die ik relevant vind
            st.dataframe(
                df_rank[show_cols].reset_index(drop=True),
                use_container_width=True,
                height=420,
            )  # volledig overzicht van de scores
    with col_chart:
        if not df_rank.empty:
            top5 = df_rank.head(5)[["naam", "score"]].copy()  # pak de top vijf voor de grafiek
            top5["score_pct"] = (top5["score"] * 100).round(0)  # zet de score om naar procenten
            fig_bar = px.bar(top5, x="naam", y="score_pct", text="score_pct", range_y=[0, 100])  # simpele staafgrafiek
            fig_bar.update_traces(textposition="outside")  # plaats de labels buiten de staven
            fig_bar.update_layout(yaxis_title="Score (0-100)", xaxis_title="", margin=dict(t=10, b=10, l=10, r=10))  # hou de visual compact
            st.plotly_chart(fig_bar, use_container_width=True)  # render de chart naast de tabel

with tabs[1]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")  # niets om toe te lichten
    else:
        sel_name = st.selectbox("Kies maatregel", df_rank["naam"])  # kies welke maatregel ik wil uitlichten
        row = df_rank[df_rank["naam"] == sel_name].iloc[0]  # haal de bijbehorende rij op

        contrib = [row[f"w_{c}"] for c in CRITERIA]  # verzamel alle bijdragen
        penalty_sum = sum(st.session_state.penalties.values()) or 1.0  # normaliseer de strafpunten voor de visual
        penalties = [
            (st.session_state.penalties[c] * (row[c] / 5.0)) / penalty_sum for c in CONSTRAINTS
        ]  # schaal de penalties naar dezelfde range

        waterfall = go.Figure(
            go.Waterfall(
                name="score",
                orientation="v",
                measure=["relative"] * len(CRITERIA) + ["relative"] * len(CONSTRAINTS) + ["total"],
                x=[c.capitalize() for c in CRITERIA] + [f"- {c}" for c in CONSTRAINTS] + ["Totaal"],
                textposition="outside",
                y=contrib + [-lambda_penalty * p for p in penalties] + [0.0],
            )
        )  # waterfall laat de score-opbouw zien
        waterfall.update_layout(title=f"Scoresamenstelling - {row['naam']}", showlegend=False, margin=dict(t=30, b=20, l=10, r=10))  # titel en strakkere marges
        st.plotly_chart(waterfall, use_container_width=True)  # plot direct onder de selectbox

        radar_df = pd.DataFrame(
            {"criterium": [c.capitalize() for c in CRITERIA], "waarde": [row[c] for c in CRITERIA]}
        )  # dataset voor de radargrafiek
        radar_fig = px.line_polar(radar_df, r="waarde", theta="criterium", line_close=True)  # radar om het profiel te tonen
        radar_fig.update_traces(fill="toself")  # kleur het oppervlak in
        radar_fig.update_polars(radialaxis=dict(range=[0, 5]))  # houd dezelfde schaal aan
        st.plotly_chart(radar_fig, use_container_width=True)  # plot naast of onder waterfall afhankelijk van scherm

with tabs[2]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")  # zonder data geen overzicht
    else:
        max_top = max(1, min(10, len(df_rank)))  # houd de slider binnen een geldige range
        default_top = min(5, max_top)  # kies een passend default
        top_n = st.slider("Toon top-N", 1, max_top, default_top, 1)  # kies hoeveel maatregelen ik wil tonen
        stack_df = df_rank.head(top_n).copy()  # beperk de data tot de gekozen top
        for criterion in CRITERIA:
            stack_df[criterion.capitalize()] = stack_df[f"w_{criterion}"]  # hernoem voor leesbare labels
        melted = stack_df.melt(
            id_vars=["naam"],
            value_vars=[c.capitalize() for c in CRITERIA],
            var_name="criterium",
            value_name="bijdrage",
        )  # smelt naar lang formaat voor de stacked bar
        stack_fig = px.bar(
            melted,
            x="naam",
            y="bijdrage",
            color="criterium",
            title="Bijdrage per criterium aan de benefit (gewogen)",
        )  # stacked bar per criterium
        st.plotly_chart(stack_fig, use_container_width=True)  # render direct in de tab

        heatmap_df = df_rank.set_index("naam")[CRITERIA].rename(columns={c: c.capitalize() for c in CRITERIA})  # matrix voor de heatmap
        heatmap_fig = px.imshow(heatmap_df, aspect="auto", text_auto=True, title="Heatmap - criteria (0-5)")  # snel overzicht per criterium
        st.plotly_chart(heatmap_fig, use_container_width=True)  # toon de heatmap onder de stacked bar

with tabs[3]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")  # niets te plotten
    else:
        size_by = st.selectbox("Grootte van de bubbel", ["kosten", "ruimteclaim", "beheerlast"])  # kies welke constraint de bubbelgrootte bepaalt
        scatter = px.scatter(
            df_rank,
            x="penalty",
            y="benefit",
            size=size_by,
            hover_name="naam",
            title="Benefit vs penalty - balans zoeken",
        )  # bubble chart om snel de verhouding te zien
        scatter.update_layout(xaxis_title="Penalty (lager is beter)", yaxis_title="Benefit (hoger is beter)")  # duidelijke assen
        st.plotly_chart(scatter, use_container_width=True)  # laat de scatter zien

        front = pareto_front(df_rank)  # filter op de pareto-optimale punten
        pareto_fig = go.Figure()  # begin met een lege figure
        pareto_fig.add_trace(
            go.Scatter(x=df_rank["penalty"], y=df_rank["benefit"], mode="markers", name="Alle maatregelen", text=df_rank["naam"])
        )  # plot eerst alle punten
        if not front.empty:
            pareto_fig.add_trace(
                go.Scatter(x=front["penalty"], y=front["benefit"], mode="markers+lines", name="Pareto-front", text=front["naam"])
            )  # teken vervolgens de pareto-lijn
        pareto_fig.update_layout(
            xaxis_title="Penalty (lager is beter)",
            yaxis_title="Benefit (hoger is beter)",
            title="Pareto-front - efficiente keuzes",
        )  # geef de grafiek dezelfde assen
        st.plotly_chart(pareto_fig, use_container_width=True)  # render de pareto grafiek

with tabs[4]:
    if df_rank.empty:
        st.info("Geen maatregelen actief.")  # scenariovergelijking heeft data nodig
    else:
        col_a, col_b, col_c = st.columns(3)  # toon drie scenario's naast elkaar
        scenario_presets = {
            "Gebalanceerd": WEIGHT_DEFAULTS,
            "Hitte-focus": {"hitte": 0.50, "wateroverlast": 0.15, "droogte": 0.10, "biodiversiteit": 0.10, "sociale_cohesie": 0.05, "toegankelijkheid": 0.05, "beleving": 0.05},
            "Water-focus": {"hitte": 0.15, "wateroverlast": 0.50, "droogte": 0.15, "biodiversiteit": 0.08, "sociale_cohesie": 0.05, "toegankelijkheid": 0.04, "beleving": 0.03},
        }  # scenario's die ik vaak naast elkaar leg
        for column, (name, weights) in zip([col_a, col_b, col_c], scenario_presets.items()):
            with column:
                normalized = normalize_weights(weights)  # normaliseer per scenario
                scenario_df, scenario_avg = compute_scores(
                    df_raw,
                    active_ids,
                    normalized,
                    st.session_state.penalties,
                    lam=lambda_penalty,
                )  # herbereken voor het scenario
                st.markdown(f"**{name}** - Gemiddelde: **{round(scenario_avg * 100):.0f}/100**")  # zet de kop en KPI neer
                st.dataframe(
                    scenario_df[["naam", "score"]].head(5).reset_index(drop=True),
                    use_container_width=True,
                    height=240,
                )  # laat de top vijf van dat scenario zien

st.markdown("---")  # afsluiten met exportopties
col_data, col_scores = st.columns(2)  # twee kolommen voor downloads
with col_data:
    csv_current = df_raw.to_csv(index=False).encode("utf-8")  # exporteer de matrix zoals hij nu is
    st.download_button(
        "Download huidige scoringsmatrix (CSV)",
        csv_current,
        file_name="kattenburg_scoring.csv",
        mime="text/csv",
    )  # knop om de ruwe matrix op te slaan
with col_scores:
    if not df_rank.empty:
        csv_rank = df_rank.to_csv(index=False).encode("utf-8")  # exporteer de berekende ranking
        st.download_button(
            "Download ranking en analyses (CSV)",
            csv_rank,
            file_name="kattenburg_ranking.csv",
            mime="text/csv",
        )  # knop om de analyse te bewaren
