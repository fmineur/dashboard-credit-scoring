import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import json
import os

#TEST
# === Configuration API ===
API_URL = "https://m3r1n1-credit-scoring-api.hf.space/predict"

st.set_page_config(page_title="Dashboard Scoring", layout="wide")

# === Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("data/data_clients_dashboard.csv")
    shap_local = pd.read_csv("data/shap_local.csv", index_col=0)
    shap_global = pd.read_csv("data/shap_global.csv")
    return df, shap_local, shap_global

df, shap_local, shap_global = load_data()
group_means = pd.read_csv("data/grouped_means.csv", index_col=0)
stats = pd.read_csv("data/dashboard_stats.csv", index_col=0, header=None).squeeze("columns").to_dict()

# === Liste des clients (mise à jour automatique)
client_ids = df["SK_ID_CURR"].unique().tolist()
client_id = st.sidebar.selectbox("📌 ID client :", client_ids)

# === Navigation
view = st.sidebar.radio("📂 Section", [
    "Vue générale", "Facteurs d'influence", "Comparaison avec voisins",
    "Moyennes comparées", "Visualisation Scatter", "Simulation", "Saisie dossier"
])

client_data = df[df["SK_ID_CURR"] == client_id].iloc[0]

try:
    neighbors_raw = client_data["neighbors"]
    neighbors_ids = json.loads(neighbors_raw.replace("'", '"')) if isinstance(neighbors_raw, str) else []
    if not isinstance(neighbors_ids, list):
        neighbors_ids = []
except Exception:
    neighbors_ids = []

neighbors_data = df[df["SK_ID_CURR"].isin(neighbors_ids)]

top_features = shap_local.columns.tolist()


def format_number(val): return f"{int(val):,}".replace(",", " ") if pd.notnull(val) else "NC"


if view == "Vue générale":
    st.title("📊 Dashboard Crédit Scoring")
    col1, col2, col3, col4 = st.columns([1.3, 1.3, 1.2, 1.2])

    with col1:
        st.subheader("👤 Données client")
        st.write({
            "Sexe": "Male" if client_data["CODE_GENDER"] == "M" else "Female",
            "Revenu": f"{client_data['AMT_INCOME_TOTAL']:,.2f}".replace(",", " ").replace(".00", ""),
            "Enfants": int(client_data["CNT_CHILDREN"]),
            "Situation": client_data["NAME_FAMILY_STATUS"],
            "Éducation": client_data["NAME_EDUCATION_TYPE"],
            "Type revenu": client_data["NAME_INCOME_TYPE"],
            "Type logement": client_data["NAME_HOUSING_TYPE"],
            "Âge": max(0, int(client_data["AGE"])) if pd.notnull(client_data["AGE"]) else "NC",
            "Ancienneté emploi (ans)": max(0, int(client_data["YEARS_EMPLOYED"])) if pd.notnull(client_data["YEARS_EMPLOYED"]) else "NC"
        })

    with col2:
        st.subheader("🏦 Données crédit")
        st.write({
            "Type contrat": client_data["NAME_CONTRACT_TYPE"],
            "Montant crédit": f"{client_data['AMT_CREDIT']:,.2f}".replace(",", " ").replace(".00", ""),
            "Montant annuité": f"{client_data['AMT_ANNUITY']:,.2f}".replace(",", " ").replace(".00", ""),
            "Montant biens": f"{client_data['AMT_GOODS_PRICE']:,.2f}".replace(",", " ").replace(".00", ""),
        })

    with col3:
        st.subheader("🧮 Données modèle (1/2)")
        for i in range(10):
            st.write(f"{top_features[i]}: {round(client_data[top_features[i]], 2)}")

    with col4:
        st.subheader("🧮 Données modèle (2/2)")
        for i in range(10, 20):
            st.write(f"{top_features[i]}: {round(client_data[top_features[i]], 2)}")

    st.markdown("---")

    input_data = client_data[top_features].astype(float).to_dict()
    warning_message = False
    try:
        r = requests.post(API_URL, json=input_data)
        if r.status_code == 200:
            result = r.json()
            proba = float(result["probability"]) * 100
            pred = int(result["prediction"])
        else:
            raise ValueError("Erreur API")
    except:
        proba = float(client_data["probability"]) * 100
        pred = int(client_data["prediction"])
        warning_message = True

    seuils = [18, 40, 70]
    col_gauge, col_statut = st.columns([3, 1])
    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(proba, 2),
            delta={"reference": seuils[0], "increasing": {"color": "red"}, "decreasing": {"color": "green"}, "valueformat": ".2f"},
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "white"},
                "steps": [
                    {"range": [0, seuils[0]], "color": "green"},
                    {"range": [seuils[0], seuils[1]], "color": "yellowgreen"},
                    {"range": [seuils[1], seuils[2]], "color": "orange"},
                    {"range": [seuils[2], 100], "color": "red"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": seuils[0]},
            },
            title={"text": "Probabilité de défaut"}
        ))
        st.plotly_chart(gauge)

    with col_statut:
        statut = "💥 Risque de défaut" if pred == 1 else "✅ Client sain"
        st.markdown(f"### **Statut : {statut}**")
        st.markdown("#### Moyennes contextuelles")
        st.write({
            "Voisins": f"{client_data['mean_proba_neighbors'] * 100:.2f}%",
            "Global": f"{stats['mean_proba_global'] * 100:.2f}%",
            "Taux défaut voisins": f"{client_data['default_rate_neighbors'] * 100:.2f}%",
            "Taux défaut global": f"{stats['defaut_rate_global'] * 100:.2f}%"
        })

    if warning_message:
        st.warning("⚠️ Utilisation des valeurs locales de prédiction")


# -------------------- SECTION : SHAP --------------------
elif view == "Facteurs d'influence":
    st.title("🧠 SHAP - Facteurs d'influence")

    import shap
    import matplotlib.pyplot as plt
    from mlflow.sklearn import load_model

    try:
        model = load_model("../models/LightGBM Top Features_final")
        X_client = pd.DataFrame([client_data[top_features]], columns=top_features).astype(float)

        X_background = df[top_features].sample(n=100, random_state=42)
        explainer = shap.TreeExplainer(model, data=X_background, model_output="probability")
        shap_values = explainer(X_client)

        st.subheader("🔍 Importance locale (Waterfall)")
        shap.plots.waterfall(shap_values[0], show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 4)  # ✅ format compact
        st.pyplot(fig)
        st.markdown(f"**🔹 Moyenne prédiction fond (E[f(X)]) :** {shap_values.base_values[0]:.4f}")

    except Exception as e:
        st.warning("⚠️ Affichage SHAP waterfall non supporté ici.")
        st.error(f"(erreur : {e})")

    st.subheader("📊 Importance globale (% de contribution)")

    try:
        shap_global_sorted = shap_global.sort_values("Percent_Contribution", ascending=True).tail(20)
        fig, ax = plt.subplots(figsize=(10, 7))  # ✅ graphique global réduit
        ax.barh(shap_global_sorted["Feature"], shap_global_sorted["Percent_Contribution"], color="skyblue")

        for i, v in enumerate(shap_global_sorted["Percent_Contribution"]):
            ax.text(v + 0.025, i, f"{v:.1f}%", va="center")

        ax.set_title("Top 20 variables influentes")
        ax.set_xlabel("Contribution (%)")
        st.pyplot(fig)

    except Exception as e:
        st.warning("❌ Erreur affichage SHAP global")
        st.text(str(e))

elif view == "Comparaison avec voisins":
    st.title("👥 Comparaison avec les voisins")

    if not isinstance(neighbors_ids, list) or len(neighbors_ids) == 0:
        st.info(f"📭 Pas de voisins enregistrés pour le client {client_id}.")
    else:
        infos_client = ["CODE_GENDER", "AMT_INCOME_TOTAL", "CNT_CHILDREN", "NAME_FAMILY_STATUS",
                        "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE", "AGE", "YEARS_EMPLOYED"]
        infos_credit = ["NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]

        mapping_client = {
            "CODE_GENDER": "Sexe", "AMT_INCOME_TOTAL": "Revenu", "CNT_CHILDREN": "Enfants",
            "NAME_FAMILY_STATUS": "Situation", "NAME_EDUCATION_TYPE": "Éducation",
            "NAME_INCOME_TYPE": "Type revenu", "NAME_HOUSING_TYPE": "Type logement",
            "AGE": "Âge", "YEARS_EMPLOYED": "Ancienneté emploi (ans)"
        }
        mapping_credit = {
            "NAME_CONTRACT_TYPE": "Type contrat", "AMT_CREDIT": "Montant crédit",
            "AMT_ANNUITY": "Montant annuité", "AMT_GOODS_PRICE": "Montant biens"
        }

        # Construction propre : client + voisins sans duplicata
        df_client_row = df[df["SK_ID_CURR"] == client_id].head(1)
        df_neighbors = neighbors_data[~neighbors_data["SK_ID_CURR"].eq(client_id)]
        df_comp = pd.concat([df_client_row, df_neighbors]).reset_index(drop=True)

        # === INFOS CLIENTS ===
        st.markdown("### 📋 Infos clients")
        df_cli = df_comp[["SK_ID_CURR"] + infos_client].rename(columns=mapping_client)

        # Formatage des colonnes numériques non entières
        for col in ["Revenu"]:
            df_cli[col] = df_cli[col].map(lambda x: f"{x:,.2f}".replace(",", " ") if pd.notnull(x) else "NC")

        def highlight_similar(row):
            return [
                "background-color: lightgreen" if row[col] == df_cli.iloc[0][col] and row.name != 0 else ""
                for col in row.index
            ]

        st.dataframe(df_cli.style.apply(highlight_similar, axis=1), use_container_width=True, height=40 * len(df_cli))

        # === INFOS CRÉDIT ===
        st.markdown("### 💳 Infos crédit")
        df_crd = df_comp[["SK_ID_CURR"] + infos_credit].rename(columns=mapping_credit)

        for col in ["Montant crédit", "Montant annuité", "Montant biens"]:
            df_crd[col] = df_crd[col].map(lambda x: f"{x:,.2f}".replace(",", " ") if pd.notnull(x) else "NC")

        def highlight_credit(row):
            return [
                "background-color: lightgreen" if row[col] == df_crd.iloc[0][col] and row.name != 0 else ""
                for col in row.index
            ]

        st.dataframe(df_crd.style.apply(highlight_credit, axis=1), use_container_width=True, height=40 * len(df_crd))

elif view == "Moyennes comparées":
    st.title("📊 Comparaison aux groupes (sains vs défaut)")
    var = st.selectbox("Variable à comparer :", shap_local.columns.tolist())

    val_client = client_data[var]
    val_sains = group_means.loc[0, var]
    val_def = group_means.loc[1, var]

    data_plot = {
        "Client": val_client,
        "Sains": val_sains,
        "Défaut": val_def
    }

    if isinstance(neighbors_ids, list) and len(neighbors_ids) > 0:
        val_voisins = neighbors_data[var].mean()
        data_plot["Voisins"] = val_voisins

    fig = px.bar(
        x=list(data_plot.keys()),
        y=list(data_plot.values()),
        color=list(data_plot.keys()),
        color_discrete_map={
            "Client": "blue", "Voisins": "lightblue", "Sains": "green", "Défaut": "red"
        },
        height=720, width=1000
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=30))
    st.plotly_chart(fig, use_container_width=False)

    for k, v in data_plot.items():
        st.markdown(f"- **{k}** : {v:.2f}")


elif view == "Visualisation Scatter":
    st.title("📈 Scatter Plot")
    variables = top_features + ["AGE", "YEARS_EMPLOYED", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"]
    x = st.selectbox("Variable X", variables)
    y = st.selectbox("Variable Y", variables, index=1)

    fig = px.scatter(df, x=x, y=y, opacity=0.3, color_discrete_sequence=["gray"])
    fig.add_scatter(x=[client_data[x]], y=[client_data[y]], mode="markers",
                    marker=dict(color="red", size=10), name="Client")

    fig.update_layout(
    height=840,
    width=1600,
    margin=dict(l=10, r=10, t=30, b=30)
    )

    st.plotly_chart(fig, use_container_width=False)

elif view == "Simulation":
    st.title("🧪 Simulation client")

    # Changement de client → réinit
    if "previous_client_id" not in st.session_state:
        st.session_state.previous_client_id = client_id

    if client_id != st.session_state.previous_client_id:
        st.session_state.sim_values = {feat: float(client_data[feat]) for feat in top_features}
        st.session_state.previous_client_id = client_id

    # Init si première fois
    if "sim_values" not in st.session_state:
        st.session_state.sim_values = {feat: float(client_data[feat]) for feat in top_features}

    # Interface de saisie
    new_values = {}
    cols = st.columns(2)
    for i, col in enumerate(top_features):
        with cols[i % 2]:
            new_values[col] = st.number_input(col, value=st.session_state.sim_values.get(col, float(client_data[col])))

    # Mise à jour en une seule fois
    st.session_state.sim_values = new_values

    # Réinitialisation manuelle
    if st.button("🔁 Réinitialiser"):
        st.session_state.sim_values = {feat: float(client_data[feat]) for feat in top_features}
        st.rerun()

    # Prédiction
    if st.button("🚀 Prédiction"):
        try:
            sim_values_clean = {col: st.session_state.sim_values[col] for col in top_features}
            r = requests.post(API_URL, json=sim_values_clean)
            if r.status_code == 200:
                result = r.json()
                proba = float(result["probability"]) * 100
                pred = int(result["prediction"])

                statut = "💥 Risque de défaut" if pred == 1 else "✅ Client sain"
                st.subheader("🎯 Résultat simulation")
                st.markdown(f"### **Statut : {statut} — Probabilité : {proba:.2f}%**")

                st.markdown("### 📋 Comparaison des valeurs")
                comp_df = pd.DataFrame({
                    "Valeur initiale": client_data[top_features],
                    "Valeur simulée": pd.Series(sim_values_clean)
                }).loc[top_features]

                def surligne_diff(x):
                    return ["background-color: lightblue" if x["Valeur initiale"] != x["Valeur simulée"] else "" for _ in x.index]

                st.dataframe(
                    comp_df.style.apply(surligne_diff, axis=1),
                    use_container_width=True,
                    height=40 * len(comp_df)
                )
            else:
                st.error("Erreur API")
        except Exception as e:
            st.error(f"Erreur : {e}")


elif view == "Saisie dossier":
    st.title("📝 Saisie nouveau dossier client")

    ref = df[df["SK_ID_CURR"] == 100001].iloc[0]
    top_feats = top_features if "top_features" in locals() else shap_local.columns.tolist()

    st.subheader("👤 Données client")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender_map = {"F": "Female", "M": "Male"}
        ref_gender = gender_map.get(ref["CODE_GENDER"], "Female")
        sexe = st.selectbox("Sexe", ["Male", "Female"], index=["Male", "Female"].index(ref_gender))
        enfants = st.number_input("Enfants", value=int(ref["CNT_CHILDREN"]))
    with col2:
        revenu = st.number_input("Revenu", value=float(ref["AMT_INCOME_TOTAL"]))
        emploi = st.number_input("Ancienneté emploi (ans)", value=max(0, int(ref["YEARS_EMPLOYED"])))
    with col3:
        situation = st.selectbox("Situation", [
            "Civil marriage", "Married", "Separated", "Single / not married", "Widow"
        ], index=1)
        logement = st.selectbox("Type logement", [
            "Co-op apartment", "House / apartment", "Municipal apartment", "Office apartment", "Rented apartment", "With parents"
        ], index=1)
    with col4:
        education = st.selectbox("Éducation", [
            "Academic degree", "Higher education", "Incomplete higher", "Lower secondary", "Secondary / secondary special"
        ], index=1)
        revenu_type = st.selectbox("Type revenu", [
            "Businessman", "Commercial associate", "Pensioner", "State Servant", "Student", "Unemployed", "Working"
        ], index=6)
        age = st.number_input("Âge", value=max(0, int(ref["AGE"])))

    st.subheader("🏦 Données crédit")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        contrat = st.selectbox("Type contrat", ["Cash loans", "Revolving loans"], index=0)
    with col6:
        credit = st.number_input("Montant crédit", value=float(ref["AMT_CREDIT"]))
    with col7:
        annuite = st.number_input("Montant annuité", value=float(ref["AMT_ANNUITY"]))
    with col8:
        biens = st.number_input("Montant biens", value=float(ref["AMT_GOODS_PRICE"]))

    st.subheader("🧮 Données modèle (20 variables)")
    input_model = {}
    cols = st.columns(2)
    for i, feat in enumerate(top_feats):
        with cols[i % 2]:
            input_model[feat] = st.number_input(feat, value=float(ref[feat]))

    if st.button("🚀 Prédiction"):
        try:
            r = requests.post(API_URL, json=input_model)
            if r.status_code == 200:
                result = r.json()
                proba = float(result["probability"]) * 100
                pred = int(result["prediction"])
                statut = "💥 Risque de défaut" if pred == 1 else "✅ Client sain"
                st.subheader("🎯 Résultat simulation")
                st.markdown(f"### **Statut : {statut} — Probabilité : {proba:.2f}%**")
            else:
                st.error("❌ Erreur lors de la requête API.")
        except Exception as e:
            st.error(f"❌ Erreur API : {e}")

    if st.button("💾 Enregistrer le dossier"):

        try:
            df_clients = pd.read_csv("data/data_clients_dashboard.csv")
            new_id = df_clients["SK_ID_CURR"].max() + 1

            row = {
                "SK_ID_CURR": new_id,
                "CODE_GENDER": sexe,
                "AMT_INCOME_TOTAL": revenu,
                "CNT_CHILDREN": enfants,
                "NAME_FAMILY_STATUS": situation,
                "NAME_EDUCATION_TYPE": education,
                "NAME_INCOME_TYPE": revenu_type,
                "NAME_HOUSING_TYPE": logement,
                "AGE": age,
                "YEARS_EMPLOYED": emploi,
                "NAME_CONTRACT_TYPE": contrat,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuite,
                "AMT_GOODS_PRICE": biens,
                **input_model
            }

            df_clients = pd.concat([df_clients, pd.DataFrame([row])], ignore_index=True)
            df_clients.to_csv("data/data_clients_dashboard.csv", index=False)

            st.success(f"✅ Nouveau dossier enregistré avec SK_ID_CURR = {new_id}")
            st.cache_data.clear()
            st.rerun()

        except Exception as e:
            st.error(f"Erreur lors de l'enregistrement : {e}")

