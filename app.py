import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
from sqlalchemy import create_engine
import re

# Configuration de la page
st.set_page_config(page_title="Recherche de Talents", page_icon="🔍", layout="wide")

# Dictionnaire de normalisation des termes techniques
TECH_NORMALIZATION = {
    r'\.net|dot[\s-]?net|dontnet': '.net',
    r'javascript|js': 'javascript',
    r'c#|c sharp': 'c#',
    r'c\+\+|cpp': 'c++',
    # Ajoutez d'autres normalisations ici au besoin
}

def normalize_tech_term(term):
    """Normalise un terme technique selon le dictionnaire de normalisation"""
    term_lower = term.lower()
    for pattern, normalized in TECH_NORMALIZATION.items():
        if re.search(pattern, term_lower):
            return normalized
    return term_lower

def database_connection():
    st.sidebar.title("🔐 Connexion à la base de données")

    host = st.sidebar.text_input("Hôte", value="localhost")
    database = st.sidebar.text_input("Nom de la base", value="medusa")
    user = st.sidebar.text_input("Utilisateur", value="postgres")
    password = st.sidebar.text_input("Mot de passe", type="password", value="admin")
    port = st.sidebar.text_input("Port", value="5432")

    if st.sidebar.button("Se connecter"):
        try:
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
            conn.close()
            st.session_state['db_params'] = {
                'host': host,
                'database': database,
                'user': user,
                'password': password,
                'port': port
            }
            st.session_state['connected'] = True
            st.sidebar.success("Connexion réussie!")
        except Exception as e:
            st.sidebar.error(f"Échec de la connexion: {e}")

    return st.session_state.get('connected', False)

if not database_connection():
    st.warning("Veuillez configurer la connexion à la base de données dans la barre latérale")
    st.stop()

@st.cache_data
def load_data():
    db_params = st.session_state['db_params']
    engine = create_engine(
        f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@"
        f"{db_params['host']}:{db_params['port']}/{db_params['database']}"
    )
    query = """
    SELECT
        poste, nom, annees_experience, localisation,
        competences, formations, certifications, languages,
        mode_travail, temps_travail, disponibilite,
        region, department, mobilite,
        inter_exp, level_experience, key_word_ia
    FROM medusa.profiles;
    """
    df = pd.read_sql(query, engine)

    df['inter_exp'] = df['inter_exp'].fillna('Non spécifié')
    df['level_experience'] = df['level_experience'].fillna('Non spécifié')
    df['key_word_ia'] = df['key_word_ia'].fillna('')

    return df

def filter_by_tech(df, tech_keywords):
    if not tech_keywords:
        return df

    # Normaliser les termes de recherche
    normalized_keywords = [normalize_tech_term(kw) for kw in tech_keywords]
    
    # Créer une regex pour chaque terme normalisé
    regex_patterns = []
    for kw in normalized_keywords:
        if kw in TECH_NORMALIZATION.values():
            # Trouver tous les patterns qui mappent à cette valeur normalisée
            patterns = [p for p, v in TECH_NORMALIZATION.items() if v == kw]
            regex_patterns.append(f"({'|'.join(patterns)})")
        else:
            regex_patterns.append(re.escape(kw))
    
    # Compiler une regex unique pour tous les termes
    combined_pattern = re.compile('|'.join(regex_patterns), re.IGNORECASE)

    mask = (
        df['competences'].str.lower().fillna('').apply(lambda text: bool(combined_pattern.search(text))) | 
        df['poste'].str.lower().fillna('').apply(lambda text: bool(combined_pattern.search(text))) | 
        df['key_word_ia'].str.lower().fillna('').apply(lambda text: bool(combined_pattern.search(text)))
    )

    return df[mask]

df = load_data()

# Interface utilisateur
st.title("🔍 Recherche de Profils IT")

# -------- Sidebar avec filtres --------
with st.sidebar:
    st.header("🔎 Critères de recherche")

    skill_search = st.text_input("Recherche par mots-clés", 
                               placeholder="ex: python, java, .net, machine learning",
                               help="Saisissez des termes techniques séparés par des virgules. Les variantes (comme '.net', 'dotnet') seront automatiquement détectées.")
    
    if skill_search:
        selected_skills = [s.strip() for s in skill_search.split(',') if s.strip()]
        # Afficher les termes normalisés pour feedback utilisateur
        normalized_skills = [normalize_tech_term(s) for s in selected_skills]
        if selected_skills != normalized_skills:
            st.caption(f"Termes recherchés (normalisés): {', '.join(normalized_skills)}")
    else:
        selected_skills = []

    st.subheader("Niveau d'expérience")
    level_options = ['Tous'] + sorted(df['level_experience'].dropna().unique().tolist())
    selected_level = st.selectbox("Niveau", options=level_options, index=0)

    st.subheader("Intervalle d'expérience")
    exp_options = ['Tous'] + sorted(df['inter_exp'].dropna().unique().tolist())
    selected_exp = st.selectbox("Intervalle", options=exp_options, index=0)

    st.subheader("Disponibilité")
    dispo_options = ['Tous'] + sorted(df['disponibilite'].dropna().unique().tolist())
    selected_dispo = st.selectbox("Disponibilité", options=dispo_options, index=0)

    st.subheader("Années d'expérience")
    min_exp, max_exp = int(df['annees_experience'].min()), int(df['annees_experience'].max())
    exp_range = st.slider("Plage d'années d'expérience", min_value=min_exp, max_value=max_exp, value=(min_exp, max_exp))

    st.subheader("Localisation")
    selected_region = st.multiselect("Région", sorted(df["region"].dropna().unique()))
    selected_department = st.multiselect("Département", sorted(df["department"].dropna().unique()))
    selected_ville = st.multiselect("Ville", sorted(df['localisation'].dropna().unique()))

# -------- Filtrage des données --------
df_filtered = df.copy()

if selected_dispo != 'Tous':
    df_filtered = df_filtered[df_filtered['disponibilite'] == selected_dispo]

if selected_level != 'Tous':
    df_filtered = df_filtered[df_filtered['level_experience'] == selected_level]

if selected_exp != 'Tous':
    df_filtered = df_filtered[df_filtered['inter_exp'] == selected_exp]

df_filtered = df_filtered[
    (df_filtered['annees_experience'] >= exp_range[0]) &
    (df_filtered['annees_experience'] <= exp_range[1])
]

if selected_skills:
    df_filtered = filter_by_tech(df_filtered, selected_skills)

if selected_region:
    df_filtered = df_filtered[df_filtered["region"].isin(selected_region)]
if selected_department:
    df_filtered = df_filtered[df_filtered["department"].isin(selected_department)]
if selected_ville:
    df_filtered = df_filtered[df_filtered["localisation"].isin(selected_ville)]

# -------- Affichage des résultats --------
st.header("📊 Résultats")

if not df_filtered.empty:
    cols = st.columns(4)
    cols[0].metric("Profils trouvés", len(df_filtered))
    if selected_skills:
        normalized_skills = [normalize_tech_term(s) for s in selected_skills]
        cols[1].metric("Mots-clés", ", ".join(normalized_skills[:3]) + ("..." if len(normalized_skills) > 3 else ""))
    if selected_level != 'Tous':
        cols[2].metric("Niveau d'expérience", selected_level)
    if selected_exp != 'Tous':
        cols[3].metric("Intervalle d'expérience", selected_exp)

    st.subheader("📍 Répartition géographique")
    tab1, tab2, tab3 = st.tabs(["Par région", "Par département", "Par ville"])

    with tab1:
        region_counts = df_filtered['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        fig = px.bar(region_counts, x='region', y='count', title="Nombre de profils par région")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dept_counts = df_filtered['department'].value_counts().reset_index()
        dept_counts.columns = ['department', 'count']
        fig = px.bar(dept_counts, x='department', y='count', title="Nombre de profils par département")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        ville_counts = df_filtered['localisation'].value_counts().reset_index()
        ville_counts.columns = ['localisation', 'count']
        fig = px.bar(ville_counts, x='localisation', y='count', title="Nombre de profils par ville")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Répartition par expérience")
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        fig = px.pie(df_filtered, names='inter_exp', title="Par intervalle d'expérience")
        st.plotly_chart(fig, use_container_width=True)

    with exp_col2:
        fig = px.pie(df_filtered, names='level_experience', title="Par niveau d'expérience")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧑‍💻 Liste des profils")
    st.dataframe(
        df_filtered[['nom', 'poste', 'competences', 'level_experience', 'inter_exp', 'localisation']],
        height=600,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="💾 Exporter les résultats (CSV)",
        data=df_filtered.to_csv(index=False).encode('utf-8'),
        file_name='profils_recherche.csv',
        mime='text/csv',
        use_container_width=True
    )
else:
    st.warning("Aucun profil ne correspond aux critères de recherche.")