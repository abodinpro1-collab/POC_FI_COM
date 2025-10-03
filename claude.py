import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
import os
import time
import uuid
import base64
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Finances Locales - Analyse Départementale",
    page_icon="📊",
    layout="wide"
)

API_URL = "https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/comptes-individuels-des-communes-fichier-global-a-compter-de-2000/records"

st.title("📊 Analyse de la santé financière des communes")
st.markdown("---")

# Sidebar pour les contrôles
st.sidebar.header("🔧 Paramètres d'analyse")

# --- Liste des départements ---
departements_dispo = [f"{i:03d}" for i in range(1, 101)] + ["2A", "2B"]
dept_selection = st.sidebar.selectbox("Département", departements_dispo)

# --- Année ---
annees_dispo = [2023, 2022, 2021]
annee_selection = st.sidebar.selectbox("Année", annees_dispo)

# --- Filtres additionnels ---
st.sidebar.subheader("🔍 Filtres")
taille_min = st.sidebar.number_input("Population minimale", min_value=0, value=0)

# --- Fonction pour récupérer toutes les communes avec gestion d'erreur ---
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def fetch_communes(dep, an):
    """Récupère les données financières des communes avec gestion d'erreur"""
    try:
        dep = str(dep).zfill(3)
        dfs = []
        limit = 100
        offset = 1

        with st.spinner(f"Récupération des données pour {dep}..."):
            while True:
                params = {
                    "where": f'dep="{dep}" AND an="{an}"',
                    "limit": limit,
                    "offset": offset
                }
                
                response = requests.get(API_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "results" not in data or not data["results"]:
                    break

                rows = []
                for r in data["results"]:
                    record = r.get("record", r)
                    rows.append({
                        "Commune": record.get("inom"),
                        "Année": record.get("an"),
                        "Population": record.get("pop1"),
                        "RRF (K€)": record.get("fprod"),
                        "RRF - Moy. strate (K€)": record.get("mprod"),
                        "DRF (K€)": record.get("fcharge"),
                        "DRF - Moy. strate (K€)": record.get("mcharge"),
                        "Encours (K€)": record.get("fdette"),
                        "Encours - Moy. strate (K€)": record.get("mdette"),
                        "Annuité (K€)": record.get("fannu"),
                        "Annuité - Moy. strate (K€)": record.get("mannu"),
                        "Département": record.get("dep"),
                        "Épargne brute (K€)": (record.get("fprod") or 0) - (record.get("fcharge") or 0),
                        "Épargne brute - Moy. strate (K€)": (record.get("mprod") or 0) - (record.get("mcharge") or 0)
                    })

                df = pd.DataFrame(rows)
                dfs.append(df)

                if len(data["results"]) < limit:
                    break
                offset += limit

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    except requests.RequestException as e:
        st.error(f"❌ Erreur de connexion à l'API : {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement des données : {e}")
        return pd.DataFrame()

# --- Fonction pour récupérer les données historiques d'une commune ---
@st.cache_data(ttl=3600)
def fetch_historical_commune_data(commune_name, dep, years_range=[2019, 2020, 2021, 2022, 2023]):
    """Récupère les données historiques d'une commune spécifique"""
    historical_data = []
    
    for year in years_range:
        try:
            df_year = fetch_communes(dep, year)
            if not df_year.empty:
                commune_data = df_year[df_year['Commune'] == commune_name]
                if not commune_data.empty:
                    historical_data.append(commune_data.iloc[0])
        except Exception as e:
            st.warning(f"Données non disponibles pour {commune_name} en {year}")
            continue
    
    if historical_data:
        df_historical = pd.DataFrame(historical_data)
        return df_historical
    return pd.DataFrame()

# --- Fonction pour calculer les KPI historiques ---
def calculate_historical_kpis(df_historical):
    """Calcule les KPI historiques pour la commune et sa strate"""
    if df_historical.empty:
        return pd.DataFrame()
    
    df_kpi_hist = df_historical.copy()
    
    # KPI Commune
    df_kpi_hist["TEB Commune (%)"] = df_kpi_hist["Épargne brute (K€)"] / df_kpi_hist["RRF (K€)"].replace(0, pd.NA) * 100
    df_kpi_hist["CD Commune (années)"] = df_kpi_hist["Encours (K€)"] / df_kpi_hist["Épargne brute (K€)"].replace(0, pd.NA)
    df_kpi_hist["Annuité/RRF Commune (%)"] = df_kpi_hist["Annuité (K€)"] / df_kpi_hist["RRF (K€)"].replace(0, pd.NA) * 100
    df_kpi_hist["Encours/hab Commune (€)"] = df_kpi_hist["Encours (K€)"] * 1000 / df_kpi_hist["Population"].replace(0, pd.NA)
    
    # KPI Strate (moyennes officielles)
    df_kpi_hist["TEB Strate (%)"] = df_kpi_hist["Épargne brute - Moy. strate (K€)"] / df_kpi_hist["RRF - Moy. strate (K€)"].replace(0, pd.NA) * 100
    df_kpi_hist["CD Strate (années)"] = df_kpi_hist["Encours - Moy. strate (K€)"] / df_kpi_hist["Épargne brute - Moy. strate (K€)"].replace(0, pd.NA)
    df_kpi_hist["Annuité/RRF Strate (%)"] = df_kpi_hist["Annuité - Moy. strate (K€)"] / df_kpi_hist["RRF - Moy. strate (K€)"].replace(0, pd.NA) * 100
    
    # Pour Encours/hab strate, on estime avec la population moyenne (approximation)
    pop_moyenne_strate = df_kpi_hist["Population"].mean()  # Approximation
    df_kpi_hist["Encours/hab Strate (€)"] = df_kpi_hist["Encours - Moy. strate (K€)"] * 1000 / pop_moyenne_strate
    
    return df_kpi_hist

# --- Fonction pour créer les graphiques d'évolution ---
def create_evolution_charts(df_historical_kpi, commune_name):
    """Crée les graphiques d'évolution des KPI"""
    if df_historical_kpi.empty:
        return None, None, None, None
    
    # Graphique 1: Évolution TEB
    fig_teb = go.Figure()
    fig_teb.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['TEB Commune (%)'],
        mode='lines+markers',
        name=f'{commune_name}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_teb.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['TEB Strate (%)'],
        mode='lines+markers',
        name='Moyenne strate',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_teb.add_hline(y=8, line_dash="dot", line_color="green", annotation_text="Seuil bon (8%)")
    fig_teb.add_hline(y=5, line_dash="dot", line_color="orange", annotation_text="Seuil critique (5%)")
    fig_teb.update_layout(
        title="📈 Évolution du Taux d'Épargne Brute",
        xaxis_title="Année",
        yaxis_title="TEB (%)",
        hovermode='x unified'
    )
    
    # Graphique 2: Évolution Capacité de désendettement
    fig_cd = go.Figure()
    fig_cd.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['CD Commune (années)'],
        mode='lines+markers',
        name=f'{commune_name}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_cd.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['CD Strate (années)'],
        mode='lines+markers',
        name='Moyenne strate',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_cd.add_hline(y=8, line_dash="dot", line_color="green", annotation_text="Seuil bon (8 ans)")
    fig_cd.add_hline(y=12, line_dash="dot", line_color="red", annotation_text="Seuil critique (12 ans)")
    fig_cd.update_layout(
        title="⏳ Évolution de la Capacité de Désendettement",
        xaxis_title="Année",
        yaxis_title="Capacité (années)",
        hovermode='x unified'
    )
    
    # Graphique 3: Évolution Ratio d'annuité
    fig_annuite = go.Figure()
    fig_annuite.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Annuité/RRF Commune (%)'],
        mode='lines+markers',
        name=f'{commune_name}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_annuite.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Annuité/RRF Strate (%)'],
        mode='lines+markers',
        name='Moyenne strate',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_annuite.add_hline(y=12, line_dash="dot", line_color="orange", annotation_text="Seuil surveillance (12%)")
    fig_annuite.add_hline(y=18, line_dash="dot", line_color="red", annotation_text="Seuil critique (18%)")
    fig_annuite.update_layout(
        title="💳 Évolution du Ratio d'Annuité",
        xaxis_title="Année",
        yaxis_title="Annuité/RRF (%)",
        hovermode='x unified'
    )
    
    # Graphique 4: Évolution Endettement par habitant
    fig_endett = go.Figure()
    fig_endett.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Encours/hab Commune (€)'],
        mode='lines+markers',
        name=f'{commune_name}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_endett.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Encours/hab Strate (€)'],
        mode='lines+markers',
        name='Moyenne strate',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_endett.update_layout(
        title="👥 Évolution de l'Endettement par Habitant",
        xaxis_title="Année",
        yaxis_title="Endettement (€/hab)",
        hovermode='x unified'
    )
    
    return fig_teb, fig_cd, fig_annuite, fig_endett
def score_sante_financiere(row, df_ref):
    """Calcule le score de santé financière (0-100, plus c'est haut mieux c'est)"""
    score = 100  # On part de 100 et on retire des points

    # Capacité de désendettement
    if pd.isna(row['CD (années)']) or row['CD (années)'] <= 0:
        score -= 40  # Très mauvais
    elif row['CD (années)'] > 12:
        score -= 30  # Mauvais
    elif row['CD (années)'] > 8:
        score -= 15  # Moyen

    # Taux d'épargne brute
    if pd.isna(row['TEB (%)']) or row['TEB (%)'] < 0:
        score -= 25  # Très mauvais
    elif row['TEB (%)'] < 5:
        score -= 25  # Critique
    elif row['TEB (%)'] < 8:
        score -= 12.5  # Acceptable

    # Annuité / RRF
    if pd.notna(row['Annuité / RRF (%)']):
        if row['Annuité / RRF (%)'] > 18:
            score -= 20  # Très élevé
        elif row['Annuité / RRF (%)'] > 12:
            score -= 10  # Élevé

    # Encours / hab
    if pd.notna(row['Encours / hab (€/hab)']):
        if row['Encours / hab (€/hab)'] > df_ref['Encours / hab (€/hab)'].quantile(0.8):
            score -= 25  # Très élevé
        elif row['Encours / hab (€/hab)'] > df_ref['Encours / hab (€/hab)'].quantile(0.6):
            score -= 12.5  # Élevé

    return max(0, score)  # Score ne peut pas être négatif

def niveau_alerte(score):
    """Détermine le niveau d'alerte (score inversé : haut = bon)"""
    if score >= 70:
        return "🟢 Vert"
    elif score >= 50:
        return "🟠 Orange"
    else:
        return "🔴 Rouge"

def get_color_alerte(niveau):
    """Retourne la couleur correspondant au niveau"""
    if "Rouge" in niveau:
        return "#FF4B4B"
    elif "Orange" in niveau:
        return "#FF8C00"
    else:
        return "#00C851"

# --- Fonction pour créer les tranches de population ---
def create_population_brackets(df):
    """Crée des tranches de population"""
    df['Tranche pop'] = pd.cut(df['Population'], 
                               bins=[0, 500, 2000, 10000, float('inf')],
                               labels=['< 500 hab', '500-2000 hab', '2000-10000 hab', '> 10000 hab'])
    return df

# --- Fonction d'export Excel ---
def create_excel_export(df_kpi):
    """Crée un fichier Excel à télécharger - Solution robuste Windows"""
    try:
        import time
        import uuid
        
        # Nom de fichier unique pour éviter les conflits
        unique_name = f"analyse_{uuid.uuid4().hex[:8]}.xlsx"
        temp_path = os.path.join(tempfile.gettempdir(), unique_name)
        
        # Création du fichier Excel (syntaxe compatible toutes versions pandas)
        with pd.ExcelWriter(temp_path, engine='xlsxwriter') as writer:
            # Feuille principale
            df_kpi.to_excel(writer, sheet_name='Analyse_KPI', index=False)
            
            # Feuille synthèse
            synthese = df_kpi.groupby('Niveau d\'alerte').agg({
                'Commune': 'count',
                'Population': 'sum',
                'Score': 'mean'
            }).round(2)
            synthese.to_excel(writer, sheet_name='Synthese')
            
            # Formatage des feuilles (si xlsxwriter disponible)
            try:
                workbook = writer.book
                
                # Format pour les headers
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Format pour les scores (codes couleurs)
                green_format = workbook.add_format({'bg_color': '#C6EFCE'})
                orange_format = workbook.add_format({'bg_color': '#FFEB9C'})
                red_format = workbook.add_format({'bg_color': '#FFC7CE'})
                
                # Formatage de la feuille principale
                worksheet = writer.sheets['Analyse_KPI']
                
                # Headers
                for col_num, value in enumerate(df_kpi.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                    # Largeurs de colonnes adaptées
                    if 'Commune' in str(value):
                        worksheet.set_column(col_num, col_num, 20)
                    elif 'Population' in str(value):
                        worksheet.set_column(col_num, col_num, 12)
                    elif 'Score' in str(value):
                        worksheet.set_column(col_num, col_num, 10)
                    else:
                        worksheet.set_column(col_num, col_num, 15)
                
                # Formatage conditionnel pour les niveaux d'alerte
                if 'Niveau d\'alerte' in df_kpi.columns:
                    alert_col = df_kpi.columns.get_loc('Niveau d\'alerte')
                    for row_num, alert_level in enumerate(df_kpi['Niveau d\'alerte'], 1):
                        if 'Vert' in str(alert_level):
                            worksheet.write(row_num, alert_col, alert_level, green_format)
                        elif 'Orange' in str(alert_level):
                            worksheet.write(row_num, alert_col, alert_level, orange_format)
                        elif 'Rouge' in str(alert_level):
                            worksheet.write(row_num, alert_col, alert_level, red_format)
            except:
                # Si le formatage échoue, on continue sans formatage
                pass
        
        # Attendre que le fichier soit complètement écrit
        time.sleep(0.2)
        
        # Lecture sécurisée du fichier
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                with open(temp_path, 'rb') as file:
                    excel_data = file.read()
                break
            except (PermissionError, FileNotFoundError) as e:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(0.2)  # Pause plus longue entre les tentatives
        
        # Nettoyage sécurisé
        try:
            if os.path.exists(temp_path):
                time.sleep(0.1)  # Petite pause avant suppression
                os.remove(temp_path)
        except (PermissionError, FileNotFoundError, OSError):
            # Si impossible à supprimer, ce n'est pas grave
            pass
        
        return excel_data
    
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier Excel : {e}")
        # Fallback : Export CSV enrichi si Excel échoue
        try:
            # CSV avec séparateur français et encodage compatible
            csv_data = df_kpi.to_csv(
                index=False, 
                sep=';',  # Séparateur français
                encoding='utf-8-sig',  # BOM pour Excel français
                decimal=',',  # Décimales françaises
                float_format='%.2f'  # 2 décimales
            )
            st.warning("⚠️ Export Excel échoué, fichier CSV généré à la place")
            return csv_data.encode('utf-8-sig')
        except Exception as csv_error:
            st.error(f"Erreur également sur l'export CSV : {csv_error}")
            return None

# --- Récupération et traitement des données ---
df_dept = fetch_communes(dept_selection, annee_selection)

if df_dept.empty:
    st.warning(f"❌ Aucune donnée disponible pour le département {dept_selection} en {annee_selection}.")
else:
    # Filtrage par taille
    if taille_min > 0:
        df_dept = df_dept[df_dept['Population'] >= taille_min]
    
    if df_dept.empty:
        st.warning("❌ Aucune commune ne correspond aux critères de filtrage.")
    else:
        # --- Calculs KPI ---
        df_kpi = df_dept.copy()
        df_kpi["TEB (%)"] = df_kpi["Épargne brute (K€)"] / df_kpi["RRF (K€)"].replace(0, pd.NA) * 100
        df_kpi["CD (années)"] = df_kpi["Encours (K€)"] / df_kpi["Épargne brute (K€)"].replace(0, pd.NA)
        df_kpi["Annuité / RRF (%)"] = df_kpi["Annuité (K€)"] / df_kpi["RRF (K€)"].replace(0, pd.NA) * 100
        df_kpi["Encours / hab (€/hab)"] = df_kpi["Encours (K€)"] * 1000 / df_kpi["Population"].replace(0, pd.NA)
        df_kpi["Rigidité (%)"] = (df_kpi["DRF (K€)"] / df_kpi["RRF (K€)"].replace(0, pd.NA) * 100)
        
        # Calcul des scores
        df_kpi['Score'] = df_kpi.apply(score_sante_financiere, axis=1, df_ref=df_kpi)
        df_kpi['Niveau d\'alerte'] = df_kpi['Score'].apply(niveau_alerte)
        
        # Création des tranches de population
        df_kpi = create_population_brackets(df_kpi)
        
        # Filtre par niveau d'alerte
        niveaux_dispo = df_kpi['Niveau d\'alerte'].unique()
        niveau_filtre = st.sidebar.multiselect("Niveau d'alerte", niveaux_dispo, default=niveaux_dispo)
        df_filtered = df_kpi[df_kpi['Niveau d\'alerte'].isin(niveau_filtre)]
        
        # === TABLEAU DE BORD PRINCIPAL ===
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📍 Communes analysées", len(df_filtered))
        
        with col2:
            score_moyen = df_filtered['Score'].mean()
            st.metric("📊 Score moyen de santé", f"{score_moyen:.1f}/100")
        
        with col3:
            pop_totale = df_filtered['Population'].sum()
            st.metric("👥 Population totale", f"{pop_totale:,}")
        
        with col4:
            pct_rouge = (df_filtered['Niveau d\'alerte'].str.contains('Rouge').sum() / len(df_filtered) * 100)
            st.metric("🚨 % Communes fragiles", f"{pct_rouge:.1f}%")
        
        st.markdown("---")
        
        # === GRAPHIQUES ===
        
        # Ligne 1 : Vue d'ensemble
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des niveaux d'alerte
            alert_counts = df_filtered['Niveau d\'alerte'].value_counts()
            colors = [get_color_alerte(niveau) for niveau in alert_counts.index]
            
            fig_pie = px.pie(values=alert_counts.values, names=alert_counts.index,
                            title="🎯 Répartition des niveaux d'alerte",
                            color_discrete_sequence=colors)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Histogramme des scores
            fig_hist = px.histogram(df_filtered, x='Score', nbins=15,
                                   title="📈 Distribution des scores de santé financière",
                                   labels={'Score': 'Score de santé', 'count': 'Nombre de communes'})
            fig_hist.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Seuil Orange")
            fig_hist.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Seuil Vert")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Ligne 2 : Analyse comparative
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot TEB vs CD
            fig_scatter = px.scatter(df_filtered, x='TEB (%)', y='CD (années)',
                                   color='Niveau d\'alerte', size='Population',
                                   hover_data=['Commune', 'Score'],
                                   title="💰 Taux d'épargne vs Capacité de désendettement",
                                   color_discrete_map={
                                       "🟢 Vert": "#00C851",
                                       "🟠 Orange": "#FF8C00", 
                                       "🔴 Rouge": "#FF4B4B"
                                   })
            fig_scatter.add_hline(y=12, line_dash="dash", line_color="red", annotation_text="Seuil critique CD")
            fig_scatter.add_vline(x=5, line_dash="dash", line_color="orange", annotation_text="Seuil TEB")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Box plot TEB par niveau
            fig_box = px.box(df_filtered, x='Niveau d\'alerte', y='TEB (%)',
                           title="📊 Distribution du TEB par niveau d'alerte",
                           color='Niveau d\'alerte',
                           color_discrete_map={
                               "🟢 Vert": "#00C851",
                               "🟠 Orange": "#FF8C00", 
                               "🔴 Rouge": "#FF4B4B"
                           })
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Ligne 3 : Analyse par taille
        col1, col2 = st.columns(2)
        
        with col1:
            # Score moyen par tranche de population
            score_by_size = df_filtered.groupby('Tranche pop')['Score'].mean().reset_index()
            fig_bar = px.bar(score_by_size, x='Tranche pop', y='Score',
                           title="📏 Score moyen par taille de commune",
                           labels={'Score': 'Score moyen', 'Tranche pop': 'Taille de commune'})
            fig_bar.add_hline(y=50, line_dash="dash", line_color="orange")
            fig_bar.add_hline(y=70, line_dash="dash", line_color="red")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Endettement par habitant vs Population
            fig_debt = px.scatter(df_filtered, x='Population', y='Encours / hab (€/hab)',
                                color='Niveau d\'alerte', 
                                title="💳 Endettement par habitant vs Population",
                                color_discrete_map={
                                    "🟢 Vert": "#00C851",
                                    "🟠 Orange": "#FF8C00", 
                                    "🔴 Rouge": "#FF4B4B"
                                },
                                hover_data=['Commune'])
            st.plotly_chart(fig_debt, use_container_width=True)
        
        # === TABLEAUX TOP/FLOP ===
        st.markdown("---")
        st.subheader("🏆 Classements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Top 10 - Communes les plus fragiles")
            top_risk = df_filtered.nsmallest(10, 'Score')[['Commune', 'Population', 'Score', 'TEB (%)', 'CD (années)']]
            st.dataframe(top_risk, use_container_width=True)
        
        with col2:
            st.markdown("#### 🟢 Top 10 - Communes les plus solides")
            top_solid = df_filtered.nlargest(10, 'Score')[['Commune', 'Population', 'Score', 'TEB (%)', 'CD (années)']]
            st.dataframe(top_solid, use_container_width=True)
        
        # === ANALYSE DÉTAILLÉE D'UNE COMMUNE ===
        st.markdown("---")
        st.subheader("🔍 Analyse détaillée d'une commune")
        
        commune_selectionnee = st.selectbox("Choisir une commune", df_filtered['Commune'].sort_values())
        
        if commune_selectionnee:
            commune_data = df_filtered[df_filtered['Commune'] == commune_selectionnee].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Commune :** {commune_data['Commune']}")
                st.markdown(f"**Population :** {commune_data['Population']:,} habitants")
                st.markdown(f"**Score de santé :** {commune_data['Score']:.1f}/100")
                st.markdown(f"**Niveau d'alerte :** {commune_data['Niveau d\'alerte']}")
            
            with col2:
                # Radar chart avec comparaison commune vs strate officielle
                categories = ['TEB', 'CD inversée', 'Rigidité inv.', 'Endettement/hab inv.', 'Annuité inv.']
                
                # Normalisation des valeurs COMMUNE (0-100, plus c'est haut mieux c'est)
                teb_norm = max(0, min(100, commune_data['TEB (%)'] * 10))
                cd_norm = max(0, min(100, 100 - commune_data['CD (années)'] * 5))
                rigidite_norm = max(0, min(100, 200 - commune_data['Rigidité (%)']))
                endett_norm = max(0, min(100, 100 - (commune_data['Encours / hab (€/hab)'] / 50)))
                annuite_norm = max(0, min(100, 100 - commune_data['Annuité / RRF (%)'] * 5))
                
                # Calcul des KPI de la STRATE OFFICIELLE (données API)
                epargne_strate = commune_data['Épargne brute - Moy. strate (K€)']
                rrf_strate = commune_data['RRF - Moy. strate (K€)']
                drf_strate = commune_data['DRF - Moy. strate (K€)']
                encours_strate = commune_data['Encours - Moy. strate (K€)']
                annuite_strate_val = commune_data['Annuité - Moy. strate (K€)']
                
                # Calcul des ratios STRATE
                teb_strate = (epargne_strate / rrf_strate * 100) if pd.notna(rrf_strate) and rrf_strate != 0 else 0
                cd_strate = (encours_strate / epargne_strate) if pd.notna(epargne_strate) and epargne_strate != 0 else 0
                rigidite_strate = (drf_strate / rrf_strate * 100) if pd.notna(rrf_strate) and rrf_strate != 0 else 0
                annuite_rrf_strate = (annuite_strate_val / rrf_strate * 100) if pd.notna(rrf_strate) and rrf_strate != 0 else 0
                # Pour l'endettement/hab de la strate, on utilise une approximation avec la pop moyenne
                endett_strate = (encours_strate * 1000 / commune_data['Population']) if pd.notna(commune_data['Population']) and commune_data['Population'] != 0 else 0
                
                # Normalisation des valeurs STRATE (même logique)
                teb_strate_norm = max(0, min(100, teb_strate * 10))
                cd_strate_norm = max(0, min(100, 100 - cd_strate * 5))
                rigidite_strate_norm = max(0, min(100, 200 - rigidite_strate))
                endett_strate_norm = max(0, min(100, 100 - (endett_strate / 50)))
                annuite_strate_norm = max(0, min(100, 100 - annuite_rrf_strate * 5))
                
                fig_radar = go.Figure()
                
                # Trace de la commune
                fig_radar.add_trace(go.Scatterpolar(
                    r=[teb_norm, cd_norm, rigidite_norm, endett_norm, annuite_norm],
                    theta=categories,
                    fill='toself',
                    name=commune_data['Commune'],
                    line=dict(color=get_color_alerte(commune_data['Niveau d\'alerte']), width=3),
                    marker=dict(size=8)
                ))
                
                # Trace de la strate officielle
                fig_radar.add_trace(go.Scatterpolar(
                    r=[teb_strate_norm, cd_strate_norm, rigidite_strate_norm, endett_strate_norm, annuite_strate_norm],
                    theta=categories,
                    fill='toself',
                    name='Moyenne Strate Officielle',
                    line=dict(color='#FFA500', width=2, dash='dash'),
                    opacity=0.5
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            showticklabels=True,
                            ticks='outside'
                        )),
                    showlegend=True,
                    title="Profil financier : Commune vs Strate Officielle",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Ajout d'un indicateur de comparaison textuel
                st.markdown("**🎯 Analyse comparative vs strate officielle :**")
                
                comparaisons = []
                if teb_norm > teb_strate_norm + 10:
                    comparaisons.append(f"✅ TEB supérieur à la strate ({commune_data['TEB (%)']:.1f}% vs {teb_strate:.1f}%)")
                elif teb_norm < teb_strate_norm - 10:
                    comparaisons.append(f"⚠️ TEB inférieur à la strate ({commune_data['TEB (%)']:.1f}% vs {teb_strate:.1f}%)")
                
                if cd_norm > cd_strate_norm + 10:
                    comparaisons.append(f"✅ Endettement mieux maîtrisé que la strate ({commune_data['CD (années)']:.1f} ans vs {cd_strate:.1f} ans)")
                elif cd_norm < cd_strate_norm - 10:
                    comparaisons.append(f"⚠️ Endettement plus élevé que la strate ({commune_data['CD (années)']:.1f} ans vs {cd_strate:.1f} ans)")
                
                if rigidite_norm > rigidite_strate_norm + 10:
                    comparaisons.append(f"✅ Plus de flexibilité budgétaire que la strate ({commune_data['Rigidité (%)']:.1f}% vs {rigidite_strate:.1f}%)")
                elif rigidite_norm < rigidite_strate_norm - 10:
                    comparaisons.append(f"⚠️ Moins de flexibilité que la strate ({commune_data['Rigidité (%)']:.1f}% vs {rigidite_strate:.1f}%)")
                
                if comparaisons:
                    for comp in comparaisons:
                        st.markdown(f"- {comp}")
                else:
                    st.markdown("- 📊 Performance globalement dans la moyenne de la strate officielle")
            
            # === ANALYSE PLURIANNUELLE ===
            st.markdown("---")
            st.subheader(f"📊 Évolution pluriannuelle : {commune_selectionnee}")
            st.markdown("*Comparaison avec la moyenne de la strate officielle (2019-2023)*")
            
            # Récupération des données historiques
            with st.spinner("Chargement des données historiques..."):
                df_historical = fetch_historical_commune_data(commune_selectionnee, dept_selection)
            
            if not df_historical.empty and len(df_historical) > 1:
                # Calcul des KPI historiques
                df_historical_kpi = calculate_historical_kpis(df_historical)
                
                # Affichage des métriques d'évolution
                col1, col2, col3, col4 = st.columns(4)
                
                if len(df_historical_kpi) >= 2:
                    # Calcul des évolutions (dernière année vs première année disponible)
                    evolution_teb = df_historical_kpi.iloc[-1]['TEB Commune (%)'] - df_historical_kpi.iloc[0]['TEB Commune (%)']
                    evolution_cd = df_historical_kpi.iloc[-1]['CD Commune (années)'] - df_historical_kpi.iloc[0]['CD Commune (années)']
                    evolution_annuite = df_historical_kpi.iloc[-1]['Annuité/RRF Commune (%)'] - df_historical_kpi.iloc[0]['Annuité/RRF Commune (%)']
                    evolution_endett = df_historical_kpi.iloc[-1]['Encours/hab Commune (€)'] - df_historical_kpi.iloc[0]['Encours/hab Commune (€)']
                    
                    with col1:
                        delta_color = "normal" if evolution_teb >= 0 else "inverse"
                        st.metric("📈 Évolution TEB", f"{evolution_teb:+.1f}%", delta=f"{evolution_teb:+.1f}pp", delta_color=delta_color)
                    
                    with col2:
                        delta_color = "inverse" if evolution_cd >= 0 else "normal"
                        st.metric("⏳ Évolution CD", f"{evolution_cd:+.1f} ans", delta=f"{evolution_cd:+.1f} ans", delta_color=delta_color)
                    
                    with col3:
                        delta_color = "inverse" if evolution_annuite >= 0 else "normal"
                        st.metric("💳 Évolution Annuité/RRF", f"{evolution_annuite:+.1f}%", delta=f"{evolution_annuite:+.1f}pp", delta_color=delta_color)
                    
                    with col4:
                        delta_color = "inverse" if evolution_endett >= 0 else "normal"
                        st.metric("👥 Évolution Endett/hab", f"{evolution_endett:+.0f}€", delta=f"{evolution_endett:+.0f}€", delta_color=delta_color)
                
                # Création des graphiques d'évolution
                fig_teb, fig_cd, fig_annuite, fig_endett = create_evolution_charts(df_historical_kpi, commune_selectionnee)
                
                # Affichage des graphiques d'évolution
                col1, col2 = st.columns(2)
                
                with col1:
                    if fig_teb:
                        st.plotly_chart(fig_teb, use_container_width=True)
                    if fig_annuite:
                        st.plotly_chart(fig_annuite, use_container_width=True)
                
                with col2:
                    if fig_cd:
                        st.plotly_chart(fig_cd, use_container_width=True)
                    if fig_endett:
                        st.plotly_chart(fig_endett, use_container_width=True)
                
                # Tableau récapitulatif de l'évolution
                st.subheader("📋 Tableau récapitulatif pluriannuel")
                
                # Sélection des colonnes pertinentes pour l'affichage
                colonnes_evolution = [
                    'Année', 'Population', 
                    'TEB Commune (%)', 'TEB Strate (%)',
                    'CD Commune (années)', 'CD Strate (années)', 
                    'Annuité/RRF Commune (%)', 'Annuité/RRF Strate (%)',
                    'Encours/hab Commune (€)', 'Encours/hab Strate (€)'
                ]
                
                # Formatage du tableau
                df_display = df_historical_kpi[colonnes_evolution].round(2)
                
                # Style conditionnel pour mettre en évidence les évolutions
                def highlight_evolution(s):
                    if s.name in ['TEB Commune (%)', 'TEB Strate (%)']:
                        return ['background-color: lightgreen' if x >= 8 else 'background-color: lightcoral' if x < 5 else '' for x in s]
                    elif s.name in ['CD Commune (années)', 'CD Strate (années)']:
                        return ['background-color: lightcoral' if x > 12 else 'background-color: lightyellow' if x > 8 else 'background-color: lightgreen' for x in s]
                    return ['' for x in s]
                
                styled_evolution = df_display.style.apply(highlight_evolution)
                st.dataframe(styled_evolution, use_container_width=True)
                
            else:
                st.warning(f"⚠️ Données historiques insuffisantes pour {commune_selectionnee} (moins de 2 années disponibles)")
                st.info("💡 L'analyse pluriannuelle nécessite au moins 2 années de données consécutives")
        
        # === TABLEAUX DÉTAILLÉS ===
        st.markdown("---")
        
        # Onglets pour les différents tableaux
        tab1, tab2 = st.tabs(["📊 Tableau KPI complet", "📋 Données brutes"])
        
        with tab1:
            colonnes_kpi = [
                "Commune", "Population", "TEB (%)", "CD (années)", 
                "Annuité / RRF (%)", "Encours / hab (€/hab)", "Rigidité (%)",
                "Score", "Niveau d'alerte"
            ]
            
            # Formatage conditionnel
            def color_niveau(val):
                if "Rouge" in str(val):
                    return 'background-color: #FFE6E6'
                elif "Orange" in str(val):
                    return 'background-color: #FFF4E6'
                else:
                    return 'background-color: #E6F7E6'
            
            styled_df = df_filtered[colonnes_kpi].style.applymap(color_niveau, subset=['Niveau d\'alerte'])
            st.dataframe(styled_df, use_container_width=True)
        
        with tab2:
            st.dataframe(df_filtered, use_container_width=True)
        
        # === EXPORT ===
        st.markdown("---")
        st.subheader("💾 Export des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export Excel
            excel_data = create_excel_export(df_filtered)
            if excel_data:
                file_extension = ".xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                # Vérifier si c'est du CSV (fallback)
                try:
                    if excel_data.decode('utf-8-sig').startswith('Commune') or excel_data.decode('utf-8').startswith('Commune'):
                        file_extension = ".csv"
                        mime_type = "text/csv"
                except:
                    pass  # Garder Excel par défaut
                
                st.download_button(
                    label=f"📥 Télécharger {'Excel' if file_extension == '.xlsx' else 'CSV'}",
                    data=excel_data,
                    file_name=f"analyse_finances_{dept_selection}_{annee_selection}{file_extension}",
                    mime=mime_type
                )
            else:
                st.error("Impossible de créer le fichier d'export")
        
        with col2:
            # Export CSV
            csv_data = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"analyse_finances_{dept_selection}_{annee_selection}.csv",
                mime="text/csv"
            )
        
        # === GÉNÉRATION PDF INDIVIDUELLE ===
        st.markdown("---")
        st.subheader("📄 Rapport PDF individuel")
        
        commune_pdf = st.selectbox(
            "Sélectionnez une commune pour générer son rapport PDF", 
            df_filtered['Commune'].sort_values(),
            key="pdf_commune_selector"
        )
        
        if commune_pdf:
            commune_pdf_data = df_filtered[df_filtered['Commune'] == commune_pdf].iloc[0]
            
            col_pdf1, col_pdf2 = st.columns([1, 2])
            
            with col_pdf1:
                st.markdown(f"**Commune sélectionnée :** {commune_pdf}")
                st.markdown(f"**Score :** {commune_pdf_data['Score']:.1f}/100")
                st.markdown(f"**Niveau :** {commune_pdf_data['Niveau d\'alerte']}")
            
            with col_pdf2:
                # Récupération des données historiques pour le PDF
                df_historical_pdf = fetch_historical_commune_data(commune_pdf, dept_selection)
                df_historical_kpi_pdf = None
                if not df_historical_pdf.empty and len(df_historical_pdf) > 1:
                    df_historical_kpi_pdf = calculate_historical_kpis(df_historical_pdf)
                    
        # === SYNTHÈSE ===
        st.markdown("---")
        st.subheader("📋 Synthèse départementale")
        
        synthese_col1, synthese_col2, synthese_col3 = st.columns(3)
        
        with synthese_col1:
            st.markdown("**🟢 Communes saines**")
            communes_vertes = len(df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Vert')])
            st.markdown(f"- Nombre : {communes_vertes}")
            st.markdown(f"- % : {communes_vertes/len(df_filtered)*100:.1f}%")
        
        with synthese_col2:
            st.markdown("**🟠 Communes sous surveillance**")
            communes_orange = len(df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Orange')])
            st.markdown(f"- Nombre : {communes_orange}")
            st.markdown(f"- % : {communes_orange/len(df_filtered)*100:.1f}%")
        
        with synthese_col3:
            st.markdown("**🔴 Communes à risque**")
            communes_rouges = len(df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Rouge')])
            st.markdown(f"- Nombre : {communes_rouges}")
            st.markdown(f"- % : {communes_rouges/len(df_filtered)*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("*Données : API des comptes individuels des communes - data.economie.gouv.fr*")