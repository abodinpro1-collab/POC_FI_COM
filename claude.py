# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tempfile
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()
import time
import uuid
import base64
import re
from functools import lru_cache
from difflib import SequenceMatcher
from datetime import datetime
from functools import lru_cache
from difflib import SequenceMatcher
from datetime import datetime
# ✅ IMPORT CORRIGÉ AVEC ALIAS
import plotly.io as pio
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
)
from reportlab.lib import colors as rl_colors  # ✅ ALIAS ICI
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from math import pi


# Configuration de la page
st.set_page_config(
    page_title="Finances Locales - Analyse Départementale",
    page_icon="📊",
    layout="wide"
)

# --- AUTHENTIFICATION ---
# Mot de passe stocké dans .env (fichier ignoré par git)
APP_PASSWORD = os.getenv("APP_PASSWORD")

def check_password():
    """Vérifie si l'utilisateur est authentifié"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Formulaire de connexion
    st.markdown("## 🔐 Connexion requise")

    with st.form("login_form"):
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")

        if submit:
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Mot de passe incorrect")

    return False

# Vérifier l'authentification avant d'afficher l'application
if not check_password():
    st.stop()

# Clear session state au démarrage pour éviter les conflits
if "initialized" not in st.session_state:
    st.session_state.initialized = True

# Mapping des années vers les nouveaux datasets
DATASETS_MAPPING = {
    2019: "comptes-individuels-des-communes-fichier-global-2019-2020",
    2020: "comptes-individuels-des-communes-fichier-global-2019-2020",
    2021: "comptes-individuels-des-communes-fichier-global-2021",
    2022: "comptes-individuels-des-communes-fichier-global-2022",
    2023: "comptes-individuels-des-communes-fichier-global-2023-2024",
    2024: "comptes-individuels-des-communes-fichier-global-2023-2024"
}

def get_dataset_for_year(annee):
    """Retourne le dataset approprié pour une année donnée"""
    return DATASETS_MAPPING.get(annee, "comptes-individuels-des-communes-fichier-global-2023-2024")

def get_api_url_for_year(annee):
    """Retourne l'URL de l'API pour une année donnée"""
    dataset = get_dataset_for_year(annee)
    return f"https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/{dataset}/records"

class RobustCommuneFetcher:
    """Fetcher robuste pour l'analyse départementale"""
    
    def __init__(self):
        self._cache = {}
    
    @lru_cache(maxsize=500)
    def normalize_commune_name(self, name):
        """Normalise un nom de commune"""
        if not name:
            return ""
        
        normalized = name.strip().upper()

        # Remplacer les tirets avant D' ou L' par des espaces
        normalized = re.sub(r"-([DL]')", r" \1", normalized)

        patterns = [
            # Gestion des articles LA/LE/LES en début
            (r'^(LA|LE|LES)\s+(.+)$', r'\2 (\1)'),
            # Gestion des articles LA/LE/LES déjà entre parenthèses (avec espaces optionnels)
            (r'^(.+)\s+\(\s*(LA|LE|LES)\s*\)$', r'\2 \1'),
            # Gestion de L'/D' en début (seulement si pas déjà dans le nom)
            (r"^(L'|D')(?!.*\s\1)(.+)$", r"\2 (\1)"),
            # Gestion de L'/D' déjà entre parenthèses
            (r"^(.+)\s*\(\s*(L'|D')\s*\)$", r"\1 (\2)"),
        ]

        for pattern, replacement in patterns:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', normalized).strip()
    
    def find_commune_variants(self, commune, departement=None):
        """Trouve les variantes d'une commune dans tous les datasets"""
        cache_key = f"{commune}_{departement}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        variants = []
        search_terms = self._generate_search_terms(commune)
        
        # Rechercher dans tous les datasets
        datasets_to_search = list(set(DATASETS_MAPPING.values()))
        
        for dataset in datasets_to_search:
            api_url = f"https://data.economie.gouv.fr/api/explore/v2.1/catalog/datasets/{dataset}/records"
            
            for term in search_terms:
                where_clause = f'inom LIKE "%{term}%"'
                if departement:
                    where_clause += f' AND dep="{departement}"'
                where_clause += ' AND an IN ("2019","2020","2021","2022","2023","2024")'
                
                params = {"where": where_clause, "limit": 50, "select": "inom,dep"}
                
                try:
                    response = requests.get(api_url, params=params, timeout=10)
                    data = response.json()
                    
                    if "results" in data:
                        for record in data["results"]:
                            nom = record.get("inom", "")
                            dept = record.get("dep", "")
                            if nom and self._is_similar_commune(commune, nom):
                                variant = {"nom": nom, "departement": dept}
                                if variant not in variants:
                                    variants.append(variant)
                except:
                    continue
        
        if not variants:
            variants = [{"nom": commune, "departement": departement or ""}]
        
        self._cache[cache_key] = variants
        return variants
    
    def _generate_search_terms(self, commune):
        """Génère les termes de recherche avec toutes les variantes possibles"""
        terms = [commune]
        commune_upper = commune.upper()

        # Variantes avec/sans tiret pour D' et L'
        if " D'" in commune_upper or " L'" in commune_upper:
            terms.append(re.sub(r"\s([DL]')", r"-\1", commune))
        if "-D'" in commune_upper or "-L'" in commune_upper:
            terms.append(re.sub(r"-([DL]')", r" \1", commune))

        # Gestion des articles LA/LE/LES en début
        if commune_upper.startswith(('LA ', 'LE ', 'LES ')):
            if commune_upper.startswith('LA '):
                base = commune[3:]
                # Générer toutes les combinaisons : avec/sans tiret + avec/sans espace dans parenthèses
                base_tiret = re.sub(r"\s([DL]')", r"-\1", base)
                terms.extend([
                    base, base_tiret,
                    f"{base} (LA)", f"{base} (LA )",
                    f"{base_tiret} (LA)", f"{base_tiret} (LA )"
                ])
            elif commune_upper.startswith('LE '):
                base = commune[3:]
                base_tiret = re.sub(r"\s([DL]')", r"-\1", base)
                terms.extend([
                    base, base_tiret,
                    f"{base} (LE)", f"{base} (LE )",
                    f"{base_tiret} (LE)", f"{base_tiret} (LE )"
                ])
            elif commune_upper.startswith('LES '):
                base = commune[4:]
                base_tiret = re.sub(r"\s([DL]')", r"-\1", base)
                terms.extend([
                    base, base_tiret,
                    f"{base} (LES)", f"{base} (LES )",
                    f"{base_tiret} (LES)", f"{base_tiret} (LES )"
                ])

        # Gestion des articles entre parenthèses
        if '(' in commune:
            base = re.sub(r'\s*\([^)]+\)\s*', '', commune).strip()
            article_match = re.search(r'\(\s*(LA|LE|LES|L\'|D\')\s*\)', commune_upper)
            if article_match:
                article = article_match.group(1)
                base_tiret = re.sub(r"\s([DL]')", r"-\1", base)
                terms.extend([
                    f"{article} {base}",
                    f"{article} {base_tiret}"
                ])

        return list(set(terms))
    
    def _is_similar_commune(self, search_commune, found_commune, threshold=0.8):
        """Vérifie la similarité"""
        norm1 = self.normalize_commune_name(search_commune)
        norm2 = self.normalize_commune_name(found_commune)
        return SequenceMatcher(None, norm1, norm2).ratio() >= threshold

# Instance globale du fetcher
@st.cache_resource
def get_commune_fetcher():
    return RobustCommuneFetcher()

st.title("📊 Analyse de la santé financière des communes")
st.markdown("---")

# Sidebar pour les contrôles
st.sidebar.header("🔧 Paramètres d'analyse")

# --- Liste des départements ---
departements_dispo = [f"{i:03d}" for i in range(1, 101)] + ["2A", "2B"]
dept_selection = st.sidebar.selectbox("Département", departements_dispo, key="dept_select_unique")

# --- Année ---
annees_dispo = [2024, 2023, 2022, 2021, 2020, 2019]
annee_selection = st.sidebar.selectbox("Année", annees_dispo, key="annee_select_unique")

# --- Filtres additionnels ---
st.sidebar.subheader("🔍 Filtres")
taille_min = st.sidebar.number_input("Population minimale", min_value=0, value=0, key="pop_min_unique")

# --- Fonction pour récupérer toutes les communes avec FDR (VERSION CORRIGÉE) ---
@st.cache_data(ttl=3600)
def fetch_communes(dep, an):
    """Récupère les données financières des communes avec FDR - VERSION CORRIGÉE"""
    try:
        dep = str(dep).zfill(3)
        api_url = get_api_url_for_year(an)
        dfs = []
        limit = 100
        offset = 0

        with st.spinner(f"Récupération des données pour {dep}..."):
            while True:
                params = {
                    "where": f'dep="{dep}" AND an="{an}"',
                    "limit": limit,
                    "offset": offset
                }
                
                response = requests.get(api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "results" not in data or not data["results"]:
                    break

                rows = []
                for r in data["results"]:
                    record = r.get("record", r)
                    


                    # ✅ Valeurs par habitant (€/hab) - COMMUNE (préfixe f)
                    fprod = record.get("fprod")  # RRF €/hab
                    fcharge = record.get("fcharge")  # DRF €/hab
                    fdet2cal = record.get("fdet2cal")  # Encours €/hab
                    fannu = record.get("fannu")  # Annuité €/hab
                    ffdr = record.get("ffdr")  # FDR €/hab

                    
                    # ✅ Valeurs par habitant (€/hab) - MOYENNE STRATE (préfixe m)
                    mprod = record.get("mprod")
                    mcharge = record.get("mcharge")
                    mdet2cal = record.get("mdet2cal")
                    mannu = record.get("mannu")
                    mfdr = record.get("mfdr")
                    
                    pop = record.get("pop1") or 1
                    
                    # Conversion €/hab → K€ pour compatibilité avec l'ancien code
                    rows.append({
                        "Commune": record.get("inom"),
                        "Année": record.get("an"),
                        "Population": pop,
                        
                        # COMMUNE - en K€ (recalculé depuis €/hab × population)
                        "RRF (K€)": record.get("prod"),
                        "DRF (K€)": record.get("charge"),
                        "Encours (K€)": record.get("det2cal"),  # Encours déjà en K€
                        "Annuité (K€)": record.get("annu"),  # Annuité déjà en K€
                        "FDR (K€)": record.get("fdr"),  # FDR déjà en K€
                        
                        # MOYENNE STRATE - en K€
                        "RRF - Moy. strate (K€)": record.get("mprod"),
                        "DRF - Moy. strate (K€)": record.get("mcharge"),
                        "Encours - Moy. strate (K€)": record.get("mdet2cal"),
                        "Annuité - Moy. strate (K€)": record.get("mannu"),
                        
                        "Département": record.get("dep"),
                        
                        # Épargne brute calculée
                        "Épargne brute (K€)": record.get("caf"),
                        "Épargne brute - Moy. strate (K€)": record.get("mcaf"),
                        "Caf brute (K€)": record.get("caf"),
                        
                        # ✅ NOUVEAUX CHAMPS : directement en €/hab (pas de conversion)
                        "FDR / hab Commune": ffdr,
                        "FDR / hab Moyenne": mfdr,
                        "DRF / hab Commune": fcharge,  # ✅ DÉJÀ en €/hab !
                        "DRF / hab Moyenne": mcharge,  # ✅ DÉJÀ en €/hab !
                        "RRF / hab Commune": fprod,
                        "Encours / hab Commune": fdet2cal,
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
def fetch_historical_commune_data(commune_name, dep, years_range=[2019, 2020, 2021, 2022, 2023, 2024]):
    """Récupère les données historiques d'une commune spécifique avec gestion des variantes"""
    fetcher = get_commune_fetcher()
    historical_data = []
    
    # Trouve les variantes de la commune
    variants = fetcher.find_commune_variants(commune_name, dep)

    # Debug: afficher les termes de recherche générés
    search_terms = fetcher._generate_search_terms(commune_name)
    st.info(f"🔍 Termes de recherche générés: {', '.join(search_terms)}")

    if len(variants) > 1:
        variant_names = [v["nom"] for v in variants]
        st.info(f"🔍 Variantes détectées pour {commune_name}: {', '.join(set(variant_names))}")
    else:
        st.info(f"ℹ️ Aucune variante trouvée dans l'API, utilisation de: {variants[0]['nom']}")
    
    for year in years_range:
        try:
            df_year = fetch_communes(dep, year)
            if not df_year.empty:
                commune_found = False

                # Debug: afficher les communes disponibles qui ressemblent au nom cherché
                if year == years_range[0]:  # Seulement pour la première année
                    matching_communes = df_year[df_year['Commune'].str.contains('CHAPELLE', case=False, na=False)]['Commune'].unique()
                    if len(matching_communes) > 0:
                        st.info(f"📋 Communes contenant 'CHAPELLE' en {year}: {', '.join(matching_communes[:10])}")

                # Utiliser directement les search_terms pour la recherche
                for term in search_terms:
                    commune_data = df_year[df_year['Commune'] == term]
                    if not commune_data.empty:
                        historical_data.append(commune_data.iloc[0])
                        commune_found = True
                        break

                if not commune_found:
                    st.warning(f"Commune {commune_name} non trouvée en {year}. Variantes testées: {search_terms}")
                    
        except Exception as e:
            st.warning(f"Données non disponibles pour {commune_name} en {year}")
            continue
    
    if historical_data:
        df_historical = pd.DataFrame(historical_data)
        return df_historical
    return pd.DataFrame()

# --- Fonction utilitaire pour rechercher une commune ---
def search_commune_in_department(commune_partial_name, dep, year=2023):
    """Recherche une commune par nom partiel avec normalisation"""
    fetcher = get_commune_fetcher()
    
    try:
        df_communes = fetch_communes(dep, year)
        if df_communes.empty:
            return []
        
        matches = df_communes[df_communes['Commune'].str.contains(commune_partial_name, case=False, na=False)]
        
        if matches.empty:
            normalized_search = fetcher.normalize_commune_name(commune_partial_name)
            for _, row in df_communes.iterrows():
                normalized_commune = fetcher.normalize_commune_name(row['Commune'])
                if normalized_search in normalized_commune or normalized_commune in normalized_search:
                    matches = pd.concat([matches, row.to_frame().T])
        
        return matches['Commune'].unique().tolist()
        
    except Exception as e:
        st.error(f"Erreur lors de la recherche: {e}")
        return []

# --- Interface principale ---
if st.button("📈 Analyser le département", key="analyze_button"):
    if dept_selection and annee_selection:
        df_communes = fetch_communes(dept_selection, annee_selection)
        
        if not df_communes.empty:
            if taille_min > 0:
                df_communes = df_communes[df_communes['Population'] >= taille_min]
            
            st.success(f"✅ {len(df_communes)} communes trouvées pour le département {dept_selection} en {annee_selection}")
            
            st.subheader("📊 Données des communes")
            st.dataframe(df_communes, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nombre de communes", len(df_communes))
            with col2:
                pop_totale = df_communes['Population'].sum()
                st.metric("Population totale", f"{pop_totale:,}")
            with col3:
                rrf_moyenne = df_communes['RRF (K€)'].mean()
                st.metric("RRF moyenne", f"{rrf_moyenne:.0f} K€")
            with col4:
                encours_moyen = df_communes['Encours (K€)'].mean()
                st.metric("Encours moyen", f"{encours_moyen:.0f} K€")
        else:
            st.warning("Aucune donnée trouvée pour ce département et cette année")

# --- Section recherche de commune ---
st.subheader("🔍 Recherche d'une commune spécifique")
commune_recherche = st.text_input("Nom de la commune à rechercher", key="commune_search")

if commune_recherche and st.button("Rechercher", key="search_button"):
    communes_trouvees = search_commune_in_department(commune_recherche, dept_selection, annee_selection)
    
    if communes_trouvees:
        st.success(f"Communes trouvées: {', '.join(communes_trouvees)}")
        
        commune_selectionnee = st.selectbox("Sélectionnez une commune pour l'analyse historique", 
                                           communes_trouvees, key="commune_select")
        
        if st.button("📈 Analyser l'historique", key="historical_button"):
            df_historique = fetch_historical_commune_data(commune_selectionnee, dept_selection)
            
            if not df_historique.empty:
                st.subheader(f"📈 Évolution financière de {commune_selectionnee}")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_historique['Année'], y=df_historique['RRF (K€)'], 
                                       mode='lines+markers', name='RRF'))
                fig.add_trace(go.Scatter(x=df_historique['Année'], y=df_historique['DRF (K€)'], 
                                       mode='lines+markers', name='DRF'))
                fig.add_trace(go.Scatter(x=df_historique['Année'], y=df_historique['Épargne brute (K€)'], 
                                       mode='lines+markers', name='Épargne brute'))
                
                fig.update_layout(title=f"Évolution financière - {commune_selectionnee}",
                                xaxis_title="Année", yaxis_title="Montant (K€)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_historique, use_container_width=True)
            else:
                st.warning("Pas de données historiques disponibles pour cette commune")
    else:
        st.warning("Aucune commune trouvée avec ce nom")

# --- Fonction pour calculer les KPI historiques (VERSION V3) ---
def calculate_historical_kpis(df_historical):
    """Calcule les KPI historiques pour la commune et sa strate"""
    if df_historical.empty:
        return pd.DataFrame()
    
    df_kpi_hist = df_historical.copy()
    
    # ============================================
    # 1. CALCULER TOUS LES KPI D'ABORD
    # ============================================
    
    # KPI Commune
    df_kpi_hist["TEB Commune (%)"] = df_kpi_hist["Épargne brute (K€)"] / df_kpi_hist["RRF (K€)"].replace(0, pd.NA) * 100
    df_kpi_hist["Années de Désendettement"] = df_kpi_hist["Encours (K€)"] / df_kpi_hist["Épargne brute (K€)"].replace(0, pd.NA)
    df_kpi_hist["Annuité/CAF Commune (%)"] = df_kpi_hist["Annuité (K€)"] / df_kpi_hist["Épargne brute (K€)"].replace(0, pd.NA) * 100
    df_kpi_hist["Caf brute (K€)"] = df_kpi_hist["Caf brute (K€)"]
    
    # ✅ FDR en jours - VERSION CORRIGÉE
    if 'FDR / hab Commune' in df_kpi_hist.columns and 'DRF / hab Commune' in df_kpi_hist.columns:
        df_kpi_hist['FDR Jours Commune'] = (
            df_kpi_hist['FDR / hab Commune'] / df_kpi_hist['DRF / hab Commune'].replace(0, pd.NA) * 365
        ).round(2)
    else:
        df_kpi_hist['FDR Jours Commune'] = pd.NA

    # KPI Strate
    df_kpi_hist["TEB Strate (%)"] = df_kpi_hist["Épargne brute - Moy. strate (K€)"] / df_kpi_hist["RRF - Moy. strate (K€)"].replace(0, pd.NA) * 100
    df_kpi_hist["CD Strate (années)"] = df_kpi_hist["Encours - Moy. strate (K€)"] / df_kpi_hist["Épargne brute - Moy. strate (K€)"].replace(0, pd.NA)
    df_kpi_hist["Annuité/CAF Strate (%)"] = df_kpi_hist["Annuité - Moy. strate (K€)"] / df_kpi_hist["Épargne brute - Moy. strate (K€)"].replace(0, pd.NA) * 100
    
    # FDR Strate
    if 'FDR / hab Moyenne' in df_kpi_hist.columns and 'DRF / hab Moyenne' in df_kpi_hist.columns:
        df_kpi_hist['FDR Jours Moyenne'] = (
            df_kpi_hist['FDR / hab Moyenne'] / df_kpi_hist['DRF / hab Moyenne'].replace(0, pd.NA) * 365
        ).round(2)
        df_kpi_hist.loc[df_kpi_hist['FDR Jours Moyenne'] > 1000, 'FDR Jours Moyenne'] = pd.NA
    else:
        df_kpi_hist['FDR Jours Moyenne'] = pd.NA
    
    # ============================================
    # 2. CRÉER LES COLONNES DE SCORING POUR V3
    # ============================================
    # Créer des colonnes temporaires pour le scoring V3
    df_kpi_hist["TEB (%)"] = df_kpi_hist["TEB Commune (%)"]
    # "Années de Désendettement" est déjà présente
    df_kpi_hist["Annuité / CAF (%)"] = df_kpi_hist["Annuité/CAF Commune (%)"]
    # FDR Jours Commune est déjà présente
    
    # ============================================
    # 3. CALCULER LE SCORE V3
    # ============================================
    df_kpi_hist['Score Commune'] = df_kpi_hist.apply(score_sante_financiere_v3, axis=1, df_ref=df_kpi_hist)
    df_kpi_hist['Niveau d\'alerte'] = df_kpi_hist['Score Commune'].apply(niveau_alerte_v3)
    
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
    fig_teb.add_hline(y=15, line_dash="dot", line_color="green", annotation_text="Seuil bon (15%)")
    fig_teb.add_hline(y=10, line_dash="dot", line_color="orange", annotation_text="Seuil critique (10%)")
    fig_teb.update_layout(
        title="📈 Évolution du Taux d'Épargne Brute (TEB)",
        xaxis_title="Année",
        yaxis_title="TEB (%)",
        hovermode='x unified'
    )
    
    # Graphique 2: Évolution Capacité de désendettement
    fig_cd = go.Figure()
    fig_cd.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Années de Désendettement'],
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
    
    # Graphique 3: Évolution Ratio Annuité/CAF
    fig_annuite = go.Figure()
    fig_annuite.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Annuité/CAF Commune (%)'],
        mode='lines+markers',
        name=f'{commune_name}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_annuite.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['Annuité/CAF Strate (%)'],
        mode='lines+markers',
        name='Moyenne strate',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_annuite.add_hline(y=50, line_dash="dot", line_color="green", annotation_text="Seuil bon (50%)")
    fig_annuite.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="Seuil critique (60%)")
    fig_annuite.update_layout(
        title="💳 Évolution du Ratio Annuité/CAF Brute",
        xaxis_title="Année",
        yaxis_title="Annuité/CAF (%)",
        hovermode='x unified'
    )
    
    # Graphique 4: Évolution FDR en jours
    fig_fdr = go.Figure()
    fig_fdr.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['FDR Jours Commune'],
        mode='lines+markers',
        name=f'{commune_name}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_fdr.add_trace(go.Scatter(
        x=df_historical_kpi['Année'], 
        y=df_historical_kpi['FDR Jours Moyenne'],
        mode='lines+markers',
        name='Moyenne strate',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    fig_fdr.add_hline(y=240, line_dash="dot", line_color="green", annotation_text="Seuil bon (240j)")
    fig_fdr.add_hline(y=60, line_dash="dot", line_color="red", annotation_text="Seuil critique (60j)")
    fig_fdr.update_layout(
        title="👥 Évolution du Fonds de Roulement",
        xaxis_title="Année",
        yaxis_title="FDR (jours de DRF)",
        hovermode='x unified'
    )
    
    return fig_teb, fig_cd, fig_annuite, fig_fdr

def create_evolution_charts_seaborn(df_historical_kpi, commune_name):
    """
    Crée les graphiques d'évolution des KPI avec Seaborn
    Retourne 4 figures matplotlib
    """
    if df_historical_kpi.empty:
        return None, None, None, None
    
    # ========================================
    # Graphique 1: Évolution TEB
    # ========================================
    fig_teb, ax = plt.subplots(figsize=(12, 6))
    
    # Ligne commune
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='TEB Commune (%)',
        marker='o',
        linewidth=3,
        markersize=10,
        label=f'{commune_name}',
        color='#1f77b4',
        ax=ax
    )
    
    # Ligne strate
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='TEB Strate (%)',
        marker='o',
        linewidth=2,
        markersize=8,
        linestyle='--',
        label='Moyenne strate',
        color='#ff7f0e',
        ax=ax
    )
    
    # Lignes de seuil
    ax.axhline(y=15, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 15.5, 'Seuil bon (15%)', 
            color='green', fontsize=9, va='bottom')
    
    ax.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 10.5, 'Seuil critique (10%)', 
            color='orange', fontsize=9, va='bottom')
    
    ax.set_title("📈 Évolution du Taux d'Épargne Brute (TEB)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12)
    ax.set_ylabel("TEB (%)", fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # ========================================
    # Graphique 2: Évolution Capacité de désendettement
    # ========================================
    fig_cd, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='Années de Désendettement',
        marker='o',
        linewidth=3,
        markersize=10,
        label=f'{commune_name}',
        color='#1f77b4',
        ax=ax
    )
    
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='CD Strate (années)',
        marker='o',
        linewidth=2,
        markersize=8,
        linestyle='--',
        label='Moyenne strate',
        color='#ff7f0e',
        ax=ax
    )
    
    ax.axhline(y=8, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 8.5, 'Seuil bon (8 ans)', 
            color='green', fontsize=9, va='bottom')
    
    ax.axhline(y=12, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 12.5, 'Seuil critique (12 ans)', 
            color='red', fontsize=9, va='bottom')
    
    ax.set_title("⏳ Évolution de la Capacité de Désendettement", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12)
    ax.set_ylabel("Capacité (années)", fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # ========================================
    # Graphique 3: Évolution Ratio Annuité/CAF
    # ========================================
    fig_annuite, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='Annuité/CAF Commune (%)',
        marker='o',
        linewidth=3,
        markersize=10,
        label=f'{commune_name}',
        color='#1f77b4',
        ax=ax
    )
    
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='Annuité/CAF Strate (%)',
        marker='o',
        linewidth=2,
        markersize=8,
        linestyle='--',
        label='Moyenne strate',
        color='#ff7f0e',
        ax=ax
    )
    
    ax.axhline(y=50, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 50.5, 'Seuil bon (50%)', 
            color='green', fontsize=9, va='bottom')
    
    ax.axhline(y=60, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 60.5, 'Seuil critique (60%)', 
            color='red', fontsize=9, va='bottom')
    
    ax.set_title("💳 Évolution du Ratio Annuité/CAF Brute", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12)
    ax.set_ylabel("Annuité/CAF (%)", fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # ========================================
    # Graphique 4: Évolution FDR en jours
    # ========================================
    fig_fdr, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='FDR Jours Commune',
        marker='o',
        linewidth=3,
        markersize=10,
        label=f'{commune_name}',
        color='#1f77b4',
        ax=ax
    )
    
    sns.lineplot(
        data=df_historical_kpi, 
        x='Année', 
        y='FDR Jours Moyenne',
        marker='o',
        linewidth=2,
        markersize=8,
        linestyle='--',
        label='Moyenne strate',
        color='#ff7f0e',
        ax=ax
    )
    
    ax.axhline(y=240, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 245, 'Seuil bon (240j)', 
            color='green', fontsize=9, va='bottom')
    
    ax.axhline(y=60, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(df_historical_kpi['Année'].min(), 65, 'Seuil critique (60j)', 
            color='red', fontsize=9, va='bottom')
    
    ax.set_title("👥 Évolution du Fonds de Roulement", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12)
    ax.set_ylabel("FDR (jours de DRF)", fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_teb, fig_cd, fig_annuite, fig_fdr


def create_score_evolution_chart_seaborn(df_historical_kpi, commune_name):
    """
    Crée un graphique d'évolution du score avec design professionnel
    """
    if df_historical_kpi.empty or len(df_historical_kpi) < 2:
        return None
    
    df = df_historical_kpi.sort_values('Année').reset_index(drop=True)
    df['Année'] = pd.to_numeric(df['Année'], errors='coerce')
    
    # ========================================
    # CONFIGURATION STYLE
    # ========================================
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.edgecolor': '.2',
        'axes.linewidth': 1.5
    })
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # ========================================
    # ZONES DE COULEUR AMÉLIORÉES
    # ========================================
    ax.axhspan(75, 100, facecolor='#00C851', alpha=0.08, zorder=0, 
               label='Zone Verte (75-100)')
    ax.axhspan(50, 75, facecolor='#FF8C00', alpha=0.08, zorder=0,
               label='Zone Orange (50-75)')
    ax.axhspan(0, 50, facecolor='#FF4B4B', alpha=0.08, zorder=0,
               label='Zone Rouge (0-50)')
    
    # ========================================
    # LIGNE PRINCIPALE AVEC GRADIENT
    # ========================================
    # Points colorés selon le niveau
    colors = []
    for score in df['Score Commune']:
        if score >= 75:
            colors.append('#00C851')  # Vert
        elif score >= 50:
            colors.append('#FF8C00')  # Orange
        else:
            colors.append('#FF4B4B')  # Rouge
    
    # Ligne de base épaisse
    ax.plot(df['Année'], df['Score Commune'], 
            color='#2C3E50', linewidth=4, alpha=0.8, zorder=3)
    
    # Points avec couleurs conditionnelles
    ax.scatter(df['Année'], df['Score Commune'], 
              c=colors, s=250, zorder=4, edgecolors='white', linewidths=2.5,
              label='Score de la commune')
    
    # ========================================
    # LIGNES DE SEUIL STYLISÉES
    # ========================================
    ax.axhline(y=75, color='#00C851', linestyle='--', linewidth=2.5, 
               alpha=0.9, zorder=2)
    ax.axhline(y=50, color='#FF8C00', linestyle='--', linewidth=2.5, 
               alpha=0.9, zorder=2)
    
    # Annotations des seuils (à droite)
    ax.text(df['Année'].max() + 0.15, 75, '✓ Seuil Vert', 
            color='#00C851', fontsize=11, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#00C851', linewidth=2))
    
    ax.text(df['Année'].max() + 0.15, 50, '⚠ Seuil Orange', 
            color='#FF8C00', fontsize=11, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#FF8C00', linewidth=2))
    
    # ========================================
    # ANNOTATIONS DES VALEURS AMÉLIORÉES
    # ========================================
    for idx, row in df.iterrows():
        score = row['Score Commune']
        
        # Déterminer la couleur et le symbole
        if score >= 75:
            color = '#00C851'
            symbol = '●'
        elif score >= 50:
            color = '#FF8C00'
            symbol = '●'
        else:
            color = '#FF4B4B'
            symbol = '●'
        
        # Annotation avec fond blanc
        ax.annotate(f"{score:.1f}", 
                   xy=(row['Année'], score),
                   xytext=(0, 15), 
                   textcoords='offset points',
                   ha='center', 
                   fontsize=10, 
                   fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', 
                            edgecolor=color,
                            linewidth=1.5,
                            alpha=0.95))
    
    # ========================================
    # TENDANCE (RÉGRESSION LINÉAIRE)
    # ========================================
    from scipy import stats
    x_num = np.arange(len(df))
    y = df['Score Commune'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_num, y)
    tendance = slope * x_num + intercept
    
    ax.plot(df['Année'], tendance, 
            color='#34495E', linestyle=':', linewidth=2.5, alpha=0.6,
            label=f'Tendance ({slope:+.1f} pts/an)')
    
    # ========================================
    # TITRES ET LABELS AMÉLIORÉS
    # ========================================
    ax.set_title(f"📊 Évolution du Score de Santé Financière\n{commune_name} | Période 2019-2024", 
                 fontsize=16, fontweight='bold', pad=25,
                 color='#2C3E50')
    
    ax.set_xlabel("Année", fontsize=13, fontweight='bold', color='#2C3E50')
    ax.set_ylabel("Score de Santé (/100)", fontsize=13, fontweight='bold', color='#2C3E50')
    
    # ========================================
    # AXES ET GRILLE
    # ========================================
    ax.set_ylim(-5, 105)
    ax.set_xlim(df['Année'].min() - 0.3, df['Année'].max() + 0.8)
    
    # Ticks personnalisés
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(['0', '25', '50 ⚠', '75 ✓', '100'], 
                       fontsize=11, fontweight='bold')
    
    ax.tick_params(axis='both', which='major', labelsize=11, 
                   colors='#2C3E50', width=1.5)
    
    # Grille améliorée
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=1, color='#95A5A6')
    ax.set_axisbelow(True)
    
    # ========================================
    # LÉGENDE PROFESSIONNELLE
    # ========================================
    legend = ax.legend(loc='upper left', 
                      frameon=True, 
                      shadow=True,
                      fancybox=True,
                      framealpha=0.95,
                      fontsize=10,
                      edgecolor='#2C3E50',
                      title='Légende',
                      title_fontsize=11)
    legend.get_frame().set_linewidth(1.5)
    
    # ========================================
    # BORDURE EXTÉRIEURE
    # ========================================
    for spine in ax.spines.values():
        spine.set_edgecolor('#2C3E50')
        spine.set_linewidth(2)
    
    # ========================================
    # NOTE DE BAS DE PAGE
    # ========================================
    fig.text(0.99, 0.01, 
             'Note : Score basé sur TEB (20%), CD (30%), Annuité/CAF (30%), FDR (20%)',
             ha='right', va='bottom', fontsize=8, style='italic', color='#7F8C8D')
    
    plt.tight_layout()
    sns.reset_defaults()  # Réinitialiser le style pour ne pas affecter les autres graphiques
    
    return fig




def create_score_evolution_chart(df_historical_kpi, commune_name):
    """
    Crée un graphique d'évolution du score avec tendance linéaire
    """
    if df_historical_kpi.empty or len(df_historical_kpi) < 2:
        return None
    
    df = df_historical_kpi.sort_values('Année').reset_index(drop=True)
    
    # Calcul de la ligne de tendance (régression linéaire)
    from scipy import stats
    
    # Convertir les années en valeurs numériques pour la régression
    x = np.arange(len(df))
    y = df['Score Commune'].values
    
    # Régression linéaire
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    tendance = slope * x + intercept
    
    # Création de la figure
    fig = go.Figure()
    
    # Ligne du score réel
    fig.add_trace(go.Scatter(
        x=df['Année'],
        y=df['Score Commune'],
        mode='lines+markers',
        name=commune_name,
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10, symbol='circle'),
        fill=None,
        hovertemplate='<b>%{x}</b><br>Score : %{y:.1f}/100<extra></extra>'
    ))
    
        
    # Zones de couleur (seuils de scoring)
    fig.add_hrect(y0=75, y1=100, fillcolor="#00C851", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="#FF8C00", opacity=0.1, layer="below", line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor="#FF4B4B", opacity=0.1, layer="below", line_width=0)
    
        
    # Mise en page
    fig.update_layout(
        title=f"📈 Évolution du score de santé financière - {commune_name} (2019-2024)",
        xaxis_title="Année",
        yaxis_title="Score de santé (/100)",
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ),
        height=500
    )
       
    return fig

# === NOUVELLES FONCTIONS DE VISUALISATION ===

def create_score_evolution_stacked_bar(df_historical_kpi, commune_name):
    """
    Crée un graphique en barres empilées montrant la contribution de chaque composante au score
    """
    if df_historical_kpi.empty or len(df_historical_kpi) < 2:
        return None
    
    df = df_historical_kpi.sort_values('Année').reset_index(drop=True)
    
    # Recalculer les composantes du score normalisées (0-100)
    # pour visualiser la contribution
    
    # 1. TEB contribution (0-20 points, normalisé à 0-100)
    teb_scores = []
    for _, row in df.iterrows():
        if pd.notna(row['TEB (%)']):
            if row['TEB (%)'] > 20:
                teb_scores.append(20)
            elif row['TEB (%)'] >= 10:
                teb_scores.append(((row['TEB (%)'] - 10) / 10) * 20)
            else:
                teb_scores.append(0)
        else:
            teb_scores.append(0)
    
    # 2. CD contribution (0-30 points, normalisé à 0-100)
    cd_scores = []
    for _, row in df.iterrows():
        if pd.notna(row.get('Années de Désendettement')) and row.get('Années de Désendettement') >= 0:
            cd_value = row.get('Années de Désendettement')
            if cd_value < 6:
                cd_scores.append(30)
            elif cd_value <= 16:
                cd_scores.append(30 - ((cd_value - 6) / 10) * 30)
            else:
                cd_scores.append(0)
        else:
            cd_scores.append(15)
    
    # 3. Annuité/CAF contribution (0-30 points, normalisé à 0-100)
    annuite_scores = []
    for _, row in df.iterrows():
        if pd.notna(row.get('Annuité / CAF (%)')):
            annuite_caf = row.get('Annuité / CAF (%)')
            
            # ⭐ VALIDATION : Annuité/CAF négatif = 0 points
            if annuite_caf < 0:
                annuite_scores.append(0)  # 🔴 Aberrant
            elif annuite_caf == 0:
                annuite_scores.append(30)  # 🟢 Zéro = excellent
            elif annuite_caf < 30:
                annuite_scores.append(30)
            elif annuite_caf <= 50:
                annuite_scores.append(30 - ((annuite_caf - 30) / 20) * 30)
            else:
                annuite_scores.append(0)
        else:
            annuite_scores.append(30)
    
    # 4. FDR contribution (0-20 points, normalisé à 0-100)
    fdr_scores = []
    for _, row in df.iterrows():
        if pd.notna(row.get('FDR Jours Commune')):
            fdr_jours = row.get('FDR Jours Commune')
            if fdr_jours > 240:
                fdr_scores.append(20)
            elif fdr_jours >= 70:
                fdr_scores.append(((fdr_jours - 70) / 170) * 20)
            elif fdr_jours >= 30:
                fdr_scores.append(((fdr_jours - 30) / 40) * 10)
            else:
                fdr_scores.append(0)
        else:
            fdr_scores.append(10)
    
    # Créer le dataframe pour le stacked bar
    df_stacked = pd.DataFrame({
        'Année': df['Année'],
        'TEB (20 pts)': teb_scores,
        'Annuité/CAF (30 pts)': annuite_scores,
        'CD (30 pts)': cd_scores,
        'FDR (20 pts)': fdr_scores,
    })
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Ordre d'empilement : FDR en bas, puis CD, Annuité/CAF, TEB en haut
    fig.add_trace(go.Bar(
        x=df_stacked['Année'],
        y=df_stacked['FDR (20 pts)'],
        name='FDR (20 pts)',
        marker_color=colors[3],
        hovertemplate='<b>%{x}</b><br>FDR : %{y:.1f} pts<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=df_stacked['Année'],
        y=df_stacked['CD (30 pts)'],
        name='CD (30 pts)',
        marker_color=colors[2],
        hovertemplate='<b>%{x}</b><br>CD : %{y:.1f} pts<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=df_stacked['Année'],
        y=df_stacked['Annuité/CAF (30 pts)'],
        name='Annuité/CAF (30 pts)',
        marker_color=colors[1],
        hovertemplate='<b>%{x}</b><br>Annuité/CAF : %{y:.1f} pts<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=df_stacked['Année'],
        y=df_stacked['TEB (20 pts)'],
        name='TEB (20 pts)',
        marker_color=colors[0],
        hovertemplate='<b>%{x}</b><br>TEB : %{y:.1f} pts<extra></extra>'
    ))
    
    fig.update_layout(
        barmode='stack',
        title=f"📊 Évolution du score par composante (stacked) - {commune_name}",
        xaxis_title="Année",
        yaxis_title="Points",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_score_evolution_lines(df_historical_kpi, commune_name):
    """
    Crée un graphique en lignes montrant l'évolution du score global 
    ET de chaque composante normalisée à 0-100
    """
    if df_historical_kpi.empty or len(df_historical_kpi) < 2:
        return None
    
    df = df_historical_kpi.sort_values('Année').reset_index(drop=True)
    
    # Recalculer les composantes du score normalisées (0-100)
    
    # 1. TEB contribution (0-20 points, normalisé à 0-100)
    teb_norm = []
    for _, row in df.iterrows():
        if pd.notna(row['TEB (%)']):
            if row['TEB (%)'] > 20:
                teb_norm.append(100)
            elif row['TEB (%)'] >= 10:
                score_pts = ((row['TEB (%)'] - 10) / 10) * 20
                teb_norm.append((score_pts / 20) * 100)
            else:
                teb_norm.append(0)
        else:
            teb_norm.append(0)
    
    # 2. CD contribution (0-30 points, normalisé à 0-100)
    cd_norm = []
    for _, row in df.iterrows():
        if pd.notna(row.get('Années de Désendettement')) and row.get('Années de Désendettement') >= 0:
            cd_value = row.get('Années de Désendettement')
            if cd_value < 6:
                cd_norm.append(100)
            elif cd_value <= 16:
                score_pts = 30 - ((cd_value - 6) / 10) * 30
                cd_norm.append((score_pts / 30) * 100)
            else:
                cd_norm.append(0)
        else:
            cd_norm.append(50)
    
    # 3. Annuité/CAF contribution (0-30 points, normalisé à 0-100)
    annuite_norm = []
    for _, row in df.iterrows():
        if pd.notna(row.get('Annuité / CAF (%)')):
            annuite_caf = row.get('Annuité / CAF (%)')
            
            # ⭐ VALIDATION : Annuité/CAF négatif = 0 (aberrant)
            if annuite_caf < 0:
                annuite_norm.append(0)  # 🔴 Aberrant = pire score
            elif annuite_caf == 0:
                annuite_norm.append(100)  # 🟢 Zéro = excellent = 100
            elif annuite_caf < 30:
                annuite_norm.append(100)
            elif annuite_caf <= 50:
                score_pts = 30 - ((annuite_caf - 30) / 20) * 30
                annuite_norm.append((score_pts / 30) * 100)
            else:
                annuite_norm.append(0)
        else:
            annuite_norm.append(100)
    
    # 4. FDR contribution (0-20 points, normalisé à 0-100)
    fdr_norm = []
    for _, row in df.iterrows():
        if pd.notna(row.get('FDR Jours Commune')):
            fdr_jours = row.get('FDR Jours Commune')
            if fdr_jours > 240:
                fdr_norm.append(100)
            elif fdr_jours >= 70:
                score_pts = ((fdr_jours - 70) / 170) * 20
                fdr_norm.append((score_pts / 20) * 100)
            elif fdr_jours >= 30:
                score_pts = ((fdr_jours - 30) / 40) * 10
                fdr_norm.append((score_pts / 20) * 100)
            else:
                fdr_norm.append(0)
        else:
            fdr_norm.append(50)
    
    fig = go.Figure()
    
    # Score global (ligne épaisse en premier plan)
    fig.add_trace(go.Scatter(
        x=df['Année'],
        y=df['Score Commune'],
        mode='lines+markers',
        name='Score Global (/100)',
        line=dict(color='black', width=4),
        marker=dict(size=12, symbol='circle'),
        hovertemplate='<b>%{x}</b><br>Score : %{y:.1f}/100<extra></extra>'
    ))
    
    # TEB normalisé (0-100)
    fig.add_trace(go.Scatter(
        x=df['Année'],
        y=teb_norm,
        mode='lines+markers',
        name='TEB Santé (0-100)',
        line=dict(color='#1f77b4', width=2, dash='dash'),
        marker=dict(size=8),
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>TEB Santé : %{y:.0f}%<extra></extra>'
    ))
    
    # Annuité/CAF normalisé (0-100)
    fig.add_trace(go.Scatter(
        x=df['Année'],
        y=annuite_norm,
        mode='lines+markers',
        name='Annuité/CAF Santé (0-100)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=8),
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>Annuité/CAF Santé : %{y:.0f}%<extra></extra>'
    ))
    
    # CD normalisé (0-100)
    fig.add_trace(go.Scatter(
        x=df['Année'],
        y=cd_norm,
        mode='lines+markers',
        name='CD Santé (0-100)',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        marker=dict(size=8),
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>CD Santé : %{y:.0f}%<extra></extra>'
    ))
    
    # FDR normalisé (0-100)
    fig.add_trace(go.Scatter(
        x=df['Année'],
        y=fdr_norm,
        mode='lines+markers',
        name='FDR Santé (0-100)',
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(size=8),
        opacity=0.7,
        hovertemplate='<b>%{x}</b><br>FDR Santé : %{y:.0f}%<extra></extra>'
    ))
    
    # Zones de seuil
    fig.add_hrect(y0=75, y1=100, fillcolor="#00C851", opacity=0.05, layer="below", line_width=0)
    fig.add_hrect(y0=50, y1=75, fillcolor="#FF8C00", opacity=0.05, layer="below", line_width=0)
    fig.add_hrect(y0=0, y1=50, fillcolor="#FF4B4B", opacity=0.05, layer="below", line_width=0)
    
    # Lignes de seuil
    fig.add_hline(y=75, line_dash="dash", line_color="green", line_width=1,
                  annotation_text="Seuil Vert (75)", annotation_position="right")
    fig.add_hline(y=50, line_dash="dash", line_color="orange", line_width=1,
                  annotation_text="Seuil Orange (50)", annotation_position="right")
    
    fig.update_layout(
        title=f"📈 Évolution détaillée du score par composante - {commune_name}",
        xaxis_title="Année",
        yaxis_title="Score (0-100)",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        yaxis=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_score_evolution_lines_seaborn(df_historical_kpi, commune_name):

    """
    Crée un graphique d'évolution détaillée par composante - Design expert
    """
    if df_historical_kpi.empty or len(df_historical_kpi) < 2:
        return None
    
    df = df_historical_kpi.sort_values('Année').reset_index(drop=True)
    df['Année'] = pd.to_numeric(df['Année'], errors='coerce')
    
    # ========================================
    # CALCUL DES COMPOSANTES NORMALISÉES
    # ========================================
    teb_norm = []
    cd_norm = []
    annuite_norm = []
    fdr_norm = []
    
    for _, row in df.iterrows():
        # TEB
        if pd.notna(row['TEB (%)']):
            if row['TEB (%)'] > 20:
                teb_norm.append(100)
            elif row['TEB (%)'] >= 10:
                score_pts = ((row['TEB (%)'] - 10) / 10) * 20
                teb_norm.append((score_pts / 20) * 100)
            else:
                teb_norm.append(0)
        else:
            teb_norm.append(0)
        
        # CD
        if pd.notna(row.get('Années de Désendettement')) and row.get('Années de Désendettement') >= 0:
            cd_value = row.get('Années de Désendettement')
            if cd_value < 6:
                cd_norm.append(100)
            elif cd_value <= 16:
                score_pts = 30 - ((cd_value - 6) / 10) * 30
                cd_norm.append((score_pts / 30) * 100)
            else:
                cd_norm.append(0)
        else:
            cd_norm.append(50)
        
        # Annuité/CAF
        if pd.notna(row.get('Annuité / CAF (%)')):
            annuite_caf = row.get('Annuité / CAF (%)')
            if annuite_caf < 0:
                annuite_norm.append(0)  # 🔴 Aberrant = pire score
            elif annuite_caf == 0:
                annuite_norm.append(100)  # 🟢 Zéro = excellent = 100
            elif annuite_caf < 30:
                annuite_norm.append(100)
            elif annuite_caf <= 50:
                score_pts = 30 - ((annuite_caf - 30) / 20) * 30
                annuite_norm.append((score_pts / 30) * 100)
            else:
                annuite_norm.append(0)
        else:
            annuite_norm.append(100)
        
        # FDR
        if pd.notna(row.get('FDR Jours Commune')):
            fdr_jours = row.get('FDR Jours Commune')
            if fdr_jours > 240:
                fdr_norm.append(100)
            elif fdr_jours >= 70:
                score_pts = ((fdr_jours - 70) / 170) * 20
                fdr_norm.append((score_pts / 20) * 100)
            elif fdr_jours >= 30:
                score_pts = ((fdr_jours - 30) / 40) * 10
                fdr_norm.append((score_pts / 20) * 100)
            else:
                fdr_norm.append(0)
        else:
            fdr_norm.append(50)
    
    # ========================================
    # CONFIGURATION STYLE EXPERT
    # ========================================
    sns.set_theme(style="whitegrid", palette="husl")
    sns.set_palette("Set2")
    
    custom_style = {
        'grid.linestyle': '--',
        'grid.alpha': 0.15,
        'axes.edgecolor': '#1a1a1a',
        'axes.linewidth': 1.2,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.bottom': False,
        'ytick.left': False,
    }
    sns.set_style("whitegrid", custom_style)
    
    fig, ax = plt.subplots(figsize=(18, 9), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFBFC')
    
    # ========================================
    # ZONES DE COULEUR DÉGRADÉE
    # ========================================
    # Zones avec gradient subtil
    ax.axhspan(75, 100, facecolor='#00C851', alpha=0.06, zorder=0, linewidth=0)
    ax.axhspan(50, 75, facecolor='#FFB84D', alpha=0.06, zorder=0, linewidth=0)
    ax.axhspan(0, 50, facecolor='#FF6B6B', alpha=0.06, zorder=0, linewidth=0)
    
    # ========================================
    # LIGNES DE SEUIL AVEC DÉGRADÉ
    # ========================================
    ax.axhline(y=75, color='#00C851', linestyle='--', linewidth=2.2, 
               alpha=0.6, zorder=2, dash_capstyle='round')
    ax.axhline(y=50, color='#FFB84D', linestyle='--', linewidth=2.2, 
               alpha=0.6, zorder=2, dash_capstyle='round')
    
    # ========================================
    # DÉFINITION DES COMPOSANTES AVEC PALETTE COHÉRENTE
    # ========================================
    components = [
        {
            'data': df['Score Commune'],
            'label': 'Score Global (/100)',
            'color': '#1A1A2E',
            'linewidth': 3.5,
            'marker': 'o',
            'markersize': 14,
            'linestyle': '-',
            'alpha': 1.0,
            'zorder': 5,
            'markerfacecolor': '#1A1A2E',
            'markeredgewidth': 2.5,
            'markeredgecolor': 'white'
        },
        {
            'data': teb_norm,
            'label': 'TEB Santé (20%)',
            'color': '#2E86AB',
            'linewidth': 2.2,
            'marker': 's',
            'markersize': 9,
            'linestyle': '--',
            'alpha': 0.7,
            'zorder': 4,
            'markerfacecolor': '#A8DADC',
            'markeredgewidth': 1.5,
            'markeredgecolor': '#2E86AB'
        },
        {
            'data': cd_norm,
            'label': 'CD Santé (30%)',
            'color': '#27AE60',
            'linewidth': 2.2,
            'marker': '^',
            'markersize': 9,
            'linestyle': '--',
            'alpha': 0.7,
            'zorder': 4,
            'markerfacecolor': '#A9DFBF',
            'markeredgewidth': 1.5,
            'markeredgecolor': '#27AE60'
        },
        {
            'data': annuite_norm,
            'label': 'Annuité/CAF Santé (30%)',
            'color': '#F39C12',
            'linewidth': 2.2,
            'marker': 'D',
            'markersize': 8,
            'linestyle': '--',
            'alpha': 0.7,
            'zorder': 4,
            'markerfacecolor': '#F9E79F',
            'markeredgewidth': 1.5,
            'markeredgecolor': '#F39C12'
        },
        {
            'data': fdr_norm,
            'label': 'FDR Santé (20%)',
            'color': '#E74C3C',
            'linewidth': 2.2,
            'marker': 'v',
            'markersize': 9,
            'linestyle': '--',
            'alpha': 0.7,
            'zorder': 4,
            'markerfacecolor': '#F5B7B1',
            'markeredgewidth': 1.5,
            'markeredgecolor': '#E74C3C'
        }
    ]
    
    # ========================================
    # TRAÇAGE AVEC EFFETS VISUELS AVANCÉS
    # ========================================
    for i, comp in enumerate(components):
        # Tracer avec effet d'ombre (blur effect)
        if i > 0:  # Pas d'ombre pour le score global
            ax.plot(df['Année'], comp['data'], 
                   color=comp['color'],
                   linewidth=comp['linewidth'] + 1.5,
                   alpha=0.1,
                   linestyle=comp['linestyle'],
                   zorder=comp['zorder'] - 1)
        
        # Ligne principale
        ax.plot(df['Année'], comp['data'], 
                marker=comp['marker'],
                linewidth=comp['linewidth'],
                markersize=comp['markersize'],
                linestyle=comp['linestyle'],
                alpha=comp['alpha'],
                label=comp['label'],
                color=comp['color'],
                zorder=comp['zorder'],
                markerfacecolor=comp['markerfacecolor'],
                markeredgewidth=comp['markeredgewidth'],
                markeredgecolor=comp['markeredgecolor'])
    
    # ========================================
    # ANNOTATIONS INTELLIGENTES DU SCORE GLOBAL
    # ========================================
    for idx, row in df.iterrows():
        score = row['Score Commune']
        année = row['Année']
        
        # Déterminer la couleur selon le score
        if score >= 75:
            color = '#00C851'
            bg_color = '#E8F8F5'
        elif score >= 50:
            color = '#F39C12'
            bg_color = '#FEF5E7'
        else:
            color = '#E74C3C'
            bg_color = '#FADBD8'
        
        # Annotation stylisée
        ax.annotate(f"{score:.1f}", 
                   xy=(année, score),
                   xytext=(0, 22), 
                   textcoords='offset points',
                   ha='center', 
                   fontsize=10, 
                   fontweight='bold',
                   color=color,
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=bg_color, 
                            edgecolor=color,
                            linewidth=2,
                            alpha=0.98),
                   arrowprops=dict(arrowstyle='->', 
                                  color=color, 
                                  lw=1, 
                                  alpha=0.4),
                   zorder=6)
    
    # ========================================
    # ANNOTATIONS DES SEUILS AMÉLIORÉES
    # ========================================
    ax.text(df['Année'].max() + 0.2, 75, '✓ Seuil Vert (75)', 
            color='#00C851', fontsize=10, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F8F5', 
                     edgecolor='#00C851', linewidth=2.2, alpha=0.98))
    
    ax.text(df['Année'].max() + 0.2, 50, '⚠ Seuil Orange (50)', 
            color='#F39C12', fontsize=10, fontweight='bold', va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FEF5E7', 
                     edgecolor='#F39C12', linewidth=2.2, alpha=0.98))
    
    # ========================================
    # TITRES ET LABELS EXPERT
    # ========================================
    ax.set_title(f"📈 Évolution détaillée du score par composante\n{commune_name} | Période 2019-2024", 
                 fontsize=18, fontweight='bold', pad=30,
                 color='#1A1A2E', loc='left')
    
    ax.set_xlabel("Année", fontsize=13, fontweight='600', color='#34495E', labelpad=15)
    ax.set_ylabel("Score de Santé (/100)", fontsize=13, fontweight='600', color='#34495E', labelpad=15)
    
    # ========================================
    # CONFIGURATION AXES AVANCÉE
    # ========================================
    ax.set_ylim(-8, 108)
    ax.set_xlim(df['Année'].min() - 0.4, df['Année'].max() + 1.2)
    
    # Ticks personnalisés avec style
    y_ticks = [0, 25, 50, 75, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '25', '50', '75', '100'], 
                       fontsize=11, fontweight='600', color='#34495E')
    
    # Petit grid pour les ticks principaux
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#BDC3C7', which='major')
    ax.grid(True, alpha=0.1, linestyle=':', linewidth=0.5, color='#BDC3C7', which='minor')
    ax.minorticks_on()
    ax.set_axisbelow(True)
    
    # Styling des ticks
    ax.tick_params(axis='x', which='major', labelsize=11, 
                   colors='#34495E', width=1.2, length=6, pad=8)
    ax.tick_params(axis='y', which='major', labelsize=11, 
                   colors='#34495E', width=1.2, length=6, pad=8)
    
    # ========================================
    # LÉGENDE PROFESSIONNELLE AVEC SEABORN
    # ========================================
    legend = ax.legend(loc='upper left', 
                      frameon=True, 
                      shadow=False,
                      fancybox=True,
                      framealpha=0.97,
                      fontsize=11,
                      edgecolor='#95A5A6',
                      facecolor='white',
                      title='📊 Composantes du Score',
                      title_fontsize=12,
                      ncol=1,
                      labelspacing=1.2,
                      handlelength=2.2,
                      handletextpad=1.2)
    
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_facecolor('#F8F9FA')
    legend.get_title().set_color('#1A1A2E')
    legend.get_title().set_fontweight('bold')
    
    # ========================================
    # BORDURES PROFESSIONNELLES
    # ========================================
    for spine in ax.spines.values():
        spine.set_edgecolor('#34495E')
        spine.set_linewidth(1.5)
        spine.set_alpha(0.8)
    
    # ========================================
    # ZONE D'INFORMATION BAS DE PAGE
    # ========================================
    fig.text(0.02, 0.02, 
             '📌 Les composantes sont normalisées (0-100)',
             ha='left', va='bottom', fontsize=8, style='italic', color='#7F8C8D',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#F0F0F0', 
                      edgecolor='#BDC3C7', linewidth=1, alpha=0.8))
    
    fig.text(0.98, 0.02, 
             'Score = TEB (20%) + CD (30%) + Annuité/CAF (30%) + FDR (20%)',
             ha='right', va='bottom', fontsize=8, style='italic', color='#7F8C8D',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#F0F0F0', 
                      edgecolor='#BDC3C7', linewidth=1, alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    sns.reset_defaults()
    
    return fig

def create_evolution_details_seaborn(df_historical_kpi, commune_name):
    """Graphique avec tous les 4 indicateurs individuels"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Evolution des Indicateurs Detailles', fontsize=16, fontweight='bold')
    
    sns.set_style("whitegrid")
    
    # TEB
    ax = axes[0, 0]
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['TEB Commune (%)'],
            marker='o', linewidth=3, markersize=8, label=commune_name, color='#1f77b4')
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['TEB Strate (%)'],
            marker='s', linewidth=2, markersize=6, linestyle='--', label='Moyenne strate', color='#ff7f0e')
    ax.axhline(y=15, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=10, color='orange', linestyle=':', alpha=0.7)
    ax.set_title('TEB - Taux d\'Epargne Brute', fontweight='bold')
    ax.set_ylabel('TEB (%)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # CD
    ax = axes[0, 1]
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['Annees de Desendettement'],
            marker='o', linewidth=3, markersize=8, label=commune_name, color='#1f77b4')
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['CD Strate (annees)'],
            marker='s', linewidth=2, markersize=6, linestyle='--', label='Moyenne strate', color='#ff7f0e')
    ax.axhline(y=8, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=12, color='red', linestyle=':', alpha=0.7)
    ax.set_title('CD - Capacite de Desendettement', fontweight='bold')
    ax.set_ylabel('Annees', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Annuité/CAF
    ax = axes[1, 0]
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['Annuite/CAF Commune (%)'],
            marker='o', linewidth=3, markersize=8, label=commune_name, color='#1f77b4')
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['Annuite/CAF Strate (%)'],
            marker='s', linewidth=2, markersize=6, linestyle='--', label='Moyenne strate', color='#ff7f0e')
    ax.axhline(y=50, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=60, color='red', linestyle=':', alpha=0.7)
    ax.set_title('Ratio Annuite / CAF Brute', fontweight='bold')
    ax.set_ylabel('Annuite/CAF (%)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # FDR
    ax = axes[1, 1]
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['FDR Jours Commune'],
            marker='o', linewidth=3, markersize=8, label=commune_name, color='#1f77b4')
    ax.plot(df_historical_kpi['Annee'], df_historical_kpi['FDR Jours Moyenne'],
            marker='s', linewidth=2, markersize=6, linestyle='--', label='Moyenne strate', color='#ff7f0e')
    ax.axhline(y=240, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=60, color='red', linestyle=':', alpha=0.7)
    ax.set_title('FDR - Fonds de Roulement', fontweight='bold')
    ax.set_ylabel('FDR (jours)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig





def enhance_figure_quality(fig):
    """Améliore la qualité visuelle d'une figure Plotly - Légende en bas"""
    fig.update_layout(
        # Fond blanc propre
        plot_bgcolor='white',
        paper_bgcolor='white',
        
        # Polices plus lisibles
        font=dict(
            family='Arial, sans-serif',
            size=11,
            color='#1a1a1a'
        ),
        
        # Grille améliorée
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)',
            showline=True,
            linewidth=1.5,
            linecolor='#333333'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)',
            showline=True,
            linewidth=1.5,
            linecolor='#333333'
        ),
        
        # Légende EN BAS - CENTREE
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.15,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#333333',
            borderwidth=1,
            font=dict(size=10)
        ),
        
        # Marges (augmentées en bas pour la légende)
        margin=dict(l=70, r=50, t=80, b=120),
        
        # Hover info
        hovermode='x unified'
    )
    
    # Améliorer les traces selon leur type
    for trace in fig.data:
        if trace.type in ['scatter', 'scatterpolar']:
            # Pour les lignes et scatter
            trace.update(
                line=dict(width=2.5),
                marker=dict(size=8)
            )
        elif trace.type == 'bar':
            # Pour les barres
            trace.update(
                marker=dict(line=dict(width=1, color='white'))
            )
    
    return fig
# === NOUVEAU SYSTÈME DE SCORING V3 (AFFINÉ) ===
def score_sante_financiere_v3(row, df_ref):
    """
    Calcule le score de santé financière avec pondérations (0-100)
    VERSION 3 - PARAMÈTRES AFFINÉS PAR CLIENT
    
    Pondérations :
    - TEB : 20 points (>20% = vert, 10-20% = progressif, <10% = rouge)
    - CD : 30 points (<6 ans = vert, 6-16 ans = progressif, >16 ans = rouge)
    - Ratio Annuité/CAF : 30 points (<30% = vert, 30-50% = progressif, >50% = rouge)
    - FDR en jours : 20 points - NOUVEAUX SEUILS CLIENT :
        * < 60 j : 0 pts (Rouge)
        * 60-120 j : 5 pts (Rouge)
        * 120-180 j : 10 pts (Orange)
        * 180-240 j : 15 pts (Vert)
        * > 240 j : 20 pts (Vert)
    """
    score = 0
    
    # 1. TAUX D'ÉPARGNE BRUTE (TEB) - 20 points
    # Nouveau seuil : 20% pour le vert (au lieu de 15%)
    if pd.notna(row['TEB (%)']):
        if row['TEB (%)'] > 20:
            score += 20  # Vert - plein score
        elif row['TEB (%)'] >= 10:
            # Interpolation linéaire entre 10% et 20%
            score += ((row['TEB (%)'] - 10) / 10) * 20
        else:
            # Sous 10%, score proportionnel (max 0 points)
            score += 0
    
    # 2. CAPACITÉ DE DÉSENDETTEMENT (CD) - 30 points
    # Nouveau : débute à 6 ans (au lieu de 8), zéro à 16 ans (au lieu de 12)
    # 0 ans = pas de dette = meilleur score possible
    if pd.notna(row['Années de Désendettement']) and row['Années de Désendettement'] >= 0:
        if row['Années de Désendettement'] < 6:
            score += 30  # Vert - plein score (inclut 0 = pas de dette)
        elif row['Années de Désendettement'] <= 16:
            # Interpolation linéaire entre 6 et 16 ans
            score += 30 - ((row['Années de Désendettement'] - 6) / 10) * 30
        else:
            # Au-dessus de 16 ans = 0 points
            score += 0
    else:
        # Données manquantes = score neutre
        score += 15
    
    # 3. RATIO ANNUITÉ / CAF BRUTE - 30 points
    # Nouveau : débute à 30% (au lieu de 50%)


    # 3. RATIO ANNUITÉ / CAF BRUTE - 30 points
    if pd.notna(row['Annuité / CAF (%)']):
        annuite_caf_value = row['Annuité / CAF (%)']
        
        # 🔴 CAS 1 : VALEUR NÉGATIVE = DONNÉE ABERRANTE = 0 POINTS
        if annuite_caf_value < 0:
            score += 0  # Pas de points pour les données aberrantes
        
        # 🟢 CAS 2 : VALEUR NULLE = PAS D'ANNUITÉS = PLEIN SCORE (EXCELLENT)
        elif annuite_caf_value == 0:
            score += 30  # Plein score si pas d'annuités
        
        # 🟢 CAS 3 : VALEUR POSITIVE NORMALE
        elif annuite_caf_value < 30:
            score += 30  # Vert - plein score
        elif annuite_caf_value <= 50:
            # Interpolation linéaire entre 30% et 50%
            score += 30 - ((annuite_caf_value - 30) / 20) * 30
        else:
            # Au-dessus de 50% = 0 points
            score += 0
    else:
        # Pas d'annuité = bonne situation (données manquantes)
        score += 30
    
    # 4. FONDS DE ROULEMENT EN JOURS - 20 points
    # ★ INTERPOLATION LINÉAIRE ENTRE 5 PALIERS CLIENTS ★
    if pd.notna(row['FDR Jours Commune']):
        fdr = row['FDR Jours Commune']
        
        if fdr >= 240:
            # >= 240j : 20 pts (Vert excellent)
            score += 20
        elif fdr >= 180:
            # 180-240j : interpolation 15 → 20 pts (Vert)
            score += 15 + ((fdr - 180) / 60) * 5
        elif fdr >= 120:
            # 120-180j : interpolation 10 → 15 pts (Orange)
            score += 10 + ((fdr - 120) / 60) * 5
        elif fdr >= 60:
            # 60-120j : interpolation 5 → 10 pts (Rouge)
            score += 5 + ((fdr - 60) / 60) * 5
        elif fdr > 0:
            # 0-60j : interpolation 0 → 5 pts (Rouge critique)
            score += (fdr / 60) * 5
        else:
            # Zéro ou négatif = 0 points
            score += 0
    else:
        # Données manquantes = score neutre (10 points)
        score += 10
    
    return round(score, 2)


def niveau_alerte_v3(score):
    """Détermine le niveau d'alerte selon le système V3 (inchangé)"""
    if pd.notna(score):
        if score >= 75:
            return "🟢 Vert"
        elif score >= 50:
            return "🟠 Orange"
        else:
            return "🔴 Rouge"
    return "❓ N/A"


# === DOCUMENTATION DES CHANGEMENTS V3 ===



def get_color_alerte(niveau):
    """Retourne la couleur correspondant au niveau"""
    if "Rouge" in niveau:
        return "#FF4B4B"
    elif "Orange" in niveau:
        return "#FF8C00"
    else:
        return "#00C851"

# ============================================================
# SECTION 4 : RADAR COHÉRENT ⭐ PLACER LES FONCTIONS RADAR ICI
# ============================================================

def normaliser_indicateurs_pour_radar(row):
    """
    Normalise les indicateurs sur une échelle cohérente de 0-100
    
    LOGIQUE UNIFORME : 
    - Plus on s'éloigne du CENTRE (0) vers l'EXTÉRIEUR (100) = MIEUX C'EST
    - Tous les critères vont dans le même sens
    
    NOUVELLES PLAGES (réalistes) :
    - TEB : 0-30% (seuil vert à 15%)
    - Années de Désendettement : 0-15 ans (seuil vert < 8 ans)
    - Annuité/CAF : 0-80% (seuil vert < 50%)
    - FDR : 0-240 jours (seuil vert > 240j)
    """
    
    # 1️⃣ TEB (%) - PLAGE 0-30%
    if pd.notna(row['TEB (%)']):
        teb_value = row['TEB (%)']
        if teb_value < 0:
            teb_value = 0  # juste éviter les négatifs
        elif teb_value > 30:
            teb_value = 30
        teb_norm = (teb_value / 30) * 100
    else:
        teb_norm = 0
    
    # 2️⃣ CD - PLAGE 0-15 ANS (INVERSÉE)
    # 0 ans = pas de dette = meilleur score (100%)
    if pd.notna(row['Années de Désendettement']) and row['Années de Désendettement'] >= 0:
        cd_value = min(row['Années de Désendettement'], 15)
        cd_norm = ((15 - cd_value) / 15) * 100
    else:
        cd_norm = 0
    
    # 3️⃣ ANNUITÉ/CAF (%) - PLAGE 0-80% (INVERSÉE) ⭐ CORRIGÉ
    # ★ GESTION EXPLICITE DES NÉGATIFS ★
    if pd.notna(row['Annuité / CAF (%)']):
        annuite_caf_value = row['Annuité / CAF (%)']
        
        # 🔴 SI NÉGATIF = ANOMALIE = NORMALISE À 0 (PIRE)
        if annuite_caf_value < 0:
            annuite_caf_norm = 0  # Score minimum pour les aberrances
        # 🟢 SI ZÉRO = PAS D'ANNUITÉS = EXCELLENT = 100
        elif annuite_caf_value == 0:
            annuite_caf_norm = 100
        else:
            # Valeur positive : clamper à [0, 80]
            annuite_caf_value = min(annuite_caf_value, 80)
            annuite_caf_norm = ((80 - annuite_caf_value) / 80) * 100
    else:
        annuite_caf_norm = 100  # Pas de donnée = supposer bon
    
    # 4️⃣ FDR - PLAGE 0-3240 JOURS
    if pd.notna(row['FDR Jours Commune']):
        fdr_value = min(row['FDR Jours Commune'], 240)
        fdr_norm = (fdr_value / 240) * 100
    else:
        fdr_norm = 50
    
    # 5️⃣ RIGIDITÉ (%) (INVERSÉE)
    if pd.notna(row['Rigidité (%)']):
        rigidite_value = min(row['Rigidité (%)'], 200)
        rigidite_norm = ((200 - rigidite_value) / 200) * 100
    else:
        rigidite_norm = 50
    
    return {
        'TEB_norm': round(teb_norm, 2),
        'CD_norm': round(cd_norm, 2),
        'Annuité_CAF_norm': round(annuite_caf_norm, 2),
        'FDR_norm': round(fdr_norm, 2),
        'Rigidité_norm': round(rigidite_norm, 2)
    }


def create_radar_coherent(commune_data, df_filtered=None):
    """
    Crée un radar COHÉRENT avec plages réalistes
    DIRECTION UNIFORME : Vers l'EXTÉRIEUR = MIEUX
    """
    
    norms = normaliser_indicateurs_pour_radar(commune_data)
    
    categories = [
        'TEB (%) 0-30%',
        'Années Désendettement 0-15 ans',
        'Annuité/CAF (%) 0-80%',
        'FDR (jours) 0-240j',
        'Rigidité (%) inversion 0-200%'
    ]
    
    values_commune = [
        norms['TEB_norm'],
        norms['CD_norm'],
        norms['Annuité_CAF_norm'],
        norms['FDR_norm'],
        norms['Rigidité_norm']
    ]
    
    # Seuils vert normalisés
    seuils_vert = [
        (15 / 30) * 100,              # TEB : 50
        ((15 - 8) / 15) * 100,        # CD : 46.67
        ((80 - 50) / 80) * 100,       # Annuité : 37.5
        (240 / 300) * 100,            # FDR : 80
        ((200 - 100) / 200) * 100     # Rigidité : 50
    ]
    
    fig = go.Figure()
    
    # Trace commune
    fig.add_trace(go.Scatterpolar(
        r=values_commune,
        theta=categories,
        fill='toself',
        name=commune_data['Commune'],
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8),
        fillcolor='rgba(59, 130, 246, 0.25)'
    ))
    
    # Trace seuils vert
    fig.add_trace(go.Scatterpolar(
        r=seuils_vert,
        theta=categories,
        fill=None,
        name='Seuil Vert',
        line=dict(color='#10b981', width=2, dash='dash'),
        marker=dict(size=6),
    ))
    
    # Trace moyenne strate
    if df_filtered is not None and not df_filtered.empty:
        moyennes_strate = df_filtered.apply(normaliser_indicateurs_pour_radar, axis=1).apply(pd.Series).mean()
        
        values_strate = [
            moyennes_strate['TEB_norm'],
            moyennes_strate['CD_norm'],
            moyennes_strate['Annuité_CAF_norm'],
            moyennes_strate['FDR_norm'],
            moyennes_strate['Rigidité_norm']
        ]
        # Clamp final AVANT le radar
        import numpy as np

        values_commune = np.clip(values_commune, 0, 100)
        values_strate = np.clip(values_strate, 0, 100)

        
        fig.add_trace(go.Scatterpolar(
            r=values_strate,
            theta=categories,
            fill='toself',
            name='Moyenne Strate',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            marker=dict(size=6),
            fillcolor='rgba(245, 158, 11, 0.15)'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='outside',
                tickfont=dict(size=10),
                gridcolor='rgba(243, 244, 246, 0.5)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        title=dict(
            text=f"<b>🎯 Profil Financier Cohérent</b><br><sub>{commune_data['Commune']} | Score: {commune_data['Score']:.0f}/100</sub>",
            font=dict(size=14)
        ),
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        font=dict(size=12),
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    fig.add_annotation(
        text="<b>📌 Logique uniforme :</b> Plus vers l'extérieur = Mieux ✅<br>Plus vers le centre = Pire ❌",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=11, color="#666"),
        align="center"
    )
    
    return fig

def create_radar_seaborn(commune_data, df_filtered=None):
    """
    Crée un radar cohérent en Matplotlib/Seaborn pour le PDF
    DIRECTION UNIFORME : Vers l'EXTERIEUR = MIEUX
    """
    
    norms = normaliser_indicateurs_pour_radar(commune_data)
    
    categories = [
        'TEB (%)\n0-30%',
        'Annees Desendettement\n0-15 ans',
        'Annuite/CAF (%)\n0-80%',
        'FDR (jours)\n0-240j',
        'Rigidite (%)\n0-200%'
    ]
    
    values_commune = [
        norms['TEB_norm'],
        norms['CD_norm'],
        norms['Annuite_CAF_norm'],
        norms['FDR_norm'],
        norms['Rigidite_norm']
    ]
    
    # Seuils vert normalises
    seuils_vert = [
        (15 / 30) * 100,              # TEB : 50
        ((15 - 8) / 15) * 100,        # CD : 46.67
        ((80 - 50) / 80) * 100,       # Annuite : 37.5
        (240 / 300) * 100,            # FDR : 80
        ((200 - 100) / 200) * 100     # Rigidite : 50
    ]
    
    # Calculer moyenne strate si disponible
    values_strate = None
    if df_filtered is not None and not df_filtered.empty:
        try:
            moyennes_strate = df_filtered.apply(normaliser_indicateurs_pour_radar, axis=1).apply(pd.Series).mean()
            values_strate = [
                moyennes_strate['TEB_norm'],
                moyennes_strate['CD_norm'],
                moyennes_strate['Annuite_CAF_norm'],
                moyennes_strate['FDR_norm'],
                moyennes_strate['Rigidite_norm']
            ]
        except:
            values_strate = None
    
    # === CREATION DU RADAR ===
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values_commune += values_commune[:1]  # Fermer le polygone
    angles += angles[:1]
    
    if values_strate:
        values_strate += values_strate[:1]
    
    seuils_vert_closed = seuils_vert + seuils_vert[:1]
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor('white')
    
    # Grille de fond
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Trace COMMUNE
    ax.plot(angles, values_commune, 'o-', linewidth=3, markersize=8,
            label=commune_data['Commune'], color='#1f77b4')
    ax.fill(angles, values_commune, alpha=0.25, color='#1f77b4')
    
    # Trace STRATE
    if values_strate:
        ax.plot(angles, values_strate, 's--', linewidth=2, markersize=6,
                label='Moyenne Strate', color='#ff7f0e', alpha=0.8)
        ax.fill(angles, values_strate, alpha=0.1, color='#ff7f0e')
    
    # Trace SEUIL VERT
    ax.plot(angles, seuils_vert_closed, ':', linewidth=2.5,
            label='Seuil Vert', color='#10b981', alpha=0.8)
    
    # Etiquettes des axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10, weight='bold')
    
    # Grille radiale
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legende
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10,
              framealpha=0.95, edgecolor='#cccccc')
    
    # Titre
    score = commune_data.get('Score', 0)
    fig.suptitle('Profil Financier Coherent', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.94, f"{commune_data['Commune']} | Score: {score:.1f}/100",
             ha='center', fontsize=11, color='#666666')
    
    # Note explicative
    fig.text(0.5, 0.02,
             'Logique uniforme: Plus vers l\'exterieur = Mieux | Plus vers le centre = Pire',
             ha='center', fontsize=9, color='#999999', style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    return fig

def create_score_evolution_stacked_bar_seaborn(df_historical_kpi, commune_name):

    """
    Crée un graphique en barres empilées avec design expert pour PDF
    """
    if df_historical_kpi.empty or len(df_historical_kpi) < 2:
        return None
    
    df = df_historical_kpi.sort_values('Année').reset_index(drop=True)
    df['Année'] = pd.to_numeric(df['Année'], errors='coerce')
    
    # ========================================
    # CALCUL DES COMPOSANTES EN POINTS
    # ========================================
    teb_scores = []
    cd_scores = []
    annuite_scores = []
    fdr_scores = []
    
    for _, row in df.iterrows():
        # TEB (max 20 pts)
        if pd.notna(row['TEB (%)']):
            if row['TEB (%)'] > 20:
                teb_scores.append(20)
            elif row['TEB (%)'] >= 10:
                teb_scores.append(((row['TEB (%)'] - 10) / 10) * 20)
            else:
                teb_scores.append(0)
        else:
            teb_scores.append(0)
        
        # CD (max 30 pts)
        if pd.notna(row.get('Années de Désendettement')) and row.get('Années de Désendettement') >= 0:
            cd_value = row.get('Années de Désendettement')
            if cd_value < 6:
                cd_scores.append(30)
            elif cd_value <= 16:
                cd_scores.append(30 - ((cd_value - 6) / 10) * 30)
            else:
                cd_scores.append(0)
        else:
            cd_scores.append(15)
        
        # Annuité/CAF (max 30 pts)
        annuite_caf = row.get('Annuité / CAF (%)')
        if pd.notna(annuite_caf):
            if annuite_caf < 0:
                # 🔴 ANNUITÉ/CAF NÉGATIF = 0 POINTS (VALEUR ABERRANTE)
                annuite_scores.append(0)
            elif annuite_caf == 0:
                # 🟢 ZÉRO = PAS D'ANNUITÉS = PLEIN SCORE
                annuite_scores.append(30)
            elif annuite_caf < 30:
                annuite_scores.append(30)
            elif annuite_caf <= 50:
                annuite_scores.append(30 - ((annuite_caf - 30) / 20) * 30)
            else:
                annuite_scores.append(0)
        else:
            # Pas de données = score optimiste
            annuite_scores.append(30)
        
        # FDR (max 20 pts)
        if pd.notna(row.get('FDR Jours Commune')):
            fdr_jours = row.get('FDR Jours Commune')
            if fdr_jours > 240:
                fdr_scores.append(20)
            elif fdr_jours >= 70:
                fdr_scores.append(((fdr_jours - 70) / 170) * 20)
            elif fdr_jours >= 30:
                fdr_scores.append(((fdr_jours - 30) / 40) * 10)
            else:
                fdr_scores.append(0)
        else:
            fdr_scores.append(10)
    
    # ========================================
    # CONFIGURATION STYLE EXPERT
    # ========================================
    sns.set_theme(style="whitegrid", palette="Set2")
    
    custom_style = {
        'grid.linestyle': '--',
        'grid.alpha': 0.15,
        'axes.edgecolor': '#1a1a1a',
        'axes.linewidth': 1.2,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.bottom': False,
        'ytick.left': False,
    }
    sns.set_style("whitegrid", custom_style)
    
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#FAFBFC')
    
    # ========================================
    # PARAMÈTRES DES BARRES
    # ========================================
    x = df['Année'].values
    width = 0.65
    
    # Palette de couleurs harmonieuse et optimisée pour l'impression
    colors = {
        'fdr': '#C0392B',      # Bleu clair
        'cd': '#27AE60',       # Vert professionnel
        'annuite': '#E67E22',  # Orange professionnel
        'teb': '#5DADE2'       # Rouge professionnel
    }
    
    # ========================================
    # TRACÉ DES BARRES EMPILÉES
    # ========================================
    # FDR (base)
    p1 = ax.bar(x, fdr_scores, width, 
                label='FDR (20 pts)', 
                color=colors['fdr'],
                edgecolor='white',
                linewidth=1.5,
                zorder=3)
    
    # CD
    p2 = ax.bar(x, cd_scores, width, 
                bottom=fdr_scores,
                label='CD (30 pts)', 
                color=colors['cd'],
                edgecolor='white',
                linewidth=1.5,
                zorder=3)
    
    # Annuité/CAF
    bottom_annuite = [fdr_scores[i] + cd_scores[i] for i in range(len(x))]
    p3 = ax.bar(x, annuite_scores, width, 
                bottom=bottom_annuite,
                label='Annuité/CAF (30 pts)', 
                color=colors['annuite'],
                edgecolor='white',
                linewidth=1.5,
                zorder=3)
    
    # TEB
    bottom_teb = [bottom_annuite[i] + annuite_scores[i] for i in range(len(x))]
    p4 = ax.bar(x, teb_scores, width, 
                bottom=bottom_teb,
                label='TEB (20 pts)', 
                color=colors['teb'],
                edgecolor='white',
                linewidth=1.5,
                zorder=3)
    
    # ========================================
    # ANNOTATIONS DES VALEURS TOTALES
    # ========================================
    for i, année in enumerate(x):
        total = df.iloc[i]['Score Commune']
        
        # Déterminer la couleur du texte selon le score
        if total >= 75:
            text_color = '#00C851'
            bg_color = '#E8F8F5'
        elif total >= 50:
            text_color = '#F39C12'
            bg_color = '#FEF5E7'
        else:
            text_color = '#E74C3C'
            bg_color = '#FADBD8'
        
        # Annotation en haut de chaque barre
        ax.text(année, total + 2, f'{total:.1f}/100',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
                color=text_color,
                bbox=dict(boxstyle='round,pad=0.5',
                         facecolor=bg_color,
                         edgecolor=text_color,
                         linewidth=1.8,
                         alpha=0.95))
    
    # ========================================
    # LIGNES DE REPÈRE HORIZONTALES
    # ========================================
    ax.axhline(y=100, color='#34495E', linestyle='-', linewidth=2.5, 
               alpha=0.3, zorder=1, label='Score max (100 pts)')
    ax.axhline(y=75, color='#00C851', linestyle='--', linewidth=2.2, 
               alpha=0.5, zorder=2)
    ax.axhline(y=50, color='#F39C12', linestyle='--', linewidth=2.2, 
               alpha=0.5, zorder=2)
    
    # ========================================
    # ZONES DE COULEUR DE FOND
    # ========================================
    ax.axhspan(75, 105, facecolor='#00C851', alpha=0.04, zorder=0)
    ax.axhspan(50, 75, facecolor='#FFB84D', alpha=0.04, zorder=0)
    ax.axhspan(0, 50, facecolor='#FF6B6B', alpha=0.04, zorder=0)
    
    # ========================================
    # TITRES ET LABELS EXPERT
    # ========================================
    ax.set_title(f"📊 Composition du Score de Santé Financière par Année\n{commune_name} | Détail des Points par Composante", 
                 fontsize=17, fontweight='bold', pad=25,
                 color='#1A1A2E', loc='left')
    
    ax.set_xlabel("Année", fontsize=12, fontweight='600', color='#34495E', labelpad=12)
    ax.set_ylabel("Points / 100", fontsize=12, fontweight='600', color='#34495E', labelpad=12)
    
    # ========================================
    # CONFIGURATION AXES AVANCÉE
    # ========================================
    ax.set_ylim(0, 110)
    ax.set_xlim(df['Année'].min() - 0.7, df['Année'].max() + 0.7)
    
    # Ticks Y
    y_ticks = [0, 25, 50, 75, 100]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0', '25', '50', '75', '100'], 
                       fontsize=11, fontweight='600', color='#34495E')
    
    # Ticks X
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(année)) for année in x], 
                       fontsize=11, fontweight='600', color='#34495E')
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8, 
            color='#BDC3C7', axis='y', which='major', zorder=0)
    ax.grid(True, alpha=0.08, linestyle=':', linewidth=0.5, 
            color='#BDC3C7', axis='y', which='minor')
    ax.minorticks_on()
    ax.set_axisbelow(True)
    
    # Styling des ticks
    ax.tick_params(axis='both', which='major', 
                   colors='#34495E', width=1.2, length=6, pad=8)
    
    # ========================================
    # LÉGENDE PROFESSIONNELLE EN BAS
    # ========================================
    legend = ax.legend(loc='lower center',
                      frameon=True,
                      shadow=False,
                      fancybox=True,
                      framealpha=0.97,
                      fontsize=11,
                      edgecolor='#95A5A6',
                      facecolor='white',
                      title='📋 Composantes du Score',
                      title_fontsize=12,
                      ncol=4,
                      labelspacing=1.2,
                      handlelength=2,
                      handletextpad=1,
                      bbox_to_anchor=(0.5, -0.18))
    
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_facecolor('#F8F9FA')
    legend.get_title().set_color('#1A1A2E')
    legend.get_title().set_fontweight('bold')
    
    # ========================================
    # BORDURES PROFESSIONNELLES
    # ========================================
    for spine in ax.spines.values():
        spine.set_edgecolor('#34495E')
        spine.set_linewidth(1.5)
        spine.set_alpha(0.8)
    
    # ========================================
    # ANNOTATIONS INFORMATIVES
    # ========================================
    # Note de gauche
    fig.text(0.02, 0.01,
             '📌 Chaque année affiche le score total /100 en haut de la barre',
             ha='left', va='bottom', fontsize=8, style='italic', 
             color='#7F8C8D',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#F0F0F0',
                      edgecolor='#BDC3C7', linewidth=1, alpha=0.8))
    
    # Note de droite
    fig.text(0.98, 0.01,
             '✓ Vert (75+) | ⚠ Orange (50-75) | ✗ Rouge (<50)',
             ha='right', va='bottom', fontsize=8, style='italic',
             color='#7F8C8D',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#F0F0F0',
                      edgecolor='#BDC3C7', linewidth=1, alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    sns.reset_defaults()
    
    return fig


def create_tableau_normalisation(commune_data):
    """
    Crée un tableau montrant les AVANT/APRÈS normalisation
    ✅ VERSION FINALE AVEC TOUS LES MESSAGES D'ALERTE
    """
    try:
        # ========================================
        # HELPER FUNCTION pour accéder aux valeurs
        # ========================================
        def get_value(key):
            try:
                if isinstance(commune_data, pd.Series):
                    return commune_data[key] if key in commune_data.index else None
                else:
                    return commune_data.get(key)
            except:
                return None
        
        # ========================================
        # 1. TEB - Garder la valeur (même si négatif)
        # ========================================
        try:
            teb_val = get_value('TEB (%)')
            teb_brute_display = f"{teb_val:.1f}%" if pd.notna(teb_val) else "N/A"
        except:
            teb_brute_display = "N/A"
        
        # ========================================
        # 2. CD - Gestion des négatifs
        # ========================================
        cd_commune = get_value('Années de Désendettement')
        teb_brute = get_value('TEB (%)')
        encours_brut = get_value('Encours (K€)')
        caf_brute = get_value('Épargne brute (K€)')
        
        # SI TEB < 0, calculer le CD brut
        if pd.notna(teb_brute) and teb_brute < 0 and pd.notna(encours_brut) and encours_brut > 0 and pd.notna(caf_brute) and caf_brute != 0:
            cd_brut = encours_brut / caf_brute
            cd_valeur_brute = f"🔴 Supérieur à 15 ans"
        # SI CD est déjà négatif
        elif pd.notna(cd_commune) and cd_commune < 0:
            cd_valeur_brute = f"🔴 Supérieur à 15 ans"
        else:
            cd_valeur_brute = f"{cd_commune:.1f} ans" if pd.notna(cd_commune) else "N/A"
        
        # ========================================
        # 3. ANNUITÉ/CAF - Gestion des négatifs
        # ========================================
        try:
            annuite_val = get_value('Annuité / CAF (%)')
            if pd.notna(annuite_val) and annuite_val < 0:
                annuite_brute = f"🔴 Supérieur à 100%"
            else:
                annuite_brute = f"{annuite_val:.1f}%" if pd.notna(annuite_val) else "N/A"
        except:
            annuite_brute = "N/A"
        
        # ========================================
        # 4. FDR - Normal (pas de négatifs attendus)
        # ========================================
        try:
            fdr_val = get_value('FDR Jours Commune')
            fdr_brute = f"{fdr_val:.0f}j" if pd.notna(fdr_val) else "N/A"
        except:
            fdr_brute = "N/A"
        
        # ========================================
        # 5. RIGIDITÉ - Normal
        # ========================================
        try:
            rigidite_val = get_value('Rigidité (%)')
            rigidite_brute = f"{rigidite_val:.1f}%" if pd.notna(rigidite_val) else "N/A"
        except:
            rigidite_brute = "N/A"
        
        # ========================================
        # NORMALISATION
        # ========================================
        norms = normaliser_indicateurs_pour_radar(commune_data)
        
        # ========================================
        # CRÉATION DU TABLEAU
        # ========================================
        tableau = pd.DataFrame({
            'Critère': [
                'TEB (%)',
                'Années Désendettement',
                'Annuité/CAF (%)',
                'FDR (jours)',
                'Rigidité (%)'
            ],
            'Valeur Brute': [
                teb_brute_display,
                cd_valeur_brute,
                annuite_brute,
                fdr_brute,
                rigidite_brute
            ],
            'Plage': [
                '0-30%',
                '0-15 ans',
                '0-80%',
                '0-240j',
                '0-200%'
            ],
            'Normalisé (0-100)': [
                f"{norms['TEB_norm']:.1f}",
                f"{norms['CD_norm']:.1f}",
                f"{norms['Annuité_CAF_norm']:.1f}",
                f"{norms['FDR_norm']:.1f}",
                f"{norms['Rigidité_norm']:.1f}"
            ]
        })
        
        return tableau
    
    except Exception as e:
        st.error(f"❌ Erreur dans create_tableau_normalisation: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# === À UTILISER ===


# --- Fonction pour créer les tranches de population ---
def create_population_brackets(df):
    """Crée des tranches de population"""
    df['Tranche pop'] = pd.cut(df['Population'],
                           bins=[0, 100, 200, 500, 2000, 3500, 5000, 10000, 25000, float('inf')],
                           labels=['Moins de 100 hab',
                                   '100-200 hab',
                                   '200-500 hab',
                                   '500-2000 hab',
                                   '2000-3500 hab',
                                   '3500-5000 hab',
                                   '5000-10000 hab',
                                   '10000-25000 hab',
                                   '+ 25000 hab'])
    return df

# ============================================================
# SECTION À INTÉGRER DANS CLAUDE.PY
# Placer APRÈS la fonction create_tableau_normalisation()
# et AVANT "=== RÉCUPÉRATION ET TRAITEMENT DES DONNÉES ==="
# ============================================================

def generate_pdf_graphs(df_historical_kpi, commune_name, commune_data, df_filtered):
    """
    Génère tous les graphiques pour le PDF et retourne la liste des fichiers temporaires
    """
    try:
        # === ÉTAPE 1 : Générer les graphiques en PNG ===
        temp_images = []
        
       # Radar plot (analyse détaillée)
        fig_radar = create_radar_plot_matplotlib(commune_data, df_filtered)
        if fig_radar:
            temp_img_radar = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig_radar.savefig(temp_img_radar.name, dpi=300, bbox_inches='tight')
            plt.close(fig_radar)
            temp_images.append(('radar', temp_img_radar.name))

        # Score global
        fig_score = create_score_evolution_chart_seaborn(df_historical_kpi, commune_name)
        if fig_score:
            temp_img1 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig_score.savefig(temp_img1.name, dpi=300, bbox_inches='tight')
            plt.close(fig_score)
            temp_images.append(('score', temp_img1.name))

        # Stacked bar
        fig_stacked = create_score_evolution_stacked_bar_seaborn(df_historical_kpi, commune_name)
        if fig_stacked:
            temp_img2 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig_stacked.savefig(temp_img2.name, dpi=300, bbox_inches='tight')
            plt.close(fig_stacked)
            temp_images.append(('stacked', temp_img2.name))

        # Lignes
        fig_lines = create_score_evolution_lines_seaborn(df_historical_kpi, commune_name)
        if fig_lines:
            temp_img3 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig_lines.savefig(temp_img3.name, dpi=300, bbox_inches='tight')
            plt.close(fig_lines)
            temp_images.append(('lines', temp_img3.name))
        
        # ✨ GRAPHIQUES INDIVIDUELS UN PAR UN (PAS DE GRID)
        
        # 1. TEB individuel
        fig_teb_ind, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['TEB Commune (%)'],
                marker='o', linewidth=3, markersize=10, label=commune_name, color='#1f77b4')
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['TEB Strate (%)'],
                marker='s', linewidth=2, markersize=8, linestyle='--', label='Moy. strate', color='#ff7f0e')
        ax.axhline(y=15, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 15.5, 'Seuil bon (15%)', color='green', fontsize=10)
        ax.axhline(y=10, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 10.5, 'Seuil critique (10%)', color='orange', fontsize=10)
        ax.set_title('📈 Évolution du Taux d\'Épargne Brute (TEB)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Année', fontsize=12, fontweight='bold')
        ax.set_ylabel('TEB (%)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        temp_file_teb = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig_teb_ind.savefig(temp_file_teb.name, dpi=300, bbox_inches='tight')
        plt.close(fig_teb_ind)
        temp_images.append(('teb_ind', temp_file_teb.name))
        
        # 2. CD individuel
        fig_cd_ind, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['Années de Désendettement'],
                marker='o', linewidth=3, markersize=10, label=commune_name, color='#1f77b4')
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['CD Strate (années)'],
                marker='s', linewidth=2, markersize=8, linestyle='--', label='Moy. strate', color='#ff7f0e')
        ax.axhline(y=8, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 8.5, 'Seuil bon (8 ans)', color='green', fontsize=10)
        ax.axhline(y=12, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 12.5, 'Seuil critique (12 ans)', color='red', fontsize=10)
        ax.set_title('⏳ Évolution de la Capacité de Désendettement', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Année', fontsize=12, fontweight='bold')
        ax.set_ylabel('Capacité (années)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        temp_file_cd = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig_cd_ind.savefig(temp_file_cd.name, dpi=300, bbox_inches='tight')
        plt.close(fig_cd_ind)
        temp_images.append(('cd_ind', temp_file_cd.name))
        
        # 3. Annuité/CAF individuel
        fig_annuite_ind, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['Annuité/CAF Commune (%)'],
                marker='o', linewidth=3, markersize=10, label=commune_name, color='#1f77b4')
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['Annuité/CAF Strate (%)'],
                marker='s', linewidth=2, markersize=8, linestyle='--', label='Moy. strate', color='#ff7f0e')
        ax.axhline(y=50, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 50.5, 'Seuil bon (50%)', color='green', fontsize=10)
        ax.axhline(y=60, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 60.5, 'Seuil critique (60%)', color='red', fontsize=10)
        ax.set_title('💳 Évolution du Ratio Annuité/CAF Brute', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Année', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annuité/CAF (%)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        temp_file_annuite = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig_annuite_ind.savefig(temp_file_annuite.name, dpi=300, bbox_inches='tight')
        plt.close(fig_annuite_ind)
        temp_images.append(('annuite_ind', temp_file_annuite.name))
        
        # 4. FDR individuel
        fig_fdr_ind, ax = plt.subplots(figsize=(14, 7))
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['FDR Jours Commune'],
                marker='o', linewidth=3, markersize=10, label=commune_name, color='#1f77b4')
        ax.plot(df_historical_kpi['Année'], df_historical_kpi['FDR Jours Moyenne'],
                marker='s', linewidth=2, markersize=8, linestyle='--', label='Moy. strate', color='#ff7f0e')
        ax.axhline(y=240, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 245, 'Seuil bon (240j)', color='green', fontsize=10)
        ax.axhline(y=60, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(df_historical_kpi['Année'].min(), 65, 'Seuil critique (60j)', color='red', fontsize=10)
        ax.set_title('👥 Évolution du Fonds de Roulement', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Année', fontsize=12, fontweight='bold')
        ax.set_ylabel('FDR (jours de DRF)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        temp_file_fdr = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        fig_fdr_ind.savefig(temp_file_fdr.name, dpi=300, bbox_inches='tight')
        plt.close(fig_fdr_ind)
        temp_images.append(('fdr_ind', temp_file_fdr.name))
        
        # ✅ RETOURNER LA LISTE DES IMAGES
        return temp_images
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération des graphiques : {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


def create_financial_summary_table_exact(df_historical_kpi):
    """
    Crée le tableau récapitulatif EXACTEMENT comme demandé
    
    Ordre des colonnes : 2024 2023 2022 2021 2020 2019
    
    Ordre des indicateurs :
    1. RRF prod
    2. DRF charge
    3. CAF Brute caf
    4. TEB caf/prod (%)
    5. CRD dette (Encours)
    6. Désendettement E/C (Années)
    7. Annuité annu
    8. Consommation de la CAF par les emprunts annu/caf (%)
    9. FdR fdr (jours)
    10. FdR Normatif (fdr*charge)/365
    """
    
    if df_historical_kpi.empty:
        return None
    
    # Trier par année croissante
    df_sorted = df_historical_kpi.sort_values('Année', ascending=True).reset_index(drop=True)
    
    # Créer l'en-tête avec les années
    header = ['Indicateur']
    annees = []
    for _, row in df_sorted.iterrows():
        annee = int(row['Année']) if pd.notna(row['Année']) else 'N/A'
        annees.append(annee)
        header.append(f"{annee}")
    
    tableau_data = [header]
    
    # ========================================
    # 1. RRF prod (Recettes Réelles de Fonctionnement) - en K€
    # ========================================
    row_rrf = ['Produits retraités des 013']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('RRF (K€)')):
            row_rrf.append(f"{row['RRF (K€)']:,.0f}")
        else:
            row_rrf.append('N/A')
    tableau_data.append(row_rrf)
    
    # ========================================
    # 2. DRF charge (Dépenses Réelles de Fonctionnement) - en K€
    # ========================================
    row_drf = ['Charges retraitées des 014']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('DRF (K€)')):
            row_drf.append(f"{row['DRF (K€)']:,.0f}")
        else:
            row_drf.append('N/A')
    tableau_data.append(row_drf)
    
    # ========================================
    # 3. CAF Brute caf (Épargne Brute) - en K€
    # ========================================
    row_caf = ['CAF Brute']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('Caf brute (K€)')):
            row_caf.append(f"{row['Caf brute (K€)']:,.0f}")
        else:
            row_caf.append('N/A')
    tableau_data.append(row_caf)
    
    # ========================================
    # 4. TEB caf/prod (%) - Taux d'Épargne Brute
    # Formule : CAF / RRF * 100
    # ========================================
    row_teb = ['TEB']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('TEB Commune (%)')):
            teb_value = row['TEB Commune (%)']
            row_teb.append(f"{teb_value:.1f}%")
        else:
            row_teb.append('N/A')
    tableau_data.append(row_teb)
    
    # ========================================
    # 5. CRD dette (Encours - Capacité de Remboursement de la Dette) - en K€
    # ========================================
    row_encours = ['Capital Restant Dû']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('Encours (K€)')):
            row_encours.append(f"{row['Encours (K€)']:,.0f}")
        else:
            row_encours.append('N/A')
    tableau_data.append(row_encours)
    
    # ========================================
    # 6. Désendettement E/C (Années) - Années de Désendettement
    # Formule : Encours / CAF Brute
    # ========================================
    row_cd = ['Année(s) de Désendettement']
    for _, row in df_sorted.iterrows():
        encours = row.get('Encours (K€)')
        caf_brute = row.get('Caf brute (K€)')
        
        if pd.notna(encours) and pd.notna(caf_brute) and caf_brute != 0:
            cd_value = encours / caf_brute
            row_cd.append(f"{cd_value:.2f}")
        else:
            row_cd.append('N/A')
    tableau_data.append(row_cd)

    # ========================================
    # 7. Annuité annu - Annuité annuelle - en K€
    # ========================================
    row_annuite = ['Annuité']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('Annuité (K€)')):
            row_annuite.append(f"{row['Annuité (K€)']:,.0f}")
        else:
            row_annuite.append('N/A')
    tableau_data.append(row_annuite)
    
    # ========================================
    # 8. Consommation de la CAF par les emprunts annu/caf (%)
    # Formule : Annuité / CAF Brute * 100
    # ========================================
    row_annuite_caf = ['Annuité / CAF Brute']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('Annuité/CAF Commune (%)')):
            annuite_caf_value = row['Annuité/CAF Commune (%)']
            row_annuite_caf.append(f"{annuite_caf_value:.1f}%")
        else:
            row_annuite_caf.append('N/A')
    tableau_data.append(row_annuite_caf)
    
    # ========================================
    # 9. FdR fdr - Fonds de Roulement - Montant en K€
    # Formule : (FDR jours * DRF) / 365
    # ========================================
    row_fdr = ['Fonds de Roulement']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('FDR (K€)')):
            row_fdr.append(f"{row['FDR (K€)']:,.0f}")
        else:
            row_fdr.append('N/A')
    tableau_data.append(row_fdr)
    
    # ========================================
    # 10. FdR Normatif - Fonds de Roulement Normatif - en jours
    # Formule : FDR jours
    # ========================================
    row_fdr_normatif = ['FdR Normatif\n(Nb jrs Charges retraitées)']
    for _, row in df_sorted.iterrows():
        if pd.notna(row.get('FDR Jours Commune')):
            fdr_value = row['FDR Jours Commune']
            # Colorer selon le seuil : > 240 jours = vert, 60-240 jours = orange, < 60 jours = rouge
            if fdr_value > 240:
                color = '🟢'
            elif fdr_value > 60:
                color = '🟠'
            else:
                color = '🔴'
            row_fdr_normatif.append(f"{color} {fdr_value:.0f}j")
        else:
            row_fdr_normatif.append('N/A')
    tableau_data.append(row_fdr_normatif)
    
    return tableau_data






# ✅ IMPORTER PLOTLY GRAPH OBJECTS (vérifier qu'il n'est pas déjà importé)
import plotly.graph_objects as go

# ============================================================
# COPIER/COLLER TOUT CECI : enhanced_pdf_export.py (COMPLÈTE)
# ============================================================

import plotly.io as pio
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib import colors

def add_header_footer(canvas, doc):
    """Ajoute en-tête et pied de page à chaque page"""
    canvas.saveState()
    
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1*cm, A4[1] - 0.7*cm, "Analyse Financière des Communes - SFP COLLECTIVITÉS")
    canvas.drawString(1*cm, 0.5*cm, f"Page {doc.page}")
    canvas.drawRightString(A4[0] - 1*cm, 0.5*cm, datetime.now().strftime('%d/%m/%Y'))
    
    canvas.setStrokeColor(rl_colors.HexColor('#1f77b4'))
    canvas.setLineWidth(0.5)
    canvas.line(1*cm, A4[1] - 0.9*cm, A4[0] - 1*cm, A4[1] - 0.9*cm)
    canvas.line(1*cm, 0.7*cm, A4[0] - 1*cm, 0.7*cm)
    
    canvas.restoreState()

def export_commune_analysis_to_pdf_enhanced(commune_data, df_historical_kpi, commune_name, dept_selection, annee_selection, df_filtered):
    """
    Exporte un PDF professionnel avec :
    1. Page de garde
    2. ANALYSE DÉTAILLÉE (année actuelle)
    4. Évolution pluriannuelle
    5. Conclusions
    """
    try:
        temp_images = generate_pdf_graphs(df_historical_kpi, commune_name, commune_data, df_filtered)

        from io import BytesIO
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY   

         # === CREATION PDF ===
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        styles = getSampleStyleSheet()
        
        # === STYLES ===
        style_titre = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=32,
            textColor=rl_colors.HexColor('#1a1a1a'),
            spaceAfter=10,
            alignment=TA_CENTER,
            leading=38
        )
        
        style_titre_light = ParagraphStyle(
            'CustomTitleLight',
            parent=styles['Heading1'],
            fontName='Helvetica',
            fontSize=32,
            textColor=rl_colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            leading=38
        )
        
        style_sous_titre = ParagraphStyle(
            'SubTitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=rl_colors.HexColor('#666666'),
            spaceAfter=50,
            alignment=TA_CENTER,
            letterSpacing=2
        )
        
        style_section = ParagraphStyle(
            'Section',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            textColor=rl_colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            spaceBefore=30
        )
        
        style_body = ParagraphStyle(
            'BodyText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            textColor=rl_colors.HexColor('#333333'),
            spaceAfter=12,
            leading=14,
            alignment=TA_JUSTIFY
        )
        
        story = []
        
        # ========================================
        # PAGE 1 : COUVERTURE
        # ========================================
        
        story.append(Spacer(1, 4*cm))
        story.append(Paragraph("<b>SANTE FINANCIERE</b>", style_titre_light))
        story.append(Paragraph("DES COMMUNES", style_titre))
        story.append(Spacer(1, 0.8*cm))

        logo_path = "logo.png"  # Adapter le chemin

        if os.path.exists(logo_path):
            try:
                # Créer le logo avec dimensions adaptées
                logo_img = Image(logo_path, width=10*cm, height=5*cm)
                
                # Centrer le logo dans une table
                logo_table = Table([[logo_img]], colWidths=[16*cm])
                logo_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                story.append(logo_table)
                story.append(Spacer(1, 1*cm))
            except Exception as e:
                st.warning(f"⚠️ Logo non chargé : {e}")
                story.append(Spacer(1, 1*cm))
        else:
            st.warning(f"⚠️ Logo non trouvé à : {logo_path}")
            story.append(Spacer(1, 1*cm))
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("RAPPORT D'ANALYSE FINANCIERE - SCORING V3", style_sous_titre))
        story.append(Paragraph("© 2025 SFP COLLECTIVITÉS.\nReproduction intégrale ou partielle interdite sauf autorisation écrite.\nFichier protégé — toute modification est strictement interdite", style_sous_titre))
        
        story.append(PageBreak())
        
        story.append(Spacer(1, 2*cm))
        story.append(Paragraph(f"<b>{commune_name.upper()}</b>", ParagraphStyle(
            'Commune',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=rl_colors.HexColor('#1a1a1a'),
            alignment=TA_CENTER,
            spaceAfter=10
        )))
        
        story.append(Paragraph(f"Departement {dept_selection}", ParagraphStyle(
            'Dept',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            textColor=rl_colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=10
        )))
        
        story.append(Paragraph(f"Année {annee_selection}", ParagraphStyle(
            'Année',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=12,
            textColor=rl_colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=60
        )))
        
        # Score box
        status_color = '#51CF66' if commune_data['Score'] >= 75 else '#FFB84D' if commune_data['Score'] >= 50 else '#FF6B6B'
        status_label = 'SAIN' if commune_data['Score'] >= 75 else 'A SURVEILLER' if commune_data['Score'] >= 50 else 'FRAGILE'
        
        score_data = [
            [Paragraph(f"<b>SCORE DE SANTE</b>", ParagraphStyle(
                'ScoreLabel',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=10,
                textColor=rl_colors.HexColor('#666666'),
                alignment=TA_CENTER,
                letterSpacing=1
            ))],
            [Paragraph(f"<b>{commune_data['Score']:.1f}/100</b>", ParagraphStyle(
                'ScoreValue',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=48,
                textColor=rl_colors.HexColor(status_color),
                alignment=TA_CENTER
            ))],
            [Paragraph(f"<b>{status_label}</b>", ParagraphStyle(
                'ScoreStatus',
                parent=styles['Normal'],
                fontName='Helvetica-Bold',
                fontSize=10,
                textColor=rl_colors.HexColor('#666666'),
                alignment=TA_CENTER,
                letterSpacing=1
            ))]
        ]
        
        score_table = Table(score_data, colWidths=[12*cm])
        score_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOX', (0, 0), (-1, -1), 2, rl_colors.HexColor(status_color)),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ]))
        
        story.append(score_table)
        story.append(Spacer(1, 2*cm))
        
        date_rapport = datetime.now().strftime("%d/%m/%Y")
        story.append(Paragraph(f"Rapport genere le {date_rapport}", ParagraphStyle(
            'Date',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            textColor=rl_colors.HexColor('#999999'),
            alignment=TA_CENTER
        )))
        
        story.append(PageBreak())
        
        # ========================================
        # PAGE 2 : SYNTHESE EXECUTIVE
        # ========================================
        
        story.append(Paragraph("SYNTHESE EXECUTIVE", style_section))
        story.append(Spacer(1, 0.5*cm))
        
        # Intro
        teb = commune_data['TEB (%)']
        cd = commune_data.get('Années de Désendettement', 0)
        if pd.isna(cd):
            cd = float('inf')
        
        if teb > 20 and cd < 6:
            intro = "Situation financiere saine : la commune dispose d'une epargne robuste et d'une capacite de Désendettement maaitrisee."
        elif teb > 15 and cd < 8:
            intro = "Situation financiere acceptable : les indicateurs sont globalement dans les normes de la strate officielle."
        elif teb < 10 or cd > 12:
            intro = "Situation financiere fragile : attention requise sur l'epargne brute et/ou la capacite de Désendettement."
        else:
            intro = "Situation financiere mitigee : certains indicateurs demandent une surveillance particuliere."
        
        story.append(Paragraph(intro, ParagraphStyle(
            'Intro',
            parent=style_body,
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=rl_colors.HexColor('#1a1a1a'),
            spaceAfter=15
        )))
        
        story.append(Spacer(1, 0.3*cm))
        
        # Insights / Forces
        insights = []
        
        if teb > 15:
            insights.append({
                'icon': '✓',
                'text': f'<b>Epargne robuste :</b> TEB de {teb:.1f}% indique une bonne capacite d\'epargne brute.'
            })
        if cd < 8:
            insights.append({
                'icon': '✓',
                'text': f'<b>Desendettement maitrise :</b> Capacite de {cd:.1f} ans, en dessous du seuil critique.'
            })
        if commune_data.get('FDR Jours Commune', 0) > 240:
            insights.append({
                'icon': '✓',
                'text': f'<b>Tresorerie saine :</b> FDR de {commune_data.get("FDR Jours Commune", 0):.0f} jours assure une liquidite suffisante.'
            })
        
        if not insights:
            insights.append({
                'icon': '!',
                'text': '<b>Points a surveiller :</b> Certains indicateurs necessitent une attention particuliere.'
            })
        
        for insight in insights[:3]:
            insight_data = [[
                Paragraph(f"{insight['icon']}", ParagraphStyle(
                    'Icon',
                    parent=styles['Normal'],
                    fontName='Helvetica-Bold',
                    fontSize=20,
                    alignment=TA_CENTER,
                    textColor=rl_colors.HexColor(status_color)
                )),
                Paragraph(insight['text'], style_body)
            ]]
            
            insight_table = Table(insight_data, colWidths=[1.5*cm, 14*cm])
            insight_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), rl_colors.HexColor('#f5f5f5')),
                ('LEFTPADDING', (0, 0), (-1, -1), 15),
                ('RIGHTPADDING', (0, 0), (-1, -1), 15),
                ('TOPPADDING', (0, 0), (-1, -1), 15),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ]))
            
            story.append(insight_table)
            story.append(Spacer(1, 0.4*cm))
        
        story.append(Spacer(1, 0.5*cm))
        
        # PRIORITES
        story.append(Paragraph("PRIORITES D'ACTION", style_section))
        story.append(Spacer(1, 0.3*cm))
        
        priorites = []
        if teb < 10:
            priorites.append("Renforcer la generation d'epargne brute")
        if cd > 12:
            priorites.append("Piloter activement l'endettement")
        if commune_data.get('FDR Jours Commune', 0) < 60:
            priorites.append("Ameliorer la gestion de la tresorerie")
        if commune_data.get('Annuite / CAF (%)', 0) > 60:
            priorites.append("Reduire le ratio d'annuite par rapport a la CAF")
        
        if not priorites:
            priorites.append("Maintenir la stabilite financiere actuelle")
            priorites.append("Continuer le suivi regulier des indicateurs")
        
        for i, priorite in enumerate(priorites, 1):
            priorite_text = f"<b>{i}.</b> {priorite}"
            story.append(Paragraph(priorite_text, style_body))
            story.append(Spacer(1, 0.3*cm))
        
        story.append(PageBreak())
        
        # ========================================
        # PAGE 3 : INDICATEURS CLES
        # ========================================
        
    
        story.append(Paragraph("INDICATEURS CLES - ANNEE EN COURS", style_section))
        story.append(Spacer(1, 0.5*cm))

    # Récupérer les données de l'année actuelle depuis df_historical_kpi
        if not df_historical_kpi.empty:
            # Prendre la DERNIÈRE année (année en cours)
            data_actuelle = df_historical_kpi.iloc[-1]
            
            kpi_data = [
                ['INDICATEUR', 'COMMUNE', 'STRATE', 'SEUIL BON', 'STATUT'],
                [
                    'TEB (%)',
                    f"{data_actuelle['TEB Commune (%)']:.1f}%" if pd.notna(data_actuelle.get('TEB Commune (%)')) else 'N/A',
                    f"{data_actuelle['TEB Strate (%)']:.1f}%" if pd.notna(data_actuelle.get('TEB Strate (%)')) else 'N/A',
                    '>15%',
                    'BON' if pd.notna(data_actuelle.get('TEB Commune (%)')) and data_actuelle['TEB Commune (%)'] > 15 else 'A SURVEILLER'
                ],
                [
                    'CD (ans)',
                    # ⭐ VALIDATION : Vérifier si négatif ou impossible
                    (f"🔴 {data_actuelle['Années de Désendettement']:.1f}"
                    if pd.notna(data_actuelle.get('Années de Désendettement')) and data_actuelle['Années de Désendettement'] < 0
                    else f"⚠️ N/A (TEB invalide)"
                    if pd.notna(data_actuelle.get('Années de Désendettement')) and data_actuelle['Années de Désendettement'] == 0
                    else f"{data_actuelle['Années de Désendettement']:.1f}"
                    if pd.notna(data_actuelle.get('Années de Désendettement'))
                    else 'N/A'),
                    f"{data_actuelle['CD Strate (années)']:.1f}" if pd.notna(data_actuelle.get('CD Strate (années)')) else 'N/A',
                    '<8',
                    # ⭐ STATUT avec validation
                    ('ELEVE'
                    if pd.notna(data_actuelle.get('Années de Désendettement')) and data_actuelle['Années de Désendettement'] < 0
                    else 'CRITIQUE'
                    if pd.notna(data_actuelle.get('Années de Désendettement')) and data_actuelle['Années de Désendettement'] == 0
                    else 'BON'
                    if pd.notna(data_actuelle.get('Années de Désendettement')) and data_actuelle['Années de Désendettement'] < 8
                    else 'A SURVEILLER')
                ],
                [
                    'Annuite/CAF (%)',
                    (f"🔴 {data_actuelle['Annuité/CAF Commune (%)']:.1f}%" 
                     if pd.notna(data_actuelle.get('Annuité/CAF Commune (%)')) and data_actuelle['Annuité/CAF Commune (%)'] < 0 
                     else f"{data_actuelle['Annuité/CAF Commune (%)']:.1f}%" 
                     if pd.notna(data_actuelle.get('Annuité/CAF Commune (%)')) 
                     else 'N/A'),
                    f"{data_actuelle['Annuité/CAF Strate (%)']:.1f}%" if pd.notna(data_actuelle.get('Annuité/CAF Strate (%)')) else 'N/A',
                    '<50%',
                    # ⭐ STATUT avec validation
                    ('CRITIQUE' 
                     if pd.notna(data_actuelle.get('Annuité/CAF Commune (%)')) and data_actuelle['Annuité/CAF Commune (%)'] < 0
                     else 'BON' 
                     if pd.notna(data_actuelle.get('Annuité/CAF Commune (%)')) and data_actuelle['Annuité/CAF Commune (%)'] < 50 
                     else 'A SURVEILLER')
                ],
                [
                    'FDR (j)',
                    f"{data_actuelle['FDR Jours Commune']:.0f}" if pd.notna(data_actuelle.get('FDR Jours Commune')) else 'N/A',
                    f"{data_actuelle['FDR Jours Moyenne']:.0f}" if pd.notna(data_actuelle.get('FDR Jours Moyenne')) else 'N/A',
                    '>240',
                    'BON' if pd.notna(data_actuelle.get('FDR Jours Commune')) and data_actuelle['FDR Jours Commune'] > 240 else 'A SURVEILLER'
                ],
            ]
        else:
            # Fallback si df_historical_kpi est vide
            st.warning("⚠️ Données historiques insuffisantes pour le tableau KPI")
            kpi_data = [['INDICATEUR', 'COMMUNE', 'STRATE', 'SEUIL BON', 'STATUT']]
               
        kpi_table = Table(kpi_data, colWidths=[3.5*cm, 3.2*cm, 3.2*cm, 3*cm, 3*cm])
        kpi_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#e8e8e8')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, rl_colors.HexColor('#1a1a1a')),
        ]))
        
        story.append(kpi_table)
        story.append(Spacer(1, 1*cm))

        story.append(Paragraph("Tableau Récapitulatif", style_section))
        story.append(Spacer(1, 0.5*cm))

        # ★★★ AJOUTER CES 4 LIGNES ★★★
        tableau_data = create_financial_summary_table_exact(df_historical_kpi)
        if tableau_data:
            # Créer le tableau avec ReportLab
            nb_colonnes = len(tableau_data[0])
            largeur_label = 6*cm
            largeur_annee = (A4[0] - 5*cm - largeur_label) / (nb_colonnes - 1)
            colWidths = [largeur_label] + [largeur_annee] * (nb_colonnes - 1)
                
            table = Table(tableau_data, colWidths=colWidths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#1a1a1a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('TOPPADDING', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#e8e8e8')),
                ('LINEBELOW', (0, 0), (-1, 0), 2, rl_colors.HexColor('#1a1a1a')),
            ]))
                
            story.append(table)
            story.append(Spacer(1, 0.3*cm))
        
        
        
        # ========================================
        # PAGE 4 : PROFIL FINANCIER (RADAR)
        # ========================================
        
        story.append(Paragraph("PROFIL FINANCIER", style_section))
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph(
            "Le radar ci-dessous positionne la commune par rapport a sa strate officielle. "
            "Plus la forme s'etend vers l'exterieur, mieux c'est.",
            style_body
        ))
        story.append(Spacer(1, 0.3*cm))
        
        radar_img = [x[1] for x in temp_images if x[0] == 'radar']
        if radar_img and os.path.exists(radar_img[0]):
            img = Image(radar_img[0], width=15*cm, height=11*cm)
            story.append(img)
        
                # ===== TABLEAU DETAIL NORMALISATION =====
        story.append(Paragraph("Detail de la Normalisation", ParagraphStyle(
            'SubSection',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=rl_colors.HexColor('#1a1a1a'),
            spaceAfter=10
        )))
        
        story.append(Paragraph(
            "Le tableau ci-dessous montre comment chaque indicateur brut est transforme en echelle 0-100 "
            "pour le radar. Cela permet une comparaison homogene sur le meme graphique.",
            ParagraphStyle(
                'Explanation',
                parent=styles['Normal'],
                fontName='Helvetica-Oblique',
                fontSize=9,
                textColor=rl_colors.HexColor('#666666'),
                spaceAfter=10
            )
        ))
        
        story.append(Spacer(1, 0.2*cm))
        
        # Calculer les valeurs normalisees
        def get_value(key):
            try:
                if isinstance(commune_data, pd.Series):
                    return commune_data[key] if key in commune_data.index else None
                else:
                    return commune_data.get(key)
            except:
                return None

        # Calculer les normes
        norms = normaliser_indicateurs_pour_radar(commune_data)

        # ========================================
        # 1. TEB
        # ========================================
        try:
            teb_val = get_value('TEB (%)')
            teb_brute = f"{teb_val:.1f}%" if pd.notna(teb_val) else "N/A"
        except:
            teb_brute = "N/A"

        # ========================================
        # 2. CD - Gestion des négatifs
        # ========================================
        cd_commune = get_value('Années de Désendettement')
        teb_brute_val = get_value('TEB (%)')
        encours_brut = get_value('Encours (K€)')
        caf_brute = get_value('Épargne brute (K€)')

        if pd.notna(teb_brute_val) and teb_brute_val < 0 and pd.notna(encours_brut) and encours_brut > 0 and pd.notna(caf_brute) and caf_brute != 0:
            cd_valeur_brute = f" Supérieur à 15 ans"
        elif pd.notna(cd_commune) and cd_commune < 0:
            cd_valeur_brute = f" Supérieur à 15 ans"
        else:
            cd_valeur_brute = f"{cd_commune:.1f} ans" if pd.notna(cd_commune) else "N/A"

        # ========================================
        # 3. ANNUITÉ/CAF
        # ========================================
        try:
            annuite_val = get_value('Annuité / CAF (%)')
            if pd.notna(annuite_val) and annuite_val < 0:
                annuite_brute = f" Supérieur à 100%"
            else:
                annuite_brute = f"{annuite_val:.1f}%" if pd.notna(annuite_val) else "N/A"
        except:
            annuite_brute = "N/A"

        # ========================================
        # 4. FDR
        # ========================================
        try:
            fdr_val = get_value('FDR Jours Commune')
            fdr_brute = f"{fdr_val:.0f}j" if pd.notna(fdr_val) else "N/A"
        except:
            fdr_brute = "N/A"

        # ========================================
        # 5. RIGIDITÉ
        # ========================================
        try:
            rigidite_val = get_value('Rigidité (%)')
            rigidite_brute = f"{rigidite_val:.1f}%" if pd.notna(rigidite_val) else "N/A"
        except:
            rigidite_brute = "N/A"

        # ========================================
        # TABLEAU norm_data
        # ========================================
        norm_data = [
            ['CRITERE', 'VALEUR BRUTE', 'PLAGE', 'NORMALISE (0-100)', 'INTERPRETATION'],
            [
                'TEB (%)',
                teb_brute,
                '0-30%',
                f"{norms['TEB_norm']:.1f}",
                'Bon' if norms['TEB_norm'] > 50 else 'A surveiller' if norms['TEB_norm'] > 25 else 'Faible'
            ],
            [
                'Années\nDesendettement',
                cd_valeur_brute,
                '0-15 ans (inversee)',
                f"{norms['CD_norm']:.1f}",
                'Bon' if norms['CD_norm'] > 50 else 'A surveiller' if norms['CD_norm'] > 25 else 'Eleve'
            ],
            [
                'Annuité/CAF (%)',
                annuite_brute,
                '0-80% (inversee)',
                f"{norms['Annuité_CAF_norm']:.1f}",
                'Bon' if norms['Annuité_CAF_norm'] > 50 else 'A surveiller' if norms['Annuité_CAF_norm'] > 25 else 'Critique'
            ],
            [
                'FDR (jours)',
                fdr_brute,
                '0-240j',
                f"{norms['FDR_norm']:.1f}",
                'Bon' if norms['FDR_norm'] > 80 else 'Acceptable' if norms['FDR_norm'] > 40 else 'Critique'
            ],
            [
                'Rigidité (%)',
                rigidite_brute,
                '0-200% (inversee)',
                f"{norms['Rigidité_norm']:.1f}",
                'Bon' if norms['Rigidité_norm'] > 50 else 'A surveiller' if norms['Rigidité_norm'] > 25 else 'Eleve'
            ]
        ]
        
        norm_table = Table(norm_data, colWidths=[2.5*cm, 2.8*cm, 2.5*cm, 2.5*cm, 3*cm])
        norm_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#e8e8e8')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, rl_colors.HexColor('#1a1a1a')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#fafafa')])
        ]))
        
        story.append(norm_table)
        
        story.append(Spacer(1, 0.4*cm))
        
        story.append(Paragraph(
            "<b>Note :</b> Les valeurs inversees (CD, Annuite/CAF, Rigidite) signifient que plus la valeur brute est basse, "
            "mieux c'est. La normalisation permet de les ramener a une echelle commune avec les autres indicateurs.",
            ParagraphStyle(
                'Note',
                parent=styles['Normal'],
                fontName='Helvetica-Oblique',
                fontSize=8,
                textColor=rl_colors.HexColor('#999999'),
                spaceAfter=10
            )
        ))
        
        story.append(PageBreak())

        
# ========================================
        # PAGE 5 : EVOLUTION PLURIANNUELLE
        # ========================================
        
        story.append(Paragraph("EVOLUTION PLURIANNUELLE (2019-2024)", style_section))
        story.append(Spacer(1, 0.3*cm))
        
        # Score evolution
        score_img = [x[1] for x in temp_images if x[0] == 'score']
        if score_img and os.path.exists(score_img[0]):
            story.append(Paragraph("1. Evolution du Score Global", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=10
            )))
            img = Image(score_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.5*cm))
        
        # Stacked
        stacked_img = [x[1] for x in temp_images if x[0] == 'stacked']
        if stacked_img and os.path.exists(stacked_img[0]):
            story.append(Paragraph("2. Contribution des Composantes", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=10
            )))
            img = Image(stacked_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.5*cm))
            story.append(PageBreak())
        
        # Lines
        lines_img = [x[1] for x in temp_images if x[0] == 'lines']
        if lines_img and os.path.exists(lines_img[0]):
            story.append(Paragraph("3. Evolution Detaillee par Composante", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=10
            )))
            img = Image(lines_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.5*cm))
        
        story.append(PageBreak())
        
        # ========================================
        # PAGE 6 : INDICATEURS INDIVIDUELS
        # ========================================
        
        story.append(Paragraph("INDICATEURS INDIVIDUELS - EVOLUTION DETAILLEE", style_section))
        story.append(Spacer(1, 0.3*cm))
        
        # TEB
        teb_img = [x[1] for x in temp_images if x[0] == 'teb_ind']
        if teb_img and os.path.exists(teb_img[0]):
            story.append(Paragraph("TEB - Taux d'Epargne Brute", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=8
            )))
            story.append(Paragraph(
                "Mesure la capacite de la commune a degager de l'epargne. "
                "Seuil vert : >15% | Seuil critique : <10%",
                ParagraphStyle(
                    'Explanation',
                    parent=styles['Normal'],
                    fontName='Helvetica-Oblique',
                    fontSize=9,
                    textColor=rl_colors.HexColor('#666666'),
                    spaceAfter=10
                )
            ))
            img = Image(teb_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.4*cm))
        
        # CD
        cd_img = [x[1] for x in temp_images if x[0] == 'cd_ind']
        if cd_img and os.path.exists(cd_img[0]):
            story.append(Paragraph("CD - Capacite de Désendettement", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=8
            )))
            story.append(Paragraph(
                "Nombre d'annees necessaires pour rembourser la dette avec l'epargne. "
                "Seuil vert : <8 ans | Seuil critique : >12 ans",
                ParagraphStyle(
                    'Explanation',
                    parent=styles['Normal'],
                    fontName='Helvetica-Oblique',
                    fontSize=9,
                    textColor=rl_colors.HexColor('#666666'),
                    spaceAfter=10
                )
            ))
            img = Image(cd_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.4*cm))
        
        story.append(PageBreak())
        
        # Annuité/CAF
        annuite_img = [x[1] for x in temp_images if x[0] == 'annuite_ind']
        if annuite_img and os.path.exists(annuite_img[0]):
            story.append(Paragraph("Ratio Annuite / CAF Brute", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=8
            )))
            story.append(Paragraph(
                "Part des annuites (dettes) dans les recettes reelles de fonctionnement. "
                "Seuil vert : <50% | Seuil critique : >60%",
                ParagraphStyle(
                    'Explanation',
                    parent=styles['Normal'],
                    fontName='Helvetica-Oblique',
                    fontSize=9,
                    textColor=rl_colors.HexColor('#666666'),
                    spaceAfter=10
                )
            ))
            img = Image(annuite_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.4*cm))
        
        # FDR
        fdr_img = [x[1] for x in temp_images if x[0] == 'fdr_ind']
        if fdr_img and os.path.exists(fdr_img[0]):
            story.append(Paragraph("FDR - Fonds de Roulement", ParagraphStyle(
                'SubSection',
                parent=styles['Heading3'],
                fontName='Helvetica-Bold',
                fontSize=11,
                textColor=rl_colors.HexColor('#1a1a1a'),
                spaceAfter=8
            )))
            story.append(Paragraph(
                "Nombre de jours de fonctionnement garantis par la tresorerie. "
                "Seuil vert : >240 j | Seuil critique : <60 j",
                ParagraphStyle(
                    'Explanation',
                    parent=styles['Normal'],
                    fontName='Helvetica-Oblique',
                    fontSize=9,
                    textColor=rl_colors.HexColor('#666666'),
                    spaceAfter=10
                )
            ))
            img = Image(fdr_img[0], width=16*cm, height=8*cm)
            story.append(img)
            story.append(Spacer(1, 0.4*cm))
        
        story.append(PageBreak())
        # ========================================
        # PAGE 6 : TABLEAU EVOLUTION
        # ========================================
        
        story.append(Paragraph("TABLEAU RECAPITULATIF", style_section))
        story.append(Spacer(1, 0.5*cm))
        
        tableau_data = [['ANNEE', 'SCORE', 'TEB (%)', 'CD (ans)', 'ANNUITE/CAF (%)', 'FDR (j)']]
        
        for _, row in df_historical_kpi.iterrows():
            tableau_data.append([
                str(int(row['Année'])),
                f"{row['Score Commune']:.1f}",
                f"{row['TEB Commune (%)']:.1f}",
                f"{row['Années de Désendettement']:.1f}",
                f"{row['Annuité/CAF Commune (%)']:.1f}",
                f"{row['FDR Jours Commune']:.0f}" if pd.notna(row.get('FDR Jours Commune')) else 'N/A'
            ])
        
        evolution_table = Table(tableau_data, colWidths=[1.5*cm, 2*cm, 2*cm, 2*cm, 2.5*cm, 2*cm])
        evolution_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#1a1a1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.HexColor('#e8e8e8')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, rl_colors.HexColor('#1a1a1a')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor('#fafafa')])
        ]))
        
        story.append(evolution_table)
        
        # === BUILD PDF ===
        doc.build(story)
        
        # Nettoyage
        for _, img_path in temp_images:
            try:
                os.unlink(img_path)
            except:
                pass
        
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    except Exception as e:
        st.error(f"Erreur lors de la generation du PDF : {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

def create_radar_plot_matplotlib(commune_data, df_filtered=None):
    """
    Crée un radar plot avec Matplotlib (pas de dépendance Chrome/Kaleido)
    """
    norms = normaliser_indicateurs_pour_radar(commune_data)
    
    categories = [
        'TEB (%) 0-30%',
        'Années Désendettement\n0-15 ans',
        'Annuité/CAF (%)\n0-80%',
        'FDR (jours)\n0-240j',
        'Rigidité (%)\ninversion 0-200%'
    ]
    
    values_commune = [
        norms['TEB_norm'],
        norms['CD_norm'],
        norms['Annuité_CAF_norm'],
        norms['FDR_norm'],
        norms['Rigidité_norm']
    ]
    
    # Seuils vert normalisés
    seuils_vert = [
        (15 / 30) * 100,              # TEB : 50
        ((15 - 8) / 15) * 100,        # CD : 46.67
        ((80 - 50) / 80) * 100,       # Annuité : 37.5
        (240 / 300) * 100,            # FDR : 80
        ((200 - 100) / 200) * 100     # Rigidité : 50
    ]
    
    # Nombre de variables
    num_vars = len(categories)
    
    # Angles pour chaque axe
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    values_commune += values_commune[:1]  # Fermer le polygone
    seuils_vert += seuils_vert[:1]
    angles += angles[:1]
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Tracer la commune
    ax.plot(angles, values_commune, 'o-', linewidth=3, label=commune_data['Commune'], color='#3b82f6')
    ax.fill(angles, values_commune, alpha=0.25, color='#3b82f6')
    
    # Tracer les seuils verts
    ax.plot(angles, seuils_vert, 'o--', linewidth=2, label='Seuil Vert', color='#10b981')
    
    # Moyenne strate si disponible
    if df_filtered is not None and not df_filtered.empty:
        moyennes_strate = df_filtered.apply(normaliser_indicateurs_pour_radar, axis=1).apply(pd.Series).mean()
        
        values_strate = [
            moyennes_strate['TEB_norm'],
            moyennes_strate['CD_norm'],
            moyennes_strate['Annuité_CAF_norm'],
            moyennes_strate['FDR_norm'],
            moyennes_strate['Rigidité_norm']
        ]
        values_strate += values_strate[:1]
        
        ax.plot(angles, values_strate, 's:', linewidth=2, label='Moyenne Strate', color='#f59e0b')
        ax.fill(angles, values_strate, alpha=0.15, color='#f59e0b')
    
    # Configurer les axes
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(['25', '50', '75', '100'], size=10)
    ax.grid(True, alpha=0.3)
    
    # Titre et légende
    plt.title(f'🎯 Profil Financier Cohérent\n{commune_data["Commune"]} | Score: {commune_data["Score"]:.0f}/100',
              size=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
    
    plt.tight_layout()
    return fig



def create_radar_plot_for_pdf(commune_data, df_filtered=None):
    """
    Crée un radar plot pour le PDF - Utilise la MÊME logique que create_radar_coherent()
    
    LOGIQUE UNIFORME : 
    - Plus on s'éloigne du CENTRE (0) vers l'EXTÉRIEUR (100) = MIEUX C'EST
    """
    
    norms = normaliser_indicateurs_pour_radar(commune_data)
    
    categories = [
        'TEB (%) 0-30%',
        'Années Désendettement 0-15 ans',
        'Annuité/CAF (%) 0-80%',
        'FDR (jours) 0-240j',
        'Rigidité (%) inversion 0-200%'
    ]
    
    values_commune = [
        norms['TEB_norm'],
        norms['CD_norm'],
        norms['Annuité_CAF_norm'],
        norms['FDR_norm'],
        norms['Rigidité_norm']
    ]
    
    # Seuils vert normalisés
    seuils_vert = [
        (15 / 30) * 100,              # TEB : 50
        ((15 - 8) / 15) * 100,        # CD : 46.67
        ((80 - 50) / 80) * 100,       # Annuité : 37.5
        (240 / 300) * 100,            # FDR : 80
        ((200 - 100) / 200) * 100     # Rigidité : 50
    ]
    
    fig = go.Figure()
    
    # Trace commune
    fig.add_trace(go.Scatterpolar(
        r=values_commune,
        theta=categories,
        fill='toself',
        name=commune_data['Commune'],
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8),
        fillcolor='rgba(59, 130, 246, 0.25)'
    ))
    
    # Trace seuils vert
    fig.add_trace(go.Scatterpolar(
        r=seuils_vert,
        theta=categories,
        fill=None,
        name='Seuil Vert',
        line=dict(color='#10b981', width=2, dash='dash'),
        marker=dict(size=6),
    ))
    
    # Trace moyenne strate
    if df_filtered is not None and not df_filtered.empty:
        moyennes_strate = df_filtered.apply(normaliser_indicateurs_pour_radar, axis=1).apply(pd.Series).mean()
        
        values_strate = [
            moyennes_strate['TEB_norm'],
            moyennes_strate['CD_norm'],
            moyennes_strate['Annuité_CAF_norm'],
            moyennes_strate['FDR_norm'],
            moyennes_strate['Rigidité_norm']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values_strate,
            theta=categories,
            fill='toself',
            name='Moyenne Strate',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            marker=dict(size=6),
            fillcolor='rgba(245, 158, 11, 0.15)'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='outside',
                tickfont=dict(size=10),
                gridcolor='rgba(243, 244, 246, 0.5)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        title=dict(
            text=f"<b>Profil Financier Coherent</b><br><sub>{commune_data['Commune']} | Score: {commune_data['Score']:.0f}/100</sub>",
            font=dict(size=14)
        ),
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        font=dict(size=12),
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    fig.add_annotation(
        text="<b>Logique uniforme :</b> Plus vers l'exterieur = Mieux<br>Plus vers le centre = Pire",
        xref="paper", yref="paper",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=11, color="#666"),
        align="center"
    )
    
    return fig





# --- Fonction d'export Excel ---
def create_excel_export(df_kpi):
    """Crée un fichier Excel à télécharger - Solution robuste Windows"""
    try:
        import time
        import uuid
        
        unique_name = f"analyse_{uuid.uuid4().hex[:8]}.xlsx"
        temp_path = os.path.join(tempfile.gettempdir(), unique_name)
        
        with pd.ExcelWriter(temp_path, engine='xlsxwriter') as writer:
            df_kpi.to_excel(writer, sheet_name='Analyse_KPI', index=False)
            
            synthese = df_kpi.groupby('Niveau d\'alerte').agg({
                'Commune': 'count',
                'Population': 'sum',
                'Score': 'mean'
            }).round(2)
            synthese.to_excel(writer, sheet_name='Synthese')
            
            try:
                workbook = writer.book
                
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                green_format = workbook.add_format({'bg_color': '#C6EFCE'})
                orange_format = workbook.add_format({'bg_color': '#FFEB9C'})
                red_format = workbook.add_format({'bg_color': '#FFC7CE'})
                
                worksheet = writer.sheets['Analyse_KPI']
                
                for col_num, value in enumerate(df_kpi.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
                    if 'Commune' in str(value):
                        worksheet.set_column(col_num, col_num, 20)
                    elif 'Population' in str(value):
                        worksheet.set_column(col_num, col_num, 12)
                    elif 'Score' in str(value):
                        worksheet.set_column(col_num, col_num, 10)
                    else:
                        worksheet.set_column(col_num, col_num, 15)
                
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
                pass
        
        time.sleep(0.2)
        
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                with open(temp_path, 'rb') as file:
                    excel_data = file.read()
                break
            except (PermissionError, FileNotFoundError) as e:
                if attempt == max_attempts - 1:
                    raise e
                time.sleep(0.2)
        
        try:
            if os.path.exists(temp_path):
                time.sleep(0.1)
                os.remove(temp_path)
        except (PermissionError, FileNotFoundError, OSError):
            pass
        
        return excel_data
    
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier Excel : {e}")
        try:
            csv_data = df_kpi.to_csv(
                index=False, 
                sep=';',
                encoding='utf-8-sig',
                decimal=',',
                float_format='%.2f'
            )
            st.warning("⚠️ Export Excel échoué, fichier CSV généré à la place")
            return csv_data.encode('utf-8-sig')
        except Exception as csv_error:
            st.error(f"Erreur également sur l'export CSV : {csv_error}")
            return None

# === RÉCUPÉRATION ET TRAITEMENT DES DONNÉES ===
df_dept = fetch_communes(dept_selection, annee_selection)

if df_dept.empty:
    st.warning(f"❌ Aucune donnée disponible pour le département {dept_selection} en {annee_selection}.")
else:
    if taille_min > 0:
        df_dept = df_dept[df_dept['Population'] >= taille_min]
    
    if df_dept.empty:
        st.warning("❌ Aucune commune ne correspond aux critères de filtrage.")
    else:
        # === CALCULS KPI V2 (VERSION CORRIGÉE) ===
        df_kpi = df_dept.copy()
        
        # --- KPI de base ---
        df_kpi["TEB (%)"] = df_kpi["Épargne brute (K€)"] / df_kpi["RRF (K€)"].replace(0, pd.NA) * 100
        # 🔧 Années de Désendettement avec gestion TEB négatif
        def calc_annees_desendettement(encours, epargne_brute):
            """Calcule les années de désendettement"""
            if pd.isna(encours) or encours <= 0:
                return 0  # Pas de dette
            if pd.isna(epargne_brute) or epargne_brute <= 0:
                return 0  # Impossible si épargne <= 0 (inclut TEB négatif)
            return encours / epargne_brute

        df_kpi["Années de Désendettement"] = df_kpi.apply(
            lambda row: calc_annees_desendettement(
                row["Encours (K€)"], 
                row["Épargne brute (K€)"]
            ),
            axis=1
        )
        df_kpi["Rigidité (%)"] = (df_kpi["DRF (K€)"] / df_kpi["RRF (K€)"].replace(0, pd.NA) * 100)
        
        # Encours / hab : utiliser directement la colonne si disponible
        if 'Encours / hab Commune' in df_kpi.columns:
            df_kpi["Encours / hab (€/hab)"] = df_kpi['Encours / hab Commune']
        else:
            df_kpi["Encours / hab (€/hab)"] = df_kpi["Encours (K€)"] * 1000 / df_kpi["Population"].replace(0, pd.NA)
        
        # --- NOUVEAUX KPI V2 ---
        
        # 1. Ratio Annuité / CAF Brute
        df_kpi["Annuité / CAF (%)"] = df_kpi["Annuité (K€)"] / df_kpi["Épargne brute (K€)"].replace(0, pd.NA) * 100
        
        # 2. ✅ FDR en jours - VERSION CORRIGÉE (utilisation directe des champs €/hab)
        if 'FDR / hab Commune' in df_kpi.columns and 'DRF / hab Commune' in df_kpi.columns:
            df_kpi['FDR Jours Commune'] = (
                df_kpi['FDR / hab Commune'] / df_kpi['DRF / hab Commune'].replace(0, pd.NA) * 365
            ).round(2)
            
            # Sécurité : plafonner à 1000 jours et identifier les anomalies
            nb_aberrants = (df_kpi['FDR Jours Commune'] > 1000).sum()
            if nb_aberrants > 0:
                st.info(f"ℹ️ {nb_aberrants} communes ont un FDR > 1000 jours (valeurs plafonnées)")
            df_kpi.loc[df_kpi['FDR Jours Commune'] > 1000, 'FDR Jours Commune'] = pd.NA
            
            # Statistiques
            fdr_valides = df_kpi['FDR Jours Commune'].notna().sum()
            if fdr_valides > 0:
                fdr_median = df_kpi['FDR Jours Commune'].median()
                fdr_min = df_kpi['FDR Jours Commune'].min()
                fdr_max = df_kpi['FDR Jours Commune'].max()
        else:
            df_kpi['FDR Jours Commune'] = pd.NA
            st.warning("⚠️ Données FDR non disponibles dans l'API pour cette année")
        
        # 3. FDR en jours - MOYENNE STRATE
        if 'FDR / hab Moyenne' in df_kpi.columns and 'DRF / hab Moyenne' in df_kpi.columns:
            df_kpi['FDR Jours Moyenne'] = (
                df_kpi['FDR / hab Moyenne'] / df_kpi['DRF / hab Moyenne'].replace(0, pd.NA) * 365
            ).round(2)
            df_kpi.loc[df_kpi['FDR Jours Moyenne'] > 1000, 'FDR Jours Moyenne'] = pd.NA
        else:
            df_kpi['FDR Jours Moyenne'] = pd.NA
        
        # --- Calcul des scores V2 ---
        df_kpi['Score'] = df_kpi.apply(score_sante_financiere_v3, axis=1, df_ref=df_kpi)
        df_kpi['Niveau d\'alerte'] = df_kpi['Score'].apply(niveau_alerte_v3)
        # Création des tranches de population
        df_kpi = create_population_brackets(df_kpi)
        
        # Filtre par niveau d'alerte
        niveaux_dispo = df_kpi['Niveau d\'alerte'].unique()
        niveau_filtre = st.sidebar.multiselect("Niveau d'alerte", niveaux_dispo, default=niveaux_dispo)
        df_filtered = df_kpi[df_kpi['Niveau d\'alerte'].isin(niveau_filtre)]
        
        # === MODE DEBUG FDR (optionnel) ===
        if st.sidebar.checkbox("🔬 Mode Debug FDR"):
            st.subheader("🔬 Diagnostic des données FDR")
            
            # Échantillon
            cols_debug = ['Commune', 'Population', 'FDR / hab Commune', 'DRF / hab Commune', 'FDR Jours Commune']
            cols_disponibles = [c for c in cols_debug if c in df_kpi.columns]
            
            if cols_disponibles:
                echantillon = df_kpi[cols_disponibles].head(10)
                st.dataframe(echantillon, use_container_width=True)
                
                # Statistiques globales
                st.write("### 📊 Statistiques FDR départemental")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min", f"{df_kpi['FDR Jours Commune'].min():.0f}j")
                with col2:
                    st.metric("Médiane", f"{df_kpi['FDR Jours Commune'].median():.0f}j")
                with col3:
                    st.metric("Max", f"{df_kpi['FDR Jours Commune'].max():.0f}j")
                with col4:
                    st.metric("Valides", f"{df_kpi['FDR Jours Commune'].notna().sum()}")
        
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
            alert_counts = df_filtered['Niveau d\'alerte'].value_counts()
            colors = [get_color_alerte(niveau) for niveau in alert_counts.index]
            
            fig_pie = px.pie(values=alert_counts.values, names=alert_counts.index,
                            title="🎯 Répartition des niveaux d'alerte",
                            color_discrete_sequence=colors)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_hist = px.histogram(df_filtered, x='Score', nbins=15,
                                   title="📈 Distribution des scores de santé financière",
                                   labels={'Score': 'Score de santé', 'count': 'Nombre de communes'})
            fig_hist.add_vline(x=50, line_dash="dash", line_color="orange", annotation_text="Seuil Orange")
            fig_hist.add_vline(x=75, line_dash="dash", line_color="green", annotation_text="Seuil Vert")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Ligne 2 : Analyse comparative avec NOUVEAUX SEUILS
        col1, col2 = st.columns(2)
        
        with col1:
            fig_scatter = px.scatter(df_filtered, x='TEB (%)', y='Années de Désendettement',
                            color='Niveau d\'alerte', size='Population',
                            hover_data=['Commune', 'Score'],
                            title="💰 Taux d'épargne vs Années de désendettement",
                            color_discrete_map={
                                "🟢 Vert": "#00C851",
                                "🟠 Orange": "#FF8C00", 
                                "🔴 Rouge": "#FF4B4B"
                            },
                            size_max=50,  # Limiter la taille max des bulles
                            opacity=0.7)  # Légère transparence pour voir les superpositions
        
            # Seuils horizontaux (Années de Désendettement)
            fig_scatter.add_hline(y=12, line_dash="dash", line_color="red", 
                                annotation_text="Seuil critique (12 ans)", 
                                annotation_position="right")
            fig_scatter.add_hline(y=8, line_dash="dash", line_color="orange", 
                                annotation_text="Seuil (8 ans)", 
                                annotation_position="right")
            
            # Seuils verticaux (TEB)
            fig_scatter.add_vline(x=10, line_dash="dash", line_color="orange", 
                                annotation_text="Seuil (10%)", 
                                annotation_position="top")
            fig_scatter.add_vline(x=15, line_dash="dash", line_color="green", 
                                annotation_text="Seuil (15%)", 
                                annotation_position="top")
            
            # Zone de sécurité (optionnel mais visuelle)
            fig_scatter.add_vrect(x0=15, x1=100, fillcolor="green", opacity=0.05, 
                                line_width=0, layer="below")
            fig_scatter.add_hrect(y0=0, y1=8, fillcolor="green", opacity=0.05, 
                                line_width=0, layer="below")
            
            # Améliorations des axes
            fig_scatter.update_yaxes(
                range=[0, 20], 
                dtick=2,
                title_font=dict(size=12, color="black"),
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray"
            )
            
            fig_scatter.update_xaxes(
                title_font=dict(size=12, color="black"),
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray"
            )
            
            # Améliorations globales
            fig_scatter.update_layout(
                hovermode='closest',  # Meilleur hover
                height=600,
                font=dict(size=11),
                plot_bgcolor="white",  # Fond subtil
                legend=dict(
                    x=0.02, y=0.98,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            fig_box = px.box(df_filtered, x='Niveau d\'alerte', y='TEB (%)',
                           title="📊 Distribution du TEB par niveau d'alerte",
                           color='Niveau d\'alerte',
                           color_discrete_map={
                               "🟢 Vert": "#00C851",
                               "🟠 Orange": "#FF8C00", 
                               "🔴 Rouge": "#FF4B4B"
                           })
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Ligne 3 : NOUVEAUX GRAPHIQUES (Annuité/Caf vs Années de désendettement )
        col1, col2 = st.columns(2)
        SEUIL_DESENDETTEMENT_BON = 8  # Vert
        SEUIL_DESENDETTEMENT_CRIT = 12  # Rouge
        with col1:
            fig_annuite_fdr = px.scatter(df_filtered, x='Annuité / CAF (%)', y='Années de Désendettement',
                                        color='Niveau d\'alerte', size='Population',
                                        hover_data=['Commune', 'Score'],
                                        title="💳 Ratio Annuité/CAF vs Années de Désendettement",
                                        color_discrete_map={
                                            "🟢 Vert": "#00C851",
                                            "🟠 Orange": "#FF8C00",
                                            "🔴 Rouge": "#FF4B4B"
                                        },
                                        size_max=50,
                                        opacity=0.7)
                
                # Seuils horizontaux avec les bonnes valeurs
            fig_annuite_fdr.add_hline(y=SEUIL_DESENDETTEMENT_BON, line_dash="dash", line_color="green", 
                                        annotation_text="Seuil bon (8 ans)", 
                                        annotation_position="right")
            fig_annuite_fdr.add_hline(y=SEUIL_DESENDETTEMENT_CRIT, line_dash="dash", line_color="red", 
                                        annotation_text="Seuil critique (12 ans)", 
                                        annotation_position="right")
                
                # Seuils verticaux (Annuité/CAF)
            fig_annuite_fdr.add_vline(x=50, line_dash="dash", line_color="orange", 
                                        annotation_text="Seuil (50%)", 
                                        annotation_position="top")
            fig_annuite_fdr.add_vline(x=60, line_dash="dash", line_color="red", 
                                        annotation_text="Seuil critique (60%)", 
                                        annotation_position="top")
                
                # Zones de sécurité
            fig_annuite_fdr.add_vrect(x0=0, x1=50, fillcolor="green", opacity=0.05, 
                                        line_width=0, layer="below")
            fig_annuite_fdr.add_hrect(y0=0, y1=SEUIL_DESENDETTEMENT_BON, fillcolor="green", opacity=0.05, 
                                        line_width=0, layer="below")
                
                # Améliorations des axes
            fig_annuite_fdr.update_yaxes(
                    range=[0, 20], 
                    dtick=2,
                    title_font=dict(size=12, color="black"),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                )
                
            fig_annuite_fdr.update_xaxes(
                    range=[0, 120],
                    dtick=10,
                    title_font=dict(size=12, color="black"),
                    showgrid=True,
                    gridwidth=1,
                    gridcolor="lightgray"
                )
                
                # Améliorations globales
            fig_annuite_fdr.update_layout(
                    hovermode='closest',
                    height=600,
                    font=dict(size=11),
                    plot_bgcolor="white",
                    legend=dict(
                        x=0.02, y=0.98,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="gray",
                        borderwidth=1
                    )
                )
                
            st.plotly_chart(fig_annuite_fdr, use_container_width=True)
        
        with col2:
            # Box plot FDR par niveau
            fig_fdr_box = px.box(df_filtered, x='Niveau d\'alerte', y='Années de Désendettement',
                     title="💰 Distribution des Années de Désendettement par niveau d'alerte",
                     color='Niveau d\'alerte',
                     color_discrete_map={
                         "🟢 Vert": "#00C851",
                         "🟠 Orange": "#FF8C00", 
                         "🔴 Rouge": "#FF4B4B"
                     },
                     points="outliers")  # Afficher les outliers

    # Seuils avec annotations
            fig_fdr_box.add_hline(y=8, line_dash="dash", line_color="green", 
                                annotation_text="Seuil bon (8 ans)",
                                annotation_position="right")
            fig_fdr_box.add_hline(y=12, line_dash="dash", line_color="red", 
                                annotation_text="Seuil critique (12 ans)",
                                annotation_position="right")

            # Zone de sécurité
            fig_fdr_box.add_hrect(y0=0, y1=8, fillcolor="green", opacity=0.05, 
                                line_width=0, layer="below")

            # Améliorer les axes
            fig_fdr_box.update_yaxes(
                range=[0, 15],  # Échelle fixée de 0 à 15 ans
                dtick=3,
                title_font=dict(size=12, color="black"),
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray"
            )

            fig_fdr_box.update_xaxes(
                title_font=dict(size=12, color="black")
            )

            # Afficher la moyenne et l'écart-type
            

            # Améliorations globales
            fig_fdr_box.update_layout(
                hovermode='closest',
                height=600,
                font=dict(size=11),
                plot_bgcolor="white",
                showlegend=False  # Pas besoin de légende, les couleurs parlent d'elles-mêmes
            )

            st.plotly_chart(fig_fdr_box, use_container_width=True)
        
        # Ligne 4 : Analyse par taille
        col1, col2 = st.columns(2)
        
        with col1:
            score_by_size = df_filtered.groupby('Tranche pop')['Score'].mean().reset_index()
            fig_bar = px.bar(score_by_size, x='Tranche pop', y='Score',
                           title="📏 Score moyen par taille de commune",
                           labels={'Score': 'Score moyen', 'Tranche pop': 'Taille de commune'})
            fig_bar.add_hline(y=50, line_dash="dash", line_color="orange")
            fig_bar.add_hline(y=75, line_dash="dash", line_color="green")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_debt = px.scatter(df_filtered, x='Population', y='Encours / hab (€/hab)',
                      color='Niveau d\'alerte',
                      title="💳 Endettement par habitant vs Population",
                      color_discrete_map={
                          "🟢 Vert": "#00C851",
                          "🟠 Orange": "#FF8C00",
                          "🔴 Rouge": "#FF4B4B"
                      },
                      hover_data=['Commune'],
                      size='Population',
                      size_max=50,
                      opacity=0.7)

            # Améliorer les axes
            fig_debt.update_xaxes(
                range=[0, 25000],
                dtick=5000,  # Graduations tous les 5000 habitants
                title_font=dict(size=12, color="black"),
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray"
            )

            fig_debt.update_yaxes(
                title_font=dict(size=12, color="black"),
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray"
            )

            # Améliorations globales
            fig_debt.update_layout(
                hovermode='closest',
                height=600,
                font=dict(size=11),
                plot_bgcolor="white",
                legend=dict(
                    x=0.02, y=0.98,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )

            st.plotly_chart(fig_debt, use_container_width=True)
                    
        # === TABLEAUX TOP/FLOP ===
        st.markdown("---")
        st.subheader("🏆 Classements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Top 25 - Communes les plus fragiles")
            colonnes_top = ['Commune', 'Population', 'Score', 'TEB (%)', 'Années de Désendettement', 'Annuité / CAF (%)']
            if 'FDR Jours Commune' in df_filtered.columns:
                colonnes_top.append('FDR Jours Commune')
            top_risk = df_filtered.nsmallest(25, 'Score')[colonnes_top]
            st.dataframe(top_risk, use_container_width=True)
        
        with col2:
            st.markdown("#### 🟢 Top 25 - Communes les plus solides")
            top_solid = df_filtered.nlargest(25, 'Score')[colonnes_top]
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
                
                st.markdown("---")
                st.markdown("**📊 Indicateurs clés :**")
                st.markdown(f"- TEB : {commune_data['TEB (%)']:.1f}%")
                cd = commune_data['Années de Désendettement']
                st.markdown(f"- **Années Désendettement :** {'Impossible (TEB négatif)' if pd.isna(cd) else f'{cd:.1f} ans'}")
                if pd.notna(commune_data['Annuité / CAF (%)']):
                    st.markdown(f"- Annuité/CAF : {commune_data['Annuité / CAF (%)']:.1f}%")
                else:
                    st.markdown(f"- Annuité/CAF : N/A")
                if pd.notna(commune_data.get('FDR Jours Commune')):
                    st.markdown(f"- FDR : {commune_data['FDR Jours Commune']:.0f} jours")
                else:
                    st.markdown(f"- FDR : Donnée non disponible")
            
            with col2:
                # Radar chart avec NOUVEAUX KPI
                categories = ['TEB', 'CD inversée', 'Annuité/CAF inv.', 'FDR Jours', 'Rigidité inv.']
                
                # Normalisation des valeurs COMMUNE (0-100)
                teb_norm = min(100, (commune_data['TEB (%)'] / 15) * 100)
                if pd.notna(commune_data['Années de Désendettement']) and commune_data['Années de Désendettement'] >= 0:
                    cd_norm = max(0, min(100, (12 - commune_data['Années de Désendettement']) / 12 * 100))
                else:
                    cd_norm = 0
                if pd.notna(commune_data.get('Annuité / CAF (%)')):
                    annuite_caf_norm = max(0, min(100, (60 - commune_data['Annuité / CAF (%)']) / 60 * 100))
                else:
                    annuite_caf_norm = 100
                
                if pd.notna(commune_data.get('FDR Jours Commune')):
                    fdr_norm = min(100, (commune_data['FDR Jours Commune'] / 240) * 100)
                else:
                    fdr_norm = 50
                
                rigidite_norm = max(0, min(100, 200 - commune_data['Rigidité (%)']))
                
                # Calcul des KPI de la STRATE OFFICIELLE
                epargne_strate = commune_data.get('Épargne brute - Moy. strate (K€)')
                rrf_strate = commune_data.get('RRF - Moy. strate (K€)')
                drf_strate = commune_data.get('DRF - Moy. strate (K€)')
                encours_strate = commune_data.get('Encours - Moy. strate (K€)')
                annuite_strate_val = commune_data.get('Annuité - Moy. strate (K€)')
                
                # Ratios STRATE
                teb_strate = (epargne_strate / rrf_strate * 100) if pd.notna(rrf_strate) and rrf_strate != 0 else 0
                cd_strate = (encours_strate / epargne_strate) if pd.notna(epargne_strate) and epargne_strate != 0 else 0
                rigidite_strate = (drf_strate / rrf_strate * 100) if pd.notna(rrf_strate) and rrf_strate != 0 else 0
                annuite_caf_strate = (annuite_strate_val / epargne_strate * 100) if pd.notna(epargne_strate) and epargne_strate != 0 else 0
                fdr_jours_strate = commune_data.get('FDR Jours Moyenne') if pd.notna(commune_data.get('FDR Jours Moyenne')) else 0
                
                # Normalisation STRATE
                teb_strate_norm = min(100, (teb_strate / 15) * 100)
                cd_strate_norm = max(0, min(100, (12 - cd_strate) / 12 * 100))
                rigidite_strate_norm = max(0, min(100, 200 - rigidite_strate))
                annuite_caf_strate_norm = max(0, min(100, (60 - annuite_caf_strate) / 60 * 100))
                fdr_strate_norm = min(100, (fdr_jours_strate / 240) * 100) if fdr_jours_strate > 0 else 50
                    
                                # Radar cohérent avec VRAIES PLAGES
                fig_radar_coherent = create_radar_coherent(commune_data, df_filtered)
                st.plotly_chart(fig_radar_coherent, use_container_width=True)


                # Tableau de normalisation (pour expliquer la transformation)
                st.subheader("📊 Détail de la normalisation")
                tableau_norm = create_tableau_normalisation(commune_data)
                st.dataframe(tableau_norm, use_container_width=True, hide_index=True)
                
                # Analyse comparative textuelle
                st.markdown("**🎯 Analyse comparative vs strate officielle :**")
                
                comparaisons = []
                if teb_norm > teb_strate_norm + 10:
                    comparaisons.append(f"✅ TEB supérieur à la strate ({commune_data['TEB (%)']:.1f}% vs {teb_strate:.1f}%)")
                elif teb_norm < teb_strate_norm - 10:
                    comparaisons.append(f"⚠️ TEB inférieur à la strate ({commune_data['TEB (%)']:.1f}% vs {teb_strate:.1f}%)")
                
                if cd_norm > cd_strate_norm + 10:
                    comparaisons.append(f"✅ Endettement mieux maîtrisé que la strate ({commune_data['Années de Désendettement']:.1f} ans vs {cd_strate:.1f} ans)")
                elif cd_norm < cd_strate_norm - 10:
                    comparaisons.append(f"⚠️ Endettement plus élevé que la strate ({commune_data['Années de Désendettement']:.1f} ans vs {cd_strate:.1f} ans)")
                
                if pd.notna(commune_data.get('Annuité / CAF (%)')):
                    if annuite_caf_norm > annuite_caf_strate_norm + 10:
                        comparaisons.append(f"✅ Ratio Annuité/CAF plus favorable que la strate ({commune_data['Annuité / CAF (%)']:.1f}% vs {annuite_caf_strate:.1f}%)")
                    elif annuite_caf_norm < annuite_caf_strate_norm - 10:
                        comparaisons.append(f"⚠️ Ratio Annuité/CAF moins favorable que la strate ({commune_data['Annuité / CAF (%)']:.1f}% vs {annuite_caf_strate:.1f}%)")
                
                if pd.notna(commune_data.get('FDR Jours Commune')) and fdr_jours_strate > 0:
                    if fdr_norm > fdr_strate_norm + 10:
                        comparaisons.append(f"✅ FDR supérieur à la strate ({commune_data['FDR Jours Commune']:.0f}j vs {fdr_jours_strate:.0f}j)")
                    elif fdr_norm < fdr_strate_norm - 10:
                        comparaisons.append(f"⚠️ FDR inférieur à la strate ({commune_data['FDR Jours Commune']:.0f}j vs {fdr_jours_strate:.0f}j)")
                
                if comparaisons:
                    for comp in comparaisons:
                        st.markdown(f"- {comp}")
                else:
                    st.markdown("- 📊 Performance globalement dans la moyenne de la strate officielle")
            
# === ANALYSE PLURIANNUELLE ===
        st.markdown("---")
        st.subheader(f"📊 Évolution pluriannuelle : {commune_selectionnee}")
        st.markdown("*Comparaison avec la moyenne de la strate officielle (2019-2024)*")
        
        with st.spinner("Chargement des données historiques..."):
            df_historical = fetch_historical_commune_data(commune_selectionnee, dept_selection)
        
        if not df_historical.empty and len(df_historical) > 1:
            df_historical_kpi = calculate_historical_kpis(df_historical)
            
            # === ONGLETS POUR LES DEUX VISUALISATIONS ===
            tab_score_global, tab_score_stacked, tab_score_lines = st.tabs([
                "📊 Score Global",
                "📦 Stacked Bar (Composantes)",
                "📈 Lignes (Comparaison Composantes)"
            ])
            
            with tab_score_global:
                # ✅ NOUVEAU GRAPHIQUE D'ÉVOLUTION DU SCORE GLOBAL
                fig_score_evolution = create_score_evolution_chart(df_historical_kpi, commune_selectionnee)
                
                if fig_score_evolution:
                    st.plotly_chart(fig_score_evolution, use_container_width=True, key="score_global_chart")
            
            with tab_score_stacked:
                st.markdown("**Visualisation en barres empilées**")
                st.markdown("*Chaque couleur représente la contribution d'une composante au score total*")
                
                fig_stacked = create_score_evolution_stacked_bar(df_historical_kpi, commune_selectionnee)
                if fig_stacked:
                    st.plotly_chart(fig_stacked, use_container_width=True, key="score_stacked_chart")
                    
                    st.info("""
                    📌 **Interprétation** :
                    - La hauteur totale de la barre = Score global (/100)
                    - Chaque segment = Contribution d'une composante
                    - **TEB (bleu)** : Capacité à dégager de l'épargne
                    - **Annuité/CAF (orange)** : Part des dettes dans les recettes
                    - **CD (vert)** : Temps pour rembourser la dette
                    - **FDR (rouge)** : Liquidité et jours de fonctionnement
                    """)
            
            with tab_score_lines:
                st.markdown("**Visualisation en lignes**")
                st.markdown("*La ligne noire épaisse = Score global | Les lignes pointillées = Santé de chaque composante (0-100%)*")
                
                fig_lines = create_score_evolution_lines(df_historical_kpi, commune_selectionnee)
                if fig_lines:
                    st.plotly_chart(fig_lines, use_container_width=True, key="score_lines_chart")
                    
                    st.info("""
                    📌 **Interprétation** :
                    - **Ligne noire épaisse** : Score global de la commune (/100)
                    - **Lignes pointillées** : "Santé" de chaque composante (0 = mauvais, 100 = excellent)
                    - Permet de voir LEQUEL des 4 critères tire le score vers le bas ou vers le haut
                    - Zones colorées : Vert (bon), Orange (vigilance), Rouge (critique)
                    """)
            
            # === MÉTRIQUES D'ÉVOLUTION ===
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            if len(df_historical_kpi) >= 2:
                evolution_teb = df_historical_kpi.iloc[-1]['TEB Commune (%)'] - df_historical_kpi.iloc[0]['TEB Commune (%)']
                evolution_cd = df_historical_kpi.iloc[-1]['Années de Désendettement'] - df_historical_kpi.iloc[0]['Années de Désendettement']
                evolution_annuite = df_historical_kpi.iloc[-1]['Annuité/CAF Commune (%)'] - df_historical_kpi.iloc[0]['Annuité/CAF Commune (%)']
                
                # Évolution du score
                evolution_score = df_historical_kpi.iloc[-1]['Score Commune'] - df_historical_kpi.iloc[0]['Score Commune']
                
                if pd.notna(df_historical_kpi.iloc[-1].get('FDR Jours Commune')) and pd.notna(df_historical_kpi.iloc[0].get('FDR Jours Commune')):
                    evolution_fdr = df_historical_kpi.iloc[-1]['FDR Jours Commune'] - df_historical_kpi.iloc[0]['FDR Jours Commune']
                else:
                    evolution_fdr = None
                
                with col1:
                    delta_color = "normal" if evolution_score >= 0 else "inverse"
                    st.metric("🎯 Évolution Score", f"{evolution_score:+.1f} pts", 
                             delta=f"{evolution_score:+.1f} pts", delta_color=delta_color)
                
                with col2:
                    delta_color = "normal" if evolution_teb >= 0 else "inverse"
                    st.metric("📈 Évolution TEB", f"{evolution_teb:+.1f}%", 
                             delta=f"{evolution_teb:+.1f}pp", delta_color=delta_color)
                
                with col3:
                    delta_color = "inverse" if evolution_cd >= 0 else "normal"
                    st.metric("⏳ Évolution CD", f"{evolution_cd:+.1f} ans", 
                             delta=f"{evolution_cd:+.1f} ans", delta_color=delta_color)
                
                with col4:
                    delta_color = "inverse" if evolution_annuite >= 0 else "normal"
                    st.metric("💳 Évolution Annuité/CAF", f"{evolution_annuite:+.1f}%", 
                             delta=f"{evolution_annuite:+.1f}pp", delta_color=delta_color)
            
            # === AUTRES GRAPHIQUES KPI ===
            st.markdown("---")
            st.subheader("📊 Évolution des indicateurs individuels")
            
            # Création des graphiques d'évolution
            fig_teb, fig_cd, fig_annuite, fig_fdr = create_evolution_charts(df_historical_kpi, commune_selectionnee)
            
            # Affichage des graphiques d'évolution
            col1, col2 = st.columns(2)
            
            with col1:
                if fig_teb:
                    st.plotly_chart(fig_teb, use_container_width=True, key="evolution_teb_chart")
                if fig_annuite:
                    st.plotly_chart(fig_annuite, use_container_width=True, key="evolution_annuite_chart")
            
            with col2:
                if fig_cd:
                    st.plotly_chart(fig_cd, use_container_width=True, key="evolution_cd_chart")
                if fig_fdr:
                    st.plotly_chart(fig_fdr, use_container_width=True, key="evolution_fdr_chart")
            
            # === TABLEAU RÉCAPITULATIF AVEC STYLING ===
        st.markdown("---")
        st.subheader("📋 Tableau récapitulatif pluriannuel")

        colonnes_evolution = [
            'Année', 'Population', 
            'Score Commune',
            'TEB Commune (%)', 'TEB Strate (%)',
            'Années de Désendettement', 'CD Strate (années)', 
            'Annuité/CAF Commune (%)', 'Annuité/CAF Strate (%)',
            'FDR Jours Commune', 'FDR Jours Moyenne'
        ]

        # Vérifier quelles colonnes existent
        colonnes_disponibles = [col for col in colonnes_evolution if col in df_historical_kpi.columns]

        # Créer une copie clean
        df_display = df_historical_kpi[colonnes_disponibles].copy()
        df_display = df_display.reset_index(drop=True)
        df_display = df_display.loc[:, ~df_display.columns.duplicated()].copy()
        df_display = df_display.round(2)

        # Fonction de coloration SIMPLE et ROBUSTE
        def color_cells(val, col_name):
            """Retourne une couleur CSS basée sur la valeur et la colonne"""
            
            if pd.isna(val):
                return 'background-color: #f0f0f0'
            
            try:
                val = float(val)
            except (ValueError, TypeError):
                return ''
            
            # TEB
            if 'TEB' in col_name:
                if val >= 20:
                    return 'background-color: #90EE90'  # Vert clair
                elif val >= 10:
                    return 'background-color: #FFFFE0'  # Jaune clair
                else:
                    return 'background-color: #FFB6C6'  # Rose clair
            
            # Années de Désendettement / CD
            elif 'Désendettement' in col_name or 'CD' in col_name:
                if val <= 8:
                    return 'background-color: #90EE90'
                elif val <= 12:
                    return 'background-color: #FFFFE0'
                else:
                    return 'background-color: #FFB6C6'
            
            # Annuité/CAF
            elif 'Annuité' in col_name:
                if val < 50:
                    return 'background-color: #90EE90'
                elif val < 60:
                    return 'background-color: #FFFFE0'
                else:
                    return 'background-color: #FFB6C6'
            
            # FDR Jours
            elif 'FDR' in col_name:
                if val > 240:
                    return 'background-color: #90EE90'
                elif val >= 60:
                    return 'background-color: #FFFFE0'
                else:
                    return 'background-color: #FFB6C6'
            
            # Score
            elif 'Score' in col_name:
                if val >= 75:
                    return 'background-color: #90EE90'
                elif val >= 50:
                    return 'background-color: #FFFFE0'
                else:
                    return 'background-color: #FFB6C6'
            
            return ''

        # Appliquer le styling avec une méthode robuste
        try:
            styled = df_display.style
            
            # Appliquer colonne par colonne pour éviter les conflits
            for col in df_display.columns:
                styled = styled.map(
                    lambda val, col_name=col: color_cells(val, col_name),
                    subset=[col]
                )
            
            st.dataframe(styled, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Erreur d'affichage avec styles : {e}")
            st.dataframe(df_display, use_container_width=True)

        # Ajouter une légende
        st.markdown("""
        ### 📌 Guide d'interprétation :

        | Indicateur | 🟢 Bon | 🟠 À surveiller | 🔴 Critique |
        |---|---|---|---|
        | **TEB (%)** | ≥ 20% | 10-20% | < 10% |
        | **CD (années)** | ≤ 8 ans | 8-12 ans | > 12 ans |
        | **Annuité/CAF (%)** | < 50% | 50-60% | ≥ 60% |
        | **FDR (jours)** | > 240j | 60-240j | < 60j |
        | **Score (/100)** | ≥ 75 | 50-75 | < 50 |
        """)
        
        # === EXPORT PDF ===
        st.markdown("---")
        st.subheader("💾 Export Rapport PDF")
        st.markdown("*Téléchargez un rapport professionnel avec page de garde, sommaire et graphiques*")
        
        col_pdf_1, col_pdf_2 = st.columns([3, 1])
        
        with col_pdf_1:
            if st.button("📄 Générer Rapport PDF Complet", key="gen_pdf_button"):
                with st.spinner("⏳ Génération du PDF en cours..."):
                    pdf_data = export_commune_analysis_to_pdf_enhanced(
                        commune_data=commune_data,
                        df_historical_kpi=df_historical_kpi,
                        commune_name=commune_selectionnee,
                        dept_selection=dept_selection,
                        annee_selection=annee_selection,
                        df_filtered=df_filtered
                    )
                
                if pdf_data:
                    st.success("✅ PDF généré avec succès !")
                    
                    st.download_button(
                        label="📥 Télécharger le rapport PDF",
                        data=pdf_data,
                        file_name=f"rapport_{commune_selectionnee.replace(' ', '_')}_{annee_selection}.pdf",
                        mime="application/pdf",
                        key="download_pdf_button"
                    )
                else:
                    st.error("❌ Erreur lors de la génération du PDF")
        
        with col_pdf_2:
            st.info("""
            📄 **Contenu du rapport:**
            • Page de garde
            • Résumé exécutif
            • Indicateurs clés
            • Graphiques principaux
            • Tableau récapitulatif
            • Conclusions
            """)
        
        # ============================================================
        # FIN DE LA SECTION PDF DOWNLOAD BUTTON
        # ============================================================
        
        
        
        # === TABLEAUX DÉTAILLÉS ===
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["📊 Tableau KPI complet", "📋 Données brutes"])
        
        with tab1:
            colonnes_kpi = [
                "Commune", "Population", 
                "TEB (%)", "Années de Désendettement", 
                "Annuité / CAF (%)", "FDR Jours Commune",
                "Rigidité (%)", "Score", "Niveau d'alerte"
            ]
            
            # Vérifier les colonnes disponibles
            colonnes_disponibles = [col for col in colonnes_kpi if col in df_filtered.columns]
            
            def color_niveau(val):
                if "Rouge" in str(val):
                    return 'background-color: #FFE6E6'
                elif "Orange" in str(val):
                    return 'background-color: #FFF4E6'
                else:
                    return 'background-color: #E6F7E6'
            
            styled_df = df_filtered[colonnes_disponibles].style.applymap(color_niveau, subset=['Niveau d\'alerte'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Légende des seuils
            st.markdown("---")
            st.markdown("**📌 Légende des seuils (Nouveau système de scoring V2) :**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**TEB** (20 pts)")
                st.markdown("- 🟢 Vert : > 15%")
                st.markdown("- 🟠 Orange : 8-15%")
                st.markdown("- 🔴 Rouge : < 8%")
            with col2:
                st.markdown("**CD** (30 pts)")
                st.markdown("- 🟢 Vert : < 8 ans")
                st.markdown("- 🟠 Orange : 8-12 ans")
                st.markdown("- 🔴 Rouge : > 12 ans")
            with col3:
                st.markdown("**Annuité/CAF** (30 pts)")
                st.markdown("- 🟢 Vert : < 50%")
                st.markdown("- 🟠 Orange : 50-60%")
                st.markdown("- 🔴 Rouge : > 60%")
            with col4:
                st.markdown("**FDR Jours** (20 pts)")
                st.markdown("- 🟢 Vert : > 240j")
                st.markdown("- 🟠 Orange : 60-240j")
                st.markdown("- 🔴 Rouge : < 60j")
        
        with tab2:
            st.dataframe(df_filtered, use_container_width=True)
        
        # === EXPORT ===
        st.markdown("---")
        st.subheader("💾 Export des données")
        
        col1, col2 = st.columns(2)
        
        with col1:
            excel_data = create_excel_export(df_filtered)
            if excel_data:
                file_extension = ".xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                
                try:
                    if excel_data.decode('utf-8-sig').startswith('Commune') or excel_data.decode('utf-8').startswith('Commune'):
                        file_extension = ".csv"
                        mime_type = "text/csv"
                except:
                    pass
                
                st.download_button(
                    label=f"📥 Télécharger {'Excel' if file_extension == '.xlsx' else 'CSV'}",
                    data=excel_data,
                    file_name=f"analyse_finances_v2_{dept_selection}_{annee_selection}{file_extension}",
                    mime=mime_type
                )
            else:
                st.error("Impossible de créer le fichier d'export")
        
        with col2:
            csv_data = df_filtered.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger CSV",
                data=csv_data,
                file_name=f"analyse_finances_v2_{dept_selection}_{annee_selection}.csv",
                mime="text/csv"
            )
        
        # === SYNTHÈSE ===
        st.markdown("---")
        st.subheader("📋 Synthèse départementale")
        
        synthese_col1, synthese_col2, synthese_col3 = st.columns(3)
        
        with synthese_col1:
            st.markdown("**🟢 Communes saines**")
            communes_vertes = len(df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Vert')])
            st.markdown(f"- Nombre : {communes_vertes}")
            st.markdown(f"- % : {communes_vertes/len(df_filtered)*100:.1f}%")
            if communes_vertes > 0:
                score_vert = df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Vert')]['Score'].mean()
                st.markdown(f"- Score moyen : {score_vert:.1f}/100")
        
        with synthese_col2:
            st.markdown("**🟠 Communes sous surveillance**")
            communes_orange = len(df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Orange')])
            st.markdown(f"- Nombre : {communes_orange}")
            st.markdown(f"- % : {communes_orange/len(df_filtered)*100:.1f}%")
            if communes_orange > 0:
                score_orange = df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Orange')]['Score'].mean()
                st.markdown(f"- Score moyen : {score_orange:.1f}/100")
        
        with synthese_col3:
            st.markdown("**🔴 Communes à risque**")
            communes_rouges = len(df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Rouge')])
            st.markdown(f"- Nombre : {communes_rouges}")
            st.markdown(f"- % : {communes_rouges/len(df_filtered)*100:.1f}%")
            if communes_rouges > 0:
                score_rouge = df_filtered[df_filtered['Niveau d\'alerte'].str.contains('Rouge')]['Score'].mean()
                st.markdown(f"- Score moyen : {score_rouge:.1f}/100")
        
        # === STATISTIQUES DÉTAILLÉES ===
        st.markdown("---")
        st.subheader("📈 Statistiques détaillées des indicateurs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Moyennes départementales**")
            stats_df = pd.DataFrame({
                'Indicateur': ['TEB (%)', 'Années de Désendettement', 'Annuité/CAF (%)', 'FDR (jours)', 'Score (/100)'],
                'Moyenne': [
                    df_filtered['TEB (%)'].mean(),
                    df_filtered['Années de Désendettement'].mean(),
                    df_filtered['Annuité / CAF (%)'].mean(),
                    df_filtered['FDR Jours Commune'].mean() if 'FDR Jours Commune' in df_filtered.columns else None,
                    df_filtered['Score'].mean()
                ],
                'Médiane': [
                    df_filtered['TEB (%)'].median(),
                    df_filtered['Années de Désendettement'].median(),
                    df_filtered['Annuité / CAF (%)'].median(),
                    df_filtered['FDR Jours Commune'].median() if 'FDR Jours Commune' in df_filtered.columns else None,
                    df_filtered['Score'].median()
                ]
            }).round(2)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**🎯 Répartition par critère**")
            
            # TEB
            teb_vert = len(df_filtered[df_filtered['TEB (%)'] > 15])
            teb_orange = len(df_filtered[(df_filtered['TEB (%)'] >= 8) & (df_filtered['TEB (%)'] <= 15)])
            teb_rouge = len(df_filtered[df_filtered['TEB (%)'] < 8])
            
            # CD
            cd_vert = len(df_filtered[df_filtered['Années de Désendettement'] < 8])
            cd_orange = len(df_filtered[(df_filtered['Années de Désendettement'] >= 8) & (df_filtered['Années de Désendettement'] <= 12)])
            cd_rouge = len(df_filtered[df_filtered['Années de Désendettement'] > 12])
            
            # Annuité/CAF
            ann_vert = len(df_filtered[df_filtered['Annuité / CAF (%)'] < 50])
            ann_orange = len(df_filtered[(df_filtered['Annuité / CAF (%)'] >= 50) & (df_filtered['Annuité / CAF (%)'] <= 60)])
            ann_rouge = len(df_filtered[df_filtered['Annuité / CAF (%)'] > 60])
            
            # FDR
            if 'FDR Jours Commune' in df_filtered.columns:
                fdr_vert = len(df_filtered[df_filtered['FDR Jours Commune'] > 240])
                fdr_orange = len(df_filtered[(df_filtered['FDR Jours Commune'] >= 60) & (df_filtered['FDR Jours Commune'] <= 240)])
                fdr_rouge = len(df_filtered[df_filtered['FDR Jours Commune'] < 60])
            else:
                fdr_vert = fdr_orange = fdr_rouge = 0
            
            repartition_df = pd.DataFrame({
                'Critère': ['TEB', 'CD', 'Annuité/CAF', 'FDR'],
                '🟢 Vert': [teb_vert, cd_vert, ann_vert, fdr_vert],
                '🟠 Orange': [teb_orange, cd_orange, ann_orange, fdr_orange],
                '🔴 Rouge': [teb_rouge, cd_rouge, ann_rouge, fdr_rouge]
            })
            st.dataframe(repartition_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("**📌 Nouveau système de scoring V3**")
st.markdown("*Données : API des comptes individuels des communes - data.economie.gouv.fr*")
st.markdown("*Scoring basé sur : TEB (20%), CD (30%), Annuité/CAF (30%), FDR (20%)*")
st.markdown("SFP COLLECTIVITÉS")
st.markdown("**© 2025 SFP COLLECTIVITÉS. Reproduction intégrale ou partielle interdite sauf autorisation écrite. Fichier protégé — toute modification est strictement interdite**")