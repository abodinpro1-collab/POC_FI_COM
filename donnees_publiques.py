# -*- coding: utf-8 -*-
"""
Module : donnees_publiques.py
Enrichissement des communes avec :
  - Potentiel fiscal (OFGL / DGCL)
  - Potentiel financier (OFGL / DGCL)
  - Valeur cadastrale / DVF (Cerema / data.gouv)

Sources :
  - https://data.ofgl.fr/api/explore/v2.1/catalog/datasets/dotations-communes/records
  - https://apidf-preprod.cerema.fr/dvf_opendata/communes/{code_insee}/
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache
import time

# ============================================================
# CONSTANTES
# ============================================================

OFGL_API_BASE = "https://data.ofgl.fr/api/explore/v2.1/catalog/datasets"
CEREMA_DVF_BASE = "https://apidf-preprod.cerema.fr/dvf_opendata/communes"

# Colonnes OFGL dotations-communes contenant pfi (potentiel fiscal) et pfin (potentiel financier)
# Source : http://www.dotations-dgcl.interieur.gouv.fr
OFGL_DOTATIONS_DATASET = "dotations-communes"

# Colonnes REI (Recensement Éléments d'Imposition) pour bases fiscales
OFGL_REI_DATASET = "rei"


# ============================================================
# 1. RÉCUPÉRATION POTENTIEL FISCAL & FINANCIER (OFGL)
# ============================================================

@st.cache_data(ttl=86400)  # Cache 24h : données annuelles stables
def fetch_potentiel_fiscal_financier(code_insee: str, annee: int) -> dict:
    """
    Récupère le potentiel fiscal et financier d'une commune.

    Source : OFGL dotations-communes (format long, une ligne par variable).

    Args:
        code_insee : code INSEE 5 caractères (ex: "35238")
        annee      : année (2018-2025)

    Returns:
        dict avec les clés :
          potentiel_fiscal_hab, potentiel_financier_hab,
          effort_fiscal, annee_donnee, source
        ou dict vide si non trouvé.
    """
    url = f"{OFGL_API_BASE}/{OFGL_DOTATIONS_DATASET}/records"

    def _query(an: int) -> dict | None:
        values = {}
        for var_label, short in OFGL_VARIABLES_MAP.items():
            try:
                params = {
                    "where": (
                        f'code_insee="{code_insee}" '
                        f'AND year(exercice)={an} '
                        f'AND variable="{var_label}"'
                    ),
                    "limit": 1,
                    "select": "valeur",
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("results"):
                    values[short] = _safe_float(data["results"][0].get("valeur"))
            except Exception:
                continue
        return values if values else None

    # Tentative année cible puis N-1
    for an in [annee, annee - 1]:
        if an < 2018:
            continue
        values = _query(an)
        if values:
            return {
                "potentiel_fiscal_hab":    values.get("pfi"),
                "potentiel_financier_hab": values.get("pfin"),
                "effort_fiscal":           values.get("ef"),
                "annee_donnee":            an,
                "source":                  f"OFGL/DGCL dotations-communes ({an})",
            }

    return {}


# Mapping des variables OFGL → noms de colonnes courts
OFGL_VARIABLES_MAP = {
    "Potentiel fiscal 4 taxes par hab.": "pfi",
    "Potentiel financier par hab.":      "pfin",
    "Effort fiscal":                     "ef",
}


def _normalize_dep_for_ofgl(dep: str):
    """
    OFGL stocke code_departement en texte sur 2 caractères
    ("01", "35", "2A", "974"...). Retourne la valeur entre guillemets
    prête à insérer dans une clause where.
    """
    dep_clean = str(dep).strip()
    if dep_clean in ("2A", "2B"):
        return f'"{dep_clean}"'
    # Métropole : zfill(2). DOM (971-976) : laissés tels quels sur 3 caractères.
    try:
        n = int(dep_clean)
        if n < 100:
            return f'"{n:02d}"'  # métropole : "01", "35"...
        else:
            return f'"{n:03d}"'  # DOM : "971"...
    except ValueError:
        return f'"{dep_clean}"'


@st.cache_data(ttl=86400)
def fetch_potentiel_fiscal_financier_departement(dep: str, annee: int) -> pd.DataFrame:
    """
    Récupère le potentiel fiscal et financier pour toutes les communes d'un département.
    Le dataset OFGL dotations-communes est en format long (une ligne par variable),
    donc on fait une requête par indicateur puis on pivote.

    Args:
        dep   : code département (ex: "035", "2A")
        annee : année

    Returns:
        DataFrame wide avec colonnes :
          insee_com, pfi, pfin, ef, annee_donnee
    """
    dep_filter = _normalize_dep_for_ofgl(dep)

    # Essayer année cible, puis N-1
    for an in [annee, annee - 1]:
        df_variables = {}
        for var_label, short_name in OFGL_VARIABLES_MAP.items():
            frames = []
            offset = 0
            limit = 100
            try:
                while True:
                    url = f"{OFGL_API_BASE}/{OFGL_DOTATIONS_DATASET}/records"
                    params = {
                        "where": (
                            f'code_departement={dep_filter} '
                            f'AND year(exercice)={an} '
                            f'AND variable="{var_label}"'
                        ),
                        "limit": limit,
                        "offset": offset,
                        "select": "code_insee,valeur",
                    }
                    resp = requests.get(url, params=params, timeout=15)
                    resp.raise_for_status()
                    data = resp.json()

                    if not data.get("results"):
                        break

                    frames.append(pd.DataFrame(data["results"]))

                    if len(data["results"]) < limit:
                        break
                    offset += limit

                if frames:
                    df_var = pd.concat(frames, ignore_index=True)
                    df_var[short_name] = pd.to_numeric(df_var["valeur"], errors="coerce")
                    df_var = df_var[["code_insee", short_name]].rename(columns={"code_insee": "insee_com"})
                    df_variables[short_name] = df_var
            except Exception:
                continue

        # Si aucun indicateur récupéré pour cette année, tenter N-1
        if not df_variables:
            continue

        # Fusionner les 3 dataframes wide
        df_result = None
        for short_name, df_var in df_variables.items():
            if df_result is None:
                df_result = df_var
            else:
                df_result = df_result.merge(df_var, on="insee_com", how="outer")

        if df_result is not None and not df_result.empty:
            # S'assurer que toutes les colonnes existent même si absentes
            for col in ("pfi", "pfin", "ef"):
                if col not in df_result.columns:
                    df_result[col] = pd.NA
            df_result["annee_donnee"] = an
            return df_result

    return pd.DataFrame()


# ============================================================
# 2. RÉCUPÉRATION BASES FISCALES REI (DGFiP via OFGL)
# ============================================================

@st.cache_data(ttl=86400)
def fetch_bases_fiscales_departement(dep: str, annee: int) -> pd.DataFrame:
    """
    Récupère les bases fiscales (TFB, TFNB) par commune via le REI.
    Utile pour approcher la valeur locative cadastrale agrégée.

    Champs principaux :
      - base_tf  : base taxe foncière bâti (K€)
      - base_tfnb: base taxe foncière non bâti (K€)
      - taux_tf  : taux voté TFB commune (%)

    Args:
        dep   : code département
        annee : 2023 ou 2024 (seules années REI disponibles sur OFGL)
    """
    # Le REI OFGL ne couvre que 2023-2024
    annee_rei = annee if annee in (2023, 2024) else 2023

    frames = []
    limit = 100
    offset = 0
    dep_filter = _normalize_dep_for_ofgl(dep)

    try:
        while True:
            url = f"{OFGL_API_BASE}/{OFGL_REI_DATASET}/records"
            params = {
                "where": (
                    f'dep={dep_filter} AND annee={annee_rei} '
                    'AND dispositif_fiscal="FB" '
                    'AND categorie="Base" '
                    'AND destinataire="Commune" '
                    'AND varlib="FB - COMMUNE / BASE NETTE"'
                ),
                "limit": limit,
                "offset": offset,
                "select": "idcom,valeur,annee",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("results"):
                break

            frames.append(pd.DataFrame(data["results"]))

            if len(data["results"]) < limit:
                break
            offset += limit

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.rename(columns={"idcom": "insee_com", "valeur": "base_tfb_ke"})
            # REI fournit les valeurs en €, conversion en K€
            df["base_tfb_ke"] = pd.to_numeric(df["base_tfb_ke"], errors="coerce") / 1000
            df["annee_rei"] = annee_rei
            return df[["insee_com", "base_tfb_ke", "annee_rei"]]

    except Exception:
        pass

    return pd.DataFrame()


# ============================================================
# 3. RÉCUPÉRATION VALEUR FONCIÈRE DVF (Cerema)
# ============================================================

@st.cache_data(ttl=86400)
def fetch_dvf_commune(code_insee: str) -> dict:
    """
    Récupère les indicateurs DVF (Demande de Valeurs Foncières) agrégés
    à la commune via l'API Cerema DVF+ open data.

    Retourne le prix médian au m² et le nombre de mutations
    pour les maisons et appartements (5 dernières années).

    Args:
        code_insee : code INSEE 5 caractères

    Returns:
        dict avec :
          prix_median_maison_m2, nb_mutations_maison,
          prix_median_appart_m2, nb_mutations_appart,
          annee_debut, annee_fin, source
    """
    result = {}
    try:
        # Endpoint DVF+ open data Cerema — agrégé commune
        url = f"{CEREMA_DVF_BASE}/{code_insee}/"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 404:
            return {}

        resp.raise_for_status()
        data = resp.json()

        # Extraire maisons
        maisons = [
            d for d in data.get("results", [])
            if d.get("type_local") == "Maison"
        ]
        appartements = [
            d for d in data.get("results", [])
            if d.get("type_local") == "Appartement"
        ]

        def median_prix(items):
            prix = [
                d.get("prix_m2_median") for d in items
                if d.get("prix_m2_median") is not None
            ]
            return round(float(np.median(prix)), 0) if prix else None

        def total_mutations(items):
            return sum(d.get("nb_mutations", 0) or 0 for d in items)

        annees = [d.get("annee") for d in data.get("results", []) if d.get("annee")]

        result = {
            "dvf_prix_median_maison_m2":    median_prix(maisons),
            "dvf_nb_mutations_maison":       total_mutations(maisons),
            "dvf_prix_median_appart_m2":    median_prix(appartements),
            "dvf_nb_mutations_appart":       total_mutations(appartements),
            "dvf_annee_debut":               min(annees) if annees else None,
            "dvf_annee_fin":                 max(annees) if annees else None,
            "dvf_source":                    "Cerema DVF+ open data"
        }

    except Exception:
        pass

    return result


@st.cache_data(ttl=86400)
def fetch_dvf_departement(dep: str) -> pd.DataFrame:
    """
    Récupère les indicateurs DVF pour toutes les communes d'un département
    via l'API Cerema DVF+ indicateurs territoires.

    Endpoint : GET /indicateurs/dv3f/communes/?code_departement={dep}

    Args:
        dep : code département (ex: "35", "2A")

    Returns:
        DataFrame avec colonnes :
          code_insee, prix_median_m2_maison, nb_ventes_maison,
          prix_median_m2_appart, nb_ventes_appart
    """
    dep_norm = dep.lstrip("0") if dep not in ("2A", "2B") else dep

    try:
        # Endpoint indicateurs agrégés
        url = "https://apidf-preprod.cerema.fr/indicateurs/dv3f/communes/"
        params = {
            "code_departement": dep_norm,
            "periode": "5ans"   # 5 dernières années
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("features"):
            return pd.DataFrame()

        rows = []
        for feature in data["features"]:
            props = feature.get("properties", {})
            rows.append({
                "code_insee":              props.get("code_insee") or props.get("codgeo"),
                "dvf_prix_median_maison_m2": _safe_float(props.get("prix_m2_median_maison")),
                "dvf_nb_mutations_maison":   _safe_int(props.get("nb_ventes_maison")),
                "dvf_prix_median_appart_m2": _safe_float(props.get("prix_m2_median_appart")),
                "dvf_nb_mutations_appart":   _safe_int(props.get("nb_ventes_appart")),
                "dvf_source":               "Cerema DVF+ indicateurs territoires"
            })

        return pd.DataFrame(rows)

    except Exception:
        # Fallback silencieux : DVF indisponible n'est pas bloquant
        return pd.DataFrame()


# ============================================================
# 4. ENRICHISSEMENT GLOBAL D'UN DATAFRAME DE COMMUNES
# ============================================================

def enrich_communes_with_public_data(
    df: pd.DataFrame,
    dep: str,
    annee: int,
    code_insee_col: str = "code_insee",
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Enrichit un DataFrame de communes avec les 3 sources publiques :
      1. Potentiel fiscal & financier (OFGL/DGCL)
      2. Bases fiscales REI (DGFiP)
      3. Valeur foncière DVF (Cerema)

    Le join se fait sur le code INSEE (colonne `code_insee_col`).
    Si la colonne n'existe pas, elle est construite depuis dep + code_commune.

    Args:
        df             : DataFrame communes (issu de fetch_communes)
        dep            : code département
        annee          : année d'analyse
        code_insee_col : nom de la colonne code INSEE dans df
        show_progress  : afficher spinner Streamlit

    Returns:
        DataFrame enrichi avec nouvelles colonnes publiques.
    """
    df = df.copy()

    # --- Construire le code INSEE si absent ---
    if code_insee_col not in df.columns:
        df = _build_code_insee(df, dep)
        code_insee_col = "code_insee"

    # Normaliser : 5 caractères
    df[code_insee_col] = df[code_insee_col].astype(str).str.strip().str.zfill(5)

    # ---- 1. Potentiel fiscal & financier ----
    if show_progress:
        with st.spinner("🏛️ Chargement potentiel fiscal & financier (OFGL/DGCL)..."):
            df_potentiel = fetch_potentiel_fiscal_financier_departement(dep, annee)
    else:
        df_potentiel = fetch_potentiel_fiscal_financier_departement(dep, annee)

    if not df_potentiel.empty and "insee_com" in df_potentiel.columns:
        df_potentiel["insee_com"] = df_potentiel["insee_com"].astype(str).str.strip().str.zfill(5)
        cols_potentiel = ["insee_com", "pfi", "pfin", "ef", "pop_dgf", "annee_donnee"]
        cols_ok = [c for c in cols_potentiel if c in df_potentiel.columns]
        df = df.merge(
            df_potentiel[cols_ok].rename(columns={
                "insee_com":    code_insee_col,
                "pfi":          "Potentiel Fiscal (€/hab)",
                "pfin":         "Potentiel Financier (€/hab)",
                "ef":           "Effort Fiscal",
                "pop_dgf":      "Pop DGF",
                "annee_donnee": "Année Potentiel"
            }),
            on=code_insee_col,
            how="left"
        )
    else:
        df["Potentiel Fiscal (€/hab)"]    = pd.NA
        df["Potentiel Financier (€/hab)"] = pd.NA
        df["Effort Fiscal"]               = pd.NA
        df["Pop DGF"]                     = pd.NA
        df["Année Potentiel"]             = pd.NA

    # ---- 2. Bases fiscales REI ----
    if show_progress:
        with st.spinner("📋 Chargement bases fiscales REI (DGFiP)..."):
            df_rei = fetch_bases_fiscales_departement(dep, annee)
    else:
        df_rei = fetch_bases_fiscales_departement(dep, annee)

    if not df_rei.empty and "insee_com" in df_rei.columns:
        df_rei["insee_com"] = df_rei["insee_com"].astype(str).str.strip().str.zfill(5)
        df = df.merge(
            df_rei[["insee_com", "base_tfb_ke", "annee_rei"]].rename(columns={
                "insee_com":    code_insee_col,
                "base_tfb_ke":  "Base TFB (K€)",
                "annee_rei":    "Année REI"
            }),
            on=code_insee_col,
            how="left"
        )
    else:
        df["Base TFB (K€)"] = pd.NA
        df["Année REI"]     = pd.NA

    # ---- 3. DVF ----
    if show_progress:
        with st.spinner("🏠 Chargement valeurs foncières DVF (Cerema)..."):
            df_dvf = fetch_dvf_departement(dep)
    else:
        df_dvf = fetch_dvf_departement(dep)

    if not df_dvf.empty and "code_insee" in df_dvf.columns:
        df_dvf["code_insee"] = df_dvf["code_insee"].astype(str).str.strip().str.zfill(5)
        df = df.merge(
            df_dvf.rename(columns={"code_insee": code_insee_col}),
            on=code_insee_col,
            how="left"
        )
    else:
        df["dvf_prix_median_maison_m2"]  = pd.NA
        df["dvf_nb_mutations_maison"]    = pd.NA
        df["dvf_prix_median_appart_m2"]  = pd.NA
        df["dvf_nb_mutations_appart"]    = pd.NA
        df["dvf_source"]                 = pd.NA

    return df


# ============================================================
# 5. HELPERS AFFICHAGE STREAMLIT
# ============================================================

def display_donnees_publiques_commune(commune_row: pd.Series):
    """
    Affiche un bloc Streamlit avec les 3 indicateurs publics
    pour une commune sélectionnée.
    """
    st.markdown("### 🏛️ Données publiques complémentaires")

    col1, col2, col3 = st.columns(3)

    # --- Potentiel fiscal ---
    with col1:
        pfi = commune_row.get("Potentiel Fiscal (€/hab)")
        pfin = commune_row.get("Potentiel Financier (€/hab)")
        ef = commune_row.get("Effort Fiscal")
        annee_pot = commune_row.get("Année Potentiel")

        st.markdown("**🏦 Potentiel fiscal & financier**")
        if pd.notna(pfi):
            st.metric(
                "Potentiel fiscal",
                f"{pfi:,.0f} €/hab",
                help="Recettes fiscales théoriques si taux moyens appliqués"
            )
        else:
            st.metric("Potentiel fiscal", "N/D")

        if pd.notna(pfin):
            st.metric(
                "Potentiel financier",
                f"{pfin:,.0f} €/hab",
                help="Potentiel fiscal + dotations de péréquation"
            )
        else:
            st.metric("Potentiel financier", "N/D")

        if pd.notna(ef):
            color = "🟢" if ef >= 1.0 else "🟠" if ef >= 0.75 else "🔴"
            st.markdown(f"**Effort fiscal** : {color} {ef:.3f}")
            st.caption("Seuil : >1.0 = bon")

        if pd.notna(annee_pot):
            st.caption(f"Source : DGCL/OFGL ({int(annee_pot)})")

    # --- Bases fiscales REI ---
    with col2:
        base_tfb = commune_row.get("Base TFB (K€)")
        annee_rei = commune_row.get("Année REI")
        pop = commune_row.get("Population", 1)

        st.markdown("**🏘️ Bases fiscales (REI)**")
        if pd.notna(base_tfb):
            st.metric(
                "Base TFB",
                f"{base_tfb:,.0f} K€",
                help="Base taxe foncière sur le bâti — proxy de la valeur locative cadastrale"
            )
            if pd.notna(pop) and pop > 0:
                base_tfb_hab = (base_tfb * 1000) / pop
                st.metric("Base TFB / hab", f"{base_tfb_hab:,.0f} €/hab")
        else:
            st.metric("Base TFB", "N/D")
            st.caption("REI disponible uniquement 2023-2024")

        if pd.notna(annee_rei):
            st.caption(f"Source : DGFiP REI via OFGL ({int(annee_rei)})")

    # --- DVF ---
    with col3:
        prix_maison = commune_row.get("dvf_prix_median_maison_m2")
        nb_maison = commune_row.get("dvf_nb_mutations_maison")
        prix_appart = commune_row.get("dvf_prix_median_appart_m2")
        nb_appart = commune_row.get("dvf_nb_mutations_appart")

        st.markdown("**🏠 Valeurs foncières (DVF)**")
        if pd.notna(prix_maison):
            st.metric(
                "Prix médian maison",
                f"{prix_maison:,.0f} €/m²",
                help=f"Sur {int(nb_maison) if pd.notna(nb_maison) else '?'} mutations"
            )
        else:
            st.metric("Prix médian maison", "N/D")

        if pd.notna(prix_appart):
            st.metric(
                "Prix médian appart",
                f"{prix_appart:,.0f} €/m²",
                help=f"Sur {int(nb_appart) if pd.notna(nb_appart) else '?'} mutations"
            )
        else:
            st.metric("Prix médian appart", "N/D")

        if pd.notna(prix_maison) or pd.notna(prix_appart):
            st.caption("Source : Cerema DVF+ (5 dernières années)")
        else:
            st.caption("DVF : communes < 5 mutations non couvertes")


def display_donnees_publiques_tableau(df: pd.DataFrame):
    """
    Affiche un tableau synthétique des indicateurs publics
    pour toutes les communes du département.
    """
    cols_public = [
        "Commune", "Population",
        "Potentiel Fiscal (€/hab)", "Potentiel Financier (€/hab)", "Effort Fiscal",
        "Base TFB (K€)",
        "dvf_prix_median_maison_m2", "dvf_prix_median_appart_m2"
    ]
    cols_ok = [c for c in cols_public if c in df.columns]
    df_display = df[cols_ok].copy()

    # Renommage pour affichage
    df_display = df_display.rename(columns={
        "dvf_prix_median_maison_m2": "DVF Maison (€/m²)",
        "dvf_prix_median_appart_m2": "DVF Appart (€/m²)"
    })

    # Coloration Effort Fiscal
    def color_ef(val):
        if pd.isna(val):
            return ""
        try:
            v = float(val)
            if v >= 1.0:
                return "background-color: #90EE90"
            elif v >= 0.75:
                return "background-color: #FFFFE0"
            else:
                return "background-color: #FFB6C6"
        except Exception:
            return ""

    styled = df_display.style
    if "Effort Fiscal" in df_display.columns:
        styled = styled.map(color_ef, subset=["Effort Fiscal"])

    st.dataframe(styled, use_container_width=True)


def get_donnees_publiques_pour_pdf(commune_row: pd.Series) -> dict:
    """
    Prépare un dict propre des données publiques pour injection dans le PDF.
    """
    return {
        "potentiel_fiscal_hab":    _safe_float(commune_row.get("Potentiel Fiscal (€/hab)")),
        "potentiel_financier_hab": _safe_float(commune_row.get("Potentiel Financier (€/hab)")),
        "effort_fiscal":           _safe_float(commune_row.get("Effort Fiscal")),
        "base_tfb_ke":             _safe_float(commune_row.get("Base TFB (K€)")),
        "dvf_maison_m2":           _safe_float(commune_row.get("dvf_prix_median_maison_m2")),
        "dvf_nb_maison":           _safe_int(commune_row.get("dvf_nb_mutations_maison")),
        "dvf_appart_m2":           _safe_float(commune_row.get("dvf_prix_median_appart_m2")),
        "dvf_nb_appart":           _safe_int(commune_row.get("dvf_nb_mutations_appart")),
        "annee_potentiel":         commune_row.get("Année Potentiel"),
    }


# ============================================================
# 6. UTILITAIRES PRIVÉS
# ============================================================

def _safe_float(val) -> float | None:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except Exception:
        return None


def _safe_int(val) -> int | None:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return int(val)
    except Exception:
        return None


def _build_code_insee(df: pd.DataFrame, dep: str) -> pd.DataFrame:
    """
    Construit le code INSEE à partir du département et d'une colonne
    de code commune si disponible, sinon via l'API geo.api.gouv.fr.
    """
    df = df.copy()

    # Si une colonne 'com' ou 'code_com' existe dans le df
    for candidate in ["com", "code_com", "code_commune", "insee"]:
        if candidate in df.columns:
            dep_padded = dep.zfill(2)
            df["code_insee"] = dep_padded + df[candidate].astype(str).str.zfill(3)
            return df

    # Fallback : résolution par nom via geo.api.gouv.fr (lent, utilisé en dernier recours)
    dep_norm = dep.lstrip("0") if dep not in ("2A", "2B") else dep
    try:
        url = f"https://geo.api.gouv.fr/departements/{dep_norm}/communes"
        params = {"fields": "nom,code", "format": "json"}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        geo_data = resp.json()
        geo_map = {item["nom"].upper(): item["code"] for item in geo_data}
        df["code_insee"] = df["Commune"].str.upper().map(geo_map)
    except Exception:
        df["code_insee"] = pd.NA

    return df
