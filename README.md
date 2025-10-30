# 📊 Analyse de la Santé Financière des Communes

> **Application de diagnostic financier professionnelle** basée sur le scoring V3 adaptatif pour les collectivités locales.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Objectifs

Fournir une analyse complète et comparative de la santé financière des communes françaises :

- **Scoring automatisé** (0-100) basé sur 4 indicateurs clés
- **Comparaison avec la strate officielle** de chaque commune
- **Détection d'anomalies** et identification des risques
- **Visualisations interactives** et rapports PDF professionnels
- **Export de données** en Excel/CSV pour analyses personnalisées

---

## 🚀 Démarrage Rapide

### Installation

```bash
# Cloner le repository
git clone <repository-url>
cd analyse-communes-finances

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Lancer l'application

```bash
streamlit run claude.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## 📋 Fonctionnalités Principales

### 1. **Analyse Départementale** 🏘️

- Sélection du département et de l'année
- Filtrage par population minimale
- Tableau de bord avec métriques agrégées
- Graphiques interactifs (Plotly)

**KPIs affichés :**
- Nombre de communes analysées
- Score moyen de santé
- Population totale
- Pourcentage de communes fragiles

### 2. **Scoring V3 Adaptatif** 🎯

Système de scoring propriétaire sur 100 points :

| Composante | Pondération | Seuil Vert | Seuil Orange | Seuil Rouge |
|-----------|-------------|-----------|------------|-----------|
| **TEB** (Taux d'Épargne Brute) | 20% | > 15% | 10-15% | < 10% |
| **CD** (Capacité Désendettement) | 30% | < 8 ans | 8-12 ans | > 12 ans |
| **Annuité/CAF** | 30% | < 50% | 50-60% | > 60% |
| **FDR** (Fonds de Roulement) | 20% | > 240j | 60-240j | < 60j |

**Score Global :** 
- 🟢 **Vert** : 75-100 (Situation saine)
- 🟠 **Orange** : 50-75 (À surveiller)
- 🔴 **Rouge** : 0-50 (Fragile)

### 3. **Visualisations Interactives** 📊

#### Graphiques Statistiques
- **Pie Chart** : Répartition des niveaux d'alerte
- **Histogramme** : Distribution des scores
- **Scatter Plots** : Comparaisons multidimensionnelles
- **Box Plots** : Analyses par catégories

#### Graphiques Spécialisés
- **Radar Cohérent** : Profil financier 360°
- **Score Evolution** : Tendance pluriannuelle
- **Stacked Bar** : Contribution des composantes
- **Lignes Détaillées** : Évolution de chaque indicateur

### 4. **Analyse Commune Détaillée** 🔍

Sélectionner une commune pour accéder à :

- **Données consolidées** : Tous les KPIs
- **Radar comparatif** : Versus strate officielle
- **Historique pluriannuel** : 2019-2024
- **Tableau de normalisation** : Transformation des données
- **Graphiques individuels** : 4 indicateurs distincts

### 5. **Export Professionnel** 📄

#### PDF Complet
Rapport généré avec :
- ✅ Page de garde personnalisée
- ✅ Synthèse exécutive
- ✅ Indicateurs clés (tableau)
- ✅ Profil radar
- ✅ Graphiques pluriannuels (6 pages min.)
- ✅ Tableaux récapitulatifs
- ✅ En-têtes et pieds de page

#### Excel/CSV
- Tableau complet avec formatage conditionnel
- Feuille "Synthèse" avec agrégations
- Codage couleur par niveau d'alerte

### 6. **Classements Top/Flop** 🏆

- **Top 25 Fragiles** : Communes avec score ≤ P25
- **Top 25 Solides** : Communes avec score ≥ P75
- Colonnes affichées : Commune, Population, Score, TEB, CD, Annuité/CAF, FDR

---

## 🔧 Architecture Technique

### Stack Technologique

```
Frontend     : Streamlit
Backend      : Python 3.8+
Visualisation: Plotly, Matplotlib, Seaborn
Données      : Pandas, NumPy
API          : data.economie.gouv.fr
Export       : ReportLab (PDF), openpyxl (Excel)
```

### Structure du Code

```
claude.py
├── Configuration Streamlit
├── Classes & Fetchers
│   └── RobustCommuneFetcher
├── Fonctions API
│   ├── fetch_communes()
│   └── fetch_historical_commune_data()
├── Calculs KPI
│   ├── score_sante_financiere_v3()
│   ├── niveau_alerte_v3()
│   └── calculate_historical_kpis()
├── Visualisations
│   ├── Plotly (interactif)
│   ├── Matplotlib/Seaborn (PDF)
│   └── Radar cohérent
├── Export
│   ├── generate_pdf_graphs()
│   └── export_commune_analysis_to_pdf_enhanced()
└── UI/Dashboards
    ├── Tableau de bord principal
    ├── Analyse détaillée
    └── Classements
```

### Flux de Données

```
┌─────────────────────────────────────────┐
│ Sélection Dept + Année + Filtres        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ API data.economie.gouv.fr            │
│ (get_api_url_for_year)               │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Traitement Données                   │
│ • Nettoyage                          │
│ • Calcul TEB, CD, Annuité/CAF, FDR   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Calcul Scoring V3                    │
│ • Normalisation                      │
│ • Pondération (20/30/30/20)          │
│ • Classification (Vert/Orange/Rouge) │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│ Visualisations & Export              │
│ • Dashboards interactifs             │
│ • PDF professionnel                  │
│ • Fichiers Excel/CSV                 │
└──────────────────────────────────────┘
```

---

## 📊 Indicateurs Financiers Expliqués

### 1. TEB (Taux d'Épargne Brute) - 20 pts
**Formule** : `CAF Brute / Recettes Réelles Fonctionnement × 100`

Mesure la capacité de la commune à dégager de l'épargne après fonctionnement.
- ✅ **Bon** : > 15% (dégagement substantiel)
- ⚠️ **À surveiller** : 8-15% (capacité modérée)
- ❌ **Critique** : < 8% (très limité)

### 2. CD (Capacité Désendettement) - 30 pts
**Formule** : `Encours Dettes / CAF Brute` (en années)

Temps nécessaire pour rembourser la dette avec l'épargne annuelle.
- ✅ **Bon** : < 8 ans (désendettement rapide)
- ⚠️ **À surveiller** : 8-12 ans (modéré)
- ❌ **Critique** : > 12 ans (endettement élevé)

### 3. Annuité/CAF - 30 pts
**Formule** : `Annuité Remboursement / CAF Brute × 100`

Part des remboursements dans l'épargne disponible.
- ✅ **Bon** : < 50% (épargne libérée)
- ⚠️ **À surveiller** : 50-60% (épargne consommée)
- ❌ **Critique** : > 60% (surengagement)

### 4. FDR (Fonds de Roulement) - 20 pts
**Formule** : `(FDR € par habitant / DRF € par habitant) × 365` (en jours)

Jours de fonctionnement garantis par la trésorerie.
- ✅ **Bon** : > 240 jours (~8 mois)
- ⚠️ **À surveiller** : 60-240 jours
- ❌ **Critique** : < 60 jours (~2 mois)

---

## 🎨 Utilisateur Guide

### Mode Analyse Rapide

1. **Barre latérale** : Sélectionner Département + Année
2. **Bouton bleu** : "Analyser le département"
3. **Résultats** : Métriques + Graphiques interactifs

### Mode Détaillé

1. Sélectionner une commune dans la liste
2. **Tabs** disponibles :
   - 📊 Score Global (évolution)
   - 📦 Stacked Bar (contribution)
   - 📈 Lignes (détails par composante)
3. **Export** : Cliquer "Générer Rapport PDF"

### Mode Debug

Cocher 🔬 "Mode Debug FDR" pour voir :
- Diagnos FDR département
- Valeurs aberrantes
- Statistiques globales

---

## 📈 Cas d'Usage

### Pour les Collectivités
✅ Diagnostiquer sa situation financière
✅ Comparer sa performance à la strate
✅ Identifier les points à améliorer
✅ Générer des rapports de synthèse

### Pour les Contrôleurs de Gestion
✅ Analyser plusieurs communes rapidement
✅ Détecter les anomalies
✅ Préparer des benchmarks
✅ Exporter pour analyses approfondies

### Pour les Élus
✅ Présenter la situation financière (PDF)
✅ Communiquer sur le score
✅ Justifier les décisions d'investissement

---

## ⚙️ Configuration & Paramétrage

### Données Disponibles

```python
DATASETS_MAPPING = {
    2019: "...2019-2020",
    2020: "...2019-2020",
    2021: "...2021",
    2022: "...2022",
    2023: "...2023-2024",
    2024: "...2023-2024"
}
```

### Seuils Personnalisables

Pour modifier les seuils, éditer les fonction :
- `score_sante_financiere_v3()` : Scoring
- `normaliser_indicateurs_pour_radar()` : Plages de normalisation

### API & Rate Limiting

- **Endpoint** : `data.economie.gouv.fr/api/explore/v2.1`
- **Pagination** : 100 records par appel
- **Cache** : 3600 secondes (1 heure)

---

## 🐛 Troubleshooting

### ⚠️ "Aucune donnée trouvée"
→ Vérifier que le département/année existe
→ Vérifier la connexion Internet
→ Consulter le Mode Debug

### ⚠️ "FDR invalide (>1000 jours)"
→ Données aberrantes plafonnées à `pd.NA`
→ Normal pour petites communes
→ Affichage : 🔬 Mode Debug

### ⚠️ "PDF généré vide"
→ Vérifier qu'au moins 2 années historiques sont disponibles
→ Graphiques individuels générés OK (check)

---

## 🔐 Sécurité & Conformité

- ✅ Pas de stockage de données personnelles
- ✅ API publique (data.economie.gouv.fr)
- ✅ Données anonymisées au niveau communes
- ✅ Cache local (session Streamlit)
- ✅ Pas de base de données externe

---

## 📦 Dépendances

```
streamlit>=1.28.0
pandas>=1.3.0
numpy>=1.21.0
requests>=2.28.0
plotly>=5.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
reportlab>=4.0.0
scipy>=1.7.0
openpyxl>=3.8.0
```

Installer tout :
```bash
pip install -r requirements.txt
```

---

## 🚀 Optimisations Futures

- [ ] Import de fichiers personnalisés
- [ ] Prédictions ML (tendances futures)
- [ ] Comparaisons inter-régionales
- [ ] Alertes automatiques (mail)
- [ ] API REST pour intégrations externes
- [ ] Dashboard temps réel

---

## 📞 Support & Contribution

### Signaler un bug
Créer une **Issue** avec :
- Département + Année
- Description du problème
- Capture d'écran si possible

### Proposer une amélioration
Créer une **Pull Request** avec :
- Description claire
- Tests unitaires
- Documentation

---

## 📄 Licence

MIT License

---

## 👥 Auteur

**Arthur Bodin pour SFP COLLECTIVITÉS**
Diagnostic financier pour collectivités territoriales

---

## 🎓 Ressources

- 📖 [Documentation API data.gouv](https://data.economie.gouv.fr/)
- 📊 [Guide Finances Locales](https://www.collectivites-territoriales.gouv.fr/)
- 🔢 [Indicateurs DGCL](https://www.dgcl.gouv.fr/)

---

## ✨ Points Forts

- 🎯 **Scoring scientifique** : Basé sur 4 KPIs reconnus
- 🌍 **Données officielles** : API gouvernementale
- 📊 **Visualisations** : 10+ graphiques interactifs
- 📄 **Export pro** : PDF/Excel de haute qualité
- 🔍 **Détail** : Historique 6 ans + radar comparatif
- ⚡ **Perfo** : Cache intelligent, requêtes optimisées
- 🎨 **UX** : Interface intuitive Streamlit
- 🔧 **Maintenable** : Code modulaire et documenté

---

**Dernière mise à jour** : 2024
**Version** : 3.0 (Scoring V3 adaptatif)