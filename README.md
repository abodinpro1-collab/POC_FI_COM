
# Alerte finances locales — mini-app (POC)

Cette mini-application **Streamlit** calcule des **indicateurs financiers** et un **score d'alerte** (Vert/Orange/Rouge) pour des collectivités, à partir d'un CSV de données publiques.

## Installation rapide

1) Créez un environnement (recommandé) et installez les dépendances :  
```bash
pip install -r requirements.txt
```

2) Lancez l'app :  
```bash
streamlit run app.py
```

3) Chargez votre CSV ou utilisez le **jeu d'exemple** (case à cocher).

## Format de données minimal requis

Colonnes (unités annuelles, en euros) :

- `insee` : code de la collectivité  
- `nom` : nom (optionnel)  
- `annee` : année (YYYY)  
- `pop` : population  
- `rrf` : recettes réelles de fonctionnement  
- `drf` : dépenses réelles de fonctionnement  
- `encours` : encours de dette (31/12)  
- `annuite` : annuité de dette (intérêts + capital)  
- `ep_brute` : épargne brute (= rrf - drf)

## Indicateurs calculés

- **Taux d'épargne brute** = ép. brute / RRF  
- **Capacité de désendettement** = encours / ép. brute (années)  
- **Annuité / RRF** (%)  
- **Encours / habitant** (€/hab)

## Scoring simplifié (ajustable dans la barre latérale)

- Seuils par défaut :
  - CD : vert ≤ 8 ans ; orange ≤ 12 ans ; sinon rouge  
  - TEB : vert ≥ 8 % ; orange ≥ 5 % ; sinon rouge  
  - Annuité/RRF : vert ≤ 12 % ; orange ≤ 18 % ; sinon rouge  
  - Encours/hab : classé par quantiles (Q60/Q80) pour l'année

- Poids par défaut (somme 100) : CD 30, TEB 25, Annuité 20, Encours/hab 25

- **Score global** ∈ [0,100], **Alerte** : Vert (≥70) ; Orange (50–69) ; Rouge (<50)

## Données d'exemple

`sample_data.csv` contient une dizaine de collectivités fictives sur 2 années, pour tester l'app sans données officielles.

## Avertissement

POC pédagogique — à adapter pour la production (strates démographiques, indicateurs élargis, seuils documentés, import automatisé des jeux officiels DGFiP/INSEE).
