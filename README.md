# Prediction de Matchs Wimbledon

Un systeme de machine learning pour predire les resultats de matchs de tennis a Wimbledon, base sur les donnees ATP de 1985 a 2025.

## L'idee

Les classements ATP donnent une vue generale du niveau des joueurs, mais ils passent a cote de details importants : la forme recente, les preferences de surface, les face-a-face, les performances au service. L'objectif ici est de combiner tout ca dans un seul modele de prediction.

L'approche est simple :
1. Construire un systeme de rating ELO qui suit le niveau des joueurs dans le temps, avec des ratings separes par surface
2. Calculer des statistiques glissantes pour chaque joueur (derniers 3, 10, 25, 50, 100 matchs)
3. Pour chaque match, calculer la difference de stats entre les deux joueurs
4. Entrainer un modele XGBoost sur 40 ans de donnees pour apprendre quelles differences comptent le plus

## Structure du Projet

```
├── 0.CleanData.ipynb        # Charger et nettoyer les donnees ATP brutes
├── 1.CreateDataset.ipynb    # Feature engineering (ELO, stats glissantes)
├── 2.TrainModel.ipynb       # Entrainer et evaluer le modele XGBoost
├── 3.Predict.ipynb          # Predire des matchs individuels
├── 4.PreditWimbledon.ipynb  # Predictions specifiques a Wimbledon
├── train.py                 # Script d'entrainement en ligne de commande
├── utils/
│   ├── updateStats.py       # Calculs ELO et suivi des stats
│   ├── getStatsScratch.py   # Extraction des stats depuis les donnees brutes
│   └── common.py            # Fonctions utilitaires
├── code/
│   ├── DecisionTree/        # Arbre de decision from scratch
│   └── RandomForest/        # Random forest from scratch
├── data/                    # Donnees de matchs, classements, joueurs
├── models/                  # Modeles entraines
└── images/                  # Visualisations
```

## Comment ca Marche

### Systeme de Rating ELO

ELO standard avec quelques ajustements :
- Ratings separes pour chaque surface (gazon, terre battue, dur)
- K-factor dynamique : plus eleve pour les joueurs avec peu de matchs, plus bas pour les joueurs etablis
- Bonus de reprise : leger boost apres une longue pause (100+ jours) pour tenir compte de la recuperation

### Features

Pour chaque match, 81 features sont calculees comme differences entre Joueur 1 et Joueur 2 :
- Ratings ELO (global et par surface)
- Taux de victoire sur les N derniers matchs
- Stats de service : % aces, % doubles fautes, % points gagnes sur 1ere/2eme balle
- Stats de retour : % conversion de balles de break, % points gagnes en retour
- Face-a-face
- Attributs physiques : age, taille

### Modele

Classifieur XGBoost avec ces parametres (trouves via recherche aleatoire sur 2000 iterations) :
- 250 arbres, learning rate 0.04, profondeur max 5
- Echantillonnage lignes 0.9, colonnes 0.95

Le modele sort une probabilite que le Joueur 1 gagne. Pour reduire le biais lie a l'ordre des joueurs, les predictions sont faites depuis les deux perspectives puis moyennees.

## Resultats

Sur les matchs 2025 mis de cote :
- Precision du modele : ~61%
- Baseline (ELO seul) : ~56%

Pas assez pour battre les bookmakers, mais mieux que les predictions basees uniquement sur le classement.

## Installation

```bash
conda env create -f environment.yml
conda activate tennisAI
```

## Utilisation

Pour executer le pipeline complet :
```bash
# Suivre les notebooks 0 a 4 dans l'ordre
```

Pour entrainer directement en ligne de commande :
```bash
python train.py
```

Pour voir les predictions et resultats Wimbledon 2025 :
```bash
# Ouvrir wimbledonFinalResults2025.ipynb
```

## Sources de Donnees

Donnees de matchs ATP provenant du repository tennis_atp de Jeff Sackmann https://github.com/jeffsackmann

## Limites

- Modele entraine sur les matchs termines uniquement, pas de donnees en direct
- Ne prend pas en compte les blessures, la meteo ou les facteurs de motivation
- Les ratings par surface necessitent ~20 matchs pour se stabiliser
- Les joueurs de qualifications manquent souvent de donnees historiques

## Licence

MIT
