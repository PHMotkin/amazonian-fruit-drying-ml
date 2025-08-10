#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs)
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09

# Script XGBoost (Modèle 2) pour prédire la masse sèche de fruits amazoniens à partir de leurs caractéristiques initiales.
# Cette version est adaptée pour prédire le rapport masse initiale / masse sèche, utilisé ensuite pour reconstituer la masse sèche. Ce choix permet de réduire l'erreur relative de prédiction, en particulier pour les échantillons de petite masse.
# Étapes incluses :
# - Nettoyage et préparation des données
# - Division du jeu de données en données d’entrainement et données de test
# - Optimisation des hyperparamètres via Optuna
# - Validation croisée répétée avec visualisations
# - Sauvegarde du modèle, des prédictions et des métriques
# - Sauvegarde de figures pertinentes (arbres, histogrammes, scatter plot, importance des variables)
# - Export structuré dans un fichier Excel multi-feuilles

import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

# Imports pour l’entrainement du modèle et la validation croisée répétée
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
# Imports pour la mesure du R², RMSE et MAE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
# Import pour la stratification des folds du jeu de données
from sklearn.model_selection import StratifiedShuffleSplit
# Modèle utilisé pour les prédictions
from xgboost import XGBRegressor, plot_importance, plot_tree 
import joblib # Pour créer et sauvegarder le modèle
import optuna # Pour l’optimisation des hyperparamètres du modèle
import optuna.visualization.matplotlib as optuna_matplotlib

# Chargement de la base de données Excel
df = pd.read_excel("D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Data Fruits Amazoniens V8.xlsx", sheet_name="Données")

# Définition des chemins
arbre_path_base = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2"

# Renommage des colonnes
df = df.rename(columns={
    "ID Fruit": "ID_Fruit",
    "Masse tot [g]": "Masse_tot_g",
    "a [mm]": "a_mm",
    "b [mm]": "b_mm",
    "c [mm]": "c_mm",
    "Statut Masse": "Statut_Masse",
    "Fruit_Code": "Fruit_Code"
})

# Préparation des données
#Sélection des colonnes utiles 
df = df[["Fruit_Code", "ID_Fruit", "Masse_tot_g", "a_mm", "b_mm", "c_mm", "Statut_Masse"]]

#Séparer les masses initiales et sèches
df_initial = df[df["Statut_Masse"] == "INITIALE"]
df_seche = df[df["Statut_Masse"] == "SECHE"]

#Fusion sur ID_Fruit – Assure que chaque échantillon de fruit (ID_Fruit) est correctement associé à sa masse initiale et à sa masse sèche correspondante
df_merged = pd.merge(df_initial, df_seche[["ID_Fruit", "Masse_tot_g"]], on="ID_Fruit", suffixes=("_initiale", "_seche"))

# Sélection limitée uniquement aux colonnes utilisées comme variables explicatives
# Nous conserverons tous les échantillons dont les caractéristiques initiales sont complètes, même si certaines données manquent du côté des mesures après l’étuve (ex. dimensions abc du fruit sec)

colonnes_X = ["Masse_tot_g_initiale", "a_mm", "b_mm", "c_mm"]

# Supprimer uniquement les échantillons pour lesquels au moins une variable explicative de colonnes_X est manquante
df_merged = df_merged.dropna(subset=colonnes_X)
# Supprimer les échantillons pour lesquels la variable cible y (masse sèche) est manquante
df_merged = df_merged[df_merged["Masse_tot_g_seche"].notna()]

# Nouvelle cible : facteur de réduction de masse (objectif = minimiser ce rapport masse initiale / masse sèche)
df_merged["Facteur_reduction"] = df_merged["Masse_tot_g_initiale"] / df_merged["Masse_tot_g_seche"]

# Définition de X et y après sélection des variables d’entrées et de sortie, et trie des échantillons 
X = df_merged[colonnes_X]
y = df_merged["Facteur_reduction"]

strat_col = df_merged["Fruit_Code"]

# Découpage stratifié du jeu de données en k folds (1 fold pour le test, k-1 pour l'entraînement)
# Ce nombre de folds stratifiés est choisi par défaut égal à 10 si le fruit ayant le nombre d’échantillons le plus faible est supérieur ou égal à 10 afin d’utiliser 90% des échantillons pour l’entrainement. Dans notre cas, ce nombre est égal à 8 car le guarana anguleux ne possède que 8 échantillons.
sss = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=42)
for train_index, test_index in sss.split(X, strat_col):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train_factor, y_test_factor = y.iloc[train_index], y.iloc[test_index]

    # Important : on utilise X_train et y_train tels que définis avant (issus de StratifiedShuffleSplit)
    strat_col_train = strat_col.loc[X_train.index]

# Définition du modèle XGBoost et calculs des scores
# Remarque : l’entier initial 42, noté à plusieurs reprises dans « random_state », constitue la « seed » (graine). Celle-ci est utilisée pour générer des nombres pseudo-aléatoires de manière reproductible dans un algorithme informatique.
# Cela garantit que le modèle et ses performances peuvent être reproduits à l’identique à partir du même jeu de données. Néanmoins, en raison de la contrainte temporelle imposée dans Optuna (300 secondes, voir ci-dessous), de légères variations dans les hyperparamètres optimaux peuvent apparaître entre deux exécutions.

# Fonction personnalisée de stratification pour Optuna (strat_col = Fruit_Code)
def repeated_stratified_kfold_optuna(X, strat_col, n_splits=5, n_repeats=1, base_seed=42):
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + repeat)
        for train_idx, test_idx in skf.split(X, strat_col):
            yield train_idx, test_idx

#  Optuna : optimisation des hyperparamètres
def objective(trial):
    params = {
        # Nombre total d’arbres (boosting rounds) : un nombre trop faible limite la capacité
        # d’apprentissage, tandis qu’un nombre trop élevé peut accroître le surapprentissage. 
        # La recherche est limitée ici à 350 pour des raisons de temps de calcul, de limitation de risques d’overfitting, 
        # et de pertinence expérimentale sur un petit dataset.
        "n_estimators": trial.suggest_int("n_estimators", 100, 350),
        # Profondeur maximale modérée : évite les arbres trop complexes et le surapprentissage
        # (arbres peu profonds)
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        # Taux d’apprentissage (learning_rate) pour contrôler la vitesse d’apprentissage
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        # La valeur maximale de l’hyperparamètre subsample est légèrement inférieure à 1.0 pour
        # contrôler le bruit et pour introduire de la diversité entre les arbres
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # Fraction de variables utilisées pour chaque arbre : favorise la diversité des arbres
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        # Graine de reproductibilité
        "random_state": 42
    }
    model = XGBRegressor(**params)

    # Validation croisée stratifiée interne dans Optuna pour trouver les meilleurs hyperparamètres - Évaluation par RMSE moyenne sur 5 folds stratifiés 
    rmse_list = []

    for train_idx, test_idx in repeated_stratified_kfold_optuna(X_train, strat_col_train, n_splits=5, n_repeats=1):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train_factor.iloc[train_idx], y_train_factor.iloc[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)

# Optimisation des hyperparamètres avec Optuna et limite de temps
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=300) # Limite à 5 minutes (300 secondes)

# Figure : Visualisation de l'évolution de l'optimisation Optuna
ax = optuna_matplotlib.plot_optimization_history(study)
ax.set_title("Évolution de l'optimisation Optuna\n(Objective Value par essai)")
ax.figure.tight_layout()
# Sauvegarde de la figure
optuna_history_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/optuna_evolution_objective.png"
ax.figure.savefig(optuna_history_path)
plt.close(ax.figure)

# Entraînement final du modèle avec les meilleurs paramètres trouvés
best_params = study.best_params

# Sauvegarde du modèle Optuna
joblib.dump(study, f"{arbre_path_base}/optuna_study_modele2.pkl")

# Validation croisée répétée stratifiée (7 folds × 4 répétitions)
def repeated_stratified_kfold(X, strat_col, n_splits, n_repeats, base_seed=42):
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + repeat)
        for train_idx, test_idx in skf.split(X, strat_col):
            yield train_idx, test_idx

r2_scores = []
rmse_scores = []
mae_scores = []

strat_col_train = strat_col.loc[X_train.index].reset_index(drop=True)
for fold_number, (train_index, test_index) in enumerate(
        repeated_stratified_kfold(X_train, strat_col_train, n_splits=7, n_repeats=4), 1):
    
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train_factor.iloc[train_index], y_train_factor.iloc[test_index]

    model_cv = XGBRegressor(**best_params)  # Important : Le modèle est réinitialisé à chaque fold pour éviter les effets de surapprentissage cumulés

    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_factor_cv = model_cv.predict(X_test_cv)

    # Conversion du facteur prédit en masse sèche prédite pour évaluer la performance réelle
    y_pred_cv = X_test_cv["Masse_tot_g_initiale"].values / y_pred_factor_cv
    y_test_cv = X_test_cv["Masse_tot_g_initiale"].values / y_test_cv.values

    r2 = r2_score(y_test_cv, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y_test_cv, y_pred_cv))
    mae = mean_absolute_error(y_test_cv, y_pred_cv)

    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold_number} - R² : {r2:.4f}, RMSE : {rmse:.4f}, MAE : {mae:.4f}")
    fold_number += 1

# Entraînement du modèle final sur tout le jeu d'entraînement
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train_factor)

# Évaluation finale sur le set de test (données jamais vu)
# Conversion du facteur en masse sèche prédite
y_pred_factor = final_model.predict(X_test)

# Conversion en masse sèche prédite
y_pred_test = X_test["Masse_tot_g_initiale"].values / y_pred_factor
y_test = X_test["Masse_tot_g_initiale"].values / y_test_factor.values

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)

# Moyennes des scores et standard deviation
r2_mean = np.mean(r2_scores)
r2_std = np.std(r2_scores)
rmse_mean = np.mean(rmse_scores)
rmse_std = np.std(rmse_scores)
mae_mean = np.mean(mae_scores)
mae_std = np.std(mae_scores)

print(f"\nR² moyen : {r2_mean:.4f} ± {r2_std:.4f}")
print(f"RMSE moyen : {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"MAE moyen : {mae_mean:.4f} ± {mae_std:.4f}")

# BOOTSTRAP SUR LE JEU DE TEST : IC 95 % des métriques 
# Objectif : estimer l'incertitude des performances du modèle final
# sur les données de test non vues, en réalisant un bootstrap
# sur X_test et y_test (échantillonnage avec remise)

n_iterations = 1000  # Nombre d'itérations bootstrap
rng = np.random.default_rng(seed=42)  # Générateur aléatoire reproductible

bootstrap_r2 = []
bootstrap_rmse = []
bootstrap_mae = []

# Bootstrap sur le jeu de test (pour évaluer les IC95% sur masse sèche prédite)
for i in range(n_iterations):
    indices = rng.choice(len(X_test), size=len(X_test), replace=True)
    X_resample = X_test.iloc[indices]
    y_resample = pd.Series(y_test).iloc[indices]

    # Prédiction du facteur
    y_pred_factor = final_model.predict(X_resample)

    # Conversion en masse sèche prédite
    mi_resample = X_resample["Masse_tot_g_initiale"].values
    y_pred_resample = mi_resample / y_pred_factor

    # Calcul des métriques (comparaison entre masses sèches observées et prédites)
    r2_boot = r2_score(y_resample, y_pred_resample)
    rmse_boot = np.sqrt(mean_squared_error(y_resample, y_pred_resample))
    mae_boot = mean_absolute_error(y_resample, y_pred_resample)

    bootstrap_r2.append(r2_boot)
    bootstrap_rmse.append(rmse_boot)
    bootstrap_mae.append(mae_boot)

# Fonction utilitaire pour IC à 95 %
def ic_95(array):
    return np.percentile(array, 2.5), np.percentile(array, 97.5)

# Calcul des intervalles de confiance à 95 %
r2_ic = ic_95(bootstrap_r2)
rmse_ic = ic_95(bootstrap_rmse)
mae_ic = ic_95(bootstrap_mae)

df_bootstrap_metrics = pd.DataFrame({
    "R2_bootstrap": bootstrap_r2,
    "RMSE_bootstrap": bootstrap_rmse,
    "MAE_bootstrap": bootstrap_mae
})
df_bootstrap_metrics.to_excel(f"{arbre_path_base}/bootstrap_test_metrics.xlsx", index=False)

# Affichage dans la console
print(f"R² bootstrap IC 95% : {r2_ic}")
print(f"RMSE bootstrap IC 95% : {rmse_ic}")
print(f"MAE bootstrap IC 95% : {mae_ic}")

# Initialisation du tableau résumé
if 'resume_df' not in globals():
    resume_df = pd.DataFrame(columns=["Métrique", "Valeur"])

# Ajout au tableau résumé
resume_df.loc[len(resume_df)] = ["R² (Bootstrap Test)", f"{np.mean(bootstrap_r2):.4f} ± ({r2_ic[0]:.4f}, {r2_ic[1]:.4f})"]
resume_df.loc[len(resume_df)] = ["RMSE (Bootstrap Test)", f"{np.mean(bootstrap_rmse):.4f} ± ({rmse_ic[0]:.4f}, {rmse_ic[1]:.4f})"]
resume_df.loc[len(resume_df)] = ["MAE (Bootstrap Test)", f"{np.mean(bootstrap_mae):.4f} ± ({mae_ic[0]:.4f}, {mae_ic[1]:.4f})"]

# Figure : Distribution bootstrap du R²
plt.figure(figsize=(8, 5))
plt.hist(bootstrap_r2, bins=30, color='cornflowerblue', edgecolor='black')
plt.axvline(r2_ic[0], color='red', linestyle='--', label=f"IC 95% Lower ({r2_ic[0]:.3f})")
plt.axvline(r2_ic[1], color='red', linestyle='--', label=f"IC 95% Upper ({r2_ic[1]:.3f})")
plt.title("Distribution bootstrap du R² sur le jeu de test")
plt.xlabel("R²")
plt.ylabel("Fréquence")
plt.legend()
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/bootstrap_r2_distribution.png", dpi=300)
plt.close()

# Figure : Distribution bootstrap du RMSE
plt.figure(figsize=(8, 5))
plt.hist(bootstrap_rmse, bins=30, color='mediumseagreen', edgecolor='black')
plt.axvline(rmse_ic[0], color='red', linestyle='--', label=f"IC 95% Lower ({rmse_ic[0]:.3f})")
plt.axvline(rmse_ic[1], color='red', linestyle='--', label=f"IC 95% Upper ({rmse_ic[1]:.3f})")
plt.title("Distribution bootstrap du RMSE sur le jeu de test")
plt.xlabel("RMSE")
plt.ylabel("Fréquence")
plt.legend()
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/bootstrap_rmse_distribution.png", dpi=300)
plt.close()

# Validation Croisée Répétée
# Figure : Évolution des scores R² et RMSE à travers les folds
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(r2_scores) + 1), r2_scores, marker='o', label='R²', linestyle='-', linewidth=2)
plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='s', label='RMSE', linestyle='--', linewidth=2)
plt.xticks(range(1, len(r2_scores) + 1))
plt.xlabel("N° de validation croisée (Fold)")
plt.ylabel("Score")
plt.title("Évolution des scores R² et RMSE\nValidation croisée répétée (7 Folds × 4 Répétitions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
cv_evolution_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/evolution_cv_scores.png"
plt.savefig(cv_evolution_path, dpi=300)
plt.close()

# Figure : Évolution du score MAE – Séparée de l’autre figure car le MAE n’a pas les mêmes unités ni les mêmes ordres de grandeur que R² et RMSE
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mae_scores) + 1), mae_scores, marker='^', color='coral', linestyle='-', linewidth=2)
plt.xticks(range(1, len(mae_scores) + 1))
plt.xlabel("N° de validation croisée (Fold)")
plt.ylabel("MAE [g]")
plt.title("Évolution du MAE (erreur absolue moyenne)\nValidation croisée répétée (7 Folds × 4 Répétitions)")
plt.grid(True)
plt.tight_layout() # Évite tout chevauchement visuel
evolution_mae_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/evolution_cv_MAE.png"
plt.savefig(evolution_mae_path)
plt.close()

# Sauvegarde du modèle pour de futures usages
modele_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/xgboost_MS_modele_2.joblib"
joblib.dump(final_model, modele_path)

# Préparation des données à exporter dans un fichier Excel 
# Résultats des prédictions et erreurs sur les données d'entrainement (R², RMSE et MAE)
# Conversion du facteur prédit en masse sèche prédite
# Prédictions des facteurs
y_pred_factor = final_model.predict(X_train)
y_pred_test_factor = final_model.predict(X_test)

# Conversion en masse sèche prédite
y_pred = X_train["Masse_tot_g_initiale"] / y_pred_factor
y_pred_test = X_test["Masse_tot_g_initiale"] / y_pred_test_factor

y_train = X_train["Masse_tot_g_initiale"] / y_train_factor.values
y_test = X_test["Masse_tot_g_initiale"] / y_test_factor.values

# Évaluation des performances
mae = mean_absolute_error(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
r2 = r2_score(y_train, y_pred)

mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

# Préparation des DataFrames
df_train_pred = X_train.copy()
df_train_pred["ID_Fruit"] = df_merged.loc[X_train.index, "ID_Fruit"].values
df_train_pred["Fruit_Code"] = df_merged.loc[X_train.index, "Fruit_Code"].values
df_train_pred["Masse_sèche_réelle"] = y_train
df_train_pred["Masse_sèche_prédite"] = y_pred
df_train_pred["Origine"] = "Train"

df_test_pred = X_test.copy()
df_test_pred["ID_Fruit"] = df_merged.loc[X_test.index, "ID_Fruit"].values
df_test_pred["Fruit_Code"] = df_merged.loc[X_test.index, "Fruit_Code"].values
df_test_pred["Masse_sèche_réelle"] = y_test
df_test_pred["Masse_sèche_prédite"] = y_pred_test
df_test_pred["Origine"] = "Test"

# Fusion et calculs d'erreurs
df_global_pred = pd.concat([df_train_pred, df_test_pred], axis=0).reset_index(drop=True)
df_global_pred["Erreur_absolue"] = np.abs(df_global_pred["Masse_sèche_prédite"] - df_global_pred["Masse_sèche_réelle"])
df_global_pred["Erreur_relative_%"] = 100 * df_global_pred["Erreur_absolue"] / df_global_pred["Masse_sèche_réelle"]
df_global_pred["Précision_%"] = 100 - df_global_pred["Erreur_relative_%"]

# BOOTSTRAP GLOBAL SUR TOUTES LES PRÉDICTIONS
bootstrap_preds_all = np.zeros((len(X), n_iterations))

for i in range(n_iterations):
    indices = rng.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]

    model_boot = XGBRegressor(**best_params)
    model_boot.fit(X_boot, y_boot)

    preds_factor = model_boot.predict(X)

    # Correction ici : utiliser les bonnes mi
    mi_for_preds = X["Masse_tot_g_initiale"].values
    preds = mi_for_preds / preds_factor

    bootstrap_preds_all[:, i] = preds

lower_bounds = np.percentile(bootstrap_preds_all, 2.5, axis=1)
upper_bounds = np.percentile(bootstrap_preds_all, 97.5, axis=1)
mean_preds = np.mean(bootstrap_preds_all, axis=1)

# Création d'un DataFrame intermédiaire avec l'alignement correct sur X
df_boot_results = df_merged.loc[X.index, ["ID_Fruit", "Fruit_Code"]].copy()
df_boot_results["Masse_sèche_prédite_boot"] = mean_preds
df_boot_results["IC_inf"] = lower_bounds
df_boot_results["IC_sup"] = upper_bounds

# Fusion avec df_global_pred via les identifiants uniques
df_global_pred = df_global_pred.merge(df_boot_results, on=["ID_Fruit", "Fruit_Code"], how="left")

# Figure : Résumé des performances bootstrap (IC 95 %) sous forme de barres d’erreur
fig, ax = plt.subplots(figsize=(8, 5))

metrics = ["R²", "RMSE", "MAE"]
means = [np.mean(bootstrap_r2), np.mean(bootstrap_rmse), np.mean(bootstrap_mae)]
lower_bounds = [r2_ic[0], rmse_ic[0], mae_ic[0]]
upper_bounds = [r2_ic[1], rmse_ic[1], mae_ic[1]]
errors = [[mean - low, up - mean] for mean, low, up in zip(means, lower_bounds, upper_bounds)]

# Convertir pour plt.errorbar : transpose les erreurs
errors = np.array(errors).T

ax.errorbar(metrics, means, yerr=errors, fmt='o', capsize=8, capthick=2, markersize=6, color='navy')
ax.set_title("Résumé des performances sur le jeu de test\n(IC 95 % via bootstrap)")
ax.set_ylabel("Score")
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/bootstrap_test_performance_summary.png", dpi=300)
plt.close()

# Réorganisation
colonnes_finales = ["Fruit_Code", "ID_Fruit", "Origine",
    "Masse_tot_g_initiale", "a_mm", "b_mm", "c_mm",
    "Masse_sèche_réelle", "Masse_sèche_prédite",
    "Masse_sèche_prédite_boot", "IC_inf", "IC_sup",
    "Erreur_absolue", "Erreur_relative_%", "Précision_%"
]
df_global_pred = df_global_pred[colonnes_finales].sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)

# Résumé des scores
resume_df = pd.DataFrame({
    "Indicateur": [
        "R² (Entrainement)", "RMSE (Entrainement)", "MAE (Entrainement)",
        "R² (Test)", "RMSE (Test)", "MAE (Test)",
        "R² CV (moy ± std)", "RMSE CV (moy ± std)", "MAE CV (moy ± std)",
        "R² (Bootstrap Test)", "RMSE (Bootstrap Test)", "MAE (Bootstrap Test)"
    ],
    "Valeur": [
        round(r2, 4), round(rmse, 4), round(mae, 4),
        round(r2_test, 4), round(rmse_test, 4), round(mae_test, 4),
        f"{r2_mean:.4f} ± {r2_std:.4f}",
        f"{rmse_mean:.4f} ± {rmse_std:.4f}",
        f"{mae_mean:.4f} ± {mae_std:.4f}",
        f"{np.mean(bootstrap_r2):.4f} ± ({r2_ic[0]:.4f}, {r2_ic[1]:.4f})",
        f"{np.mean(bootstrap_rmse):.4f} ± ({rmse_ic[0]:.4f}, {rmse_ic[1]:.4f})",
        f"{np.mean(bootstrap_mae):.4f} ± ({mae_ic[0]:.4f}, {mae_ic[1]:.4f})"
    ]
})

# Préparation des résultats de chaque fold dans le fichier Excel
cv_results = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(len(r2_scores))],
    "R2": r2_scores,
    "RMSE": rmse_scores,
    "MAE": mae_scores
})
cv_results.loc[len(cv_results)] = ["R² moyen", r2_mean, np.nan, np.nan]
cv_results.loc[len(cv_results)] = ["R² écart-type", r2_std, np.nan, np.nan]
cv_results.loc[len(cv_results)] = ["RMSE moyen", np.nan, rmse_mean, np.nan]
cv_results.loc[len(cv_results)] = ["RMSE écart-type", np.nan, rmse_std, np.nan]
cv_results.loc[len(cv_results)] = ["MAE moyen", np.nan, np.nan, mae_mean]
cv_results.loc[len(cv_results)] = ["MAE écart-type", np.nan, np.nan, mae_std]

# Résultats uniquement pour le set d'entraînement
df_train_export = df_global_pred[df_global_pred["Origine"] == "Train"].copy()
df_train_export = df_train_export.sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)

# Résultats uniquement pour le set de test
df_test_export = df_global_pred[df_global_pred["Origine"] == "Test"].copy()
df_test_export = df_test_export.sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)

# Création du DataFrame des hyperparamètres optimaux trouvés par Optuna
params_df = pd.DataFrame(list(best_params.items()), columns=["Paramètre", "Valeur"])

# Figure : Visualisation de l’importance des variables
plt.figure(figsize=(8, 6))
plot_importance(final_model)
plt.title("Importance des variables (XGBoost)")
plt.tight_layout()
importance_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/importance_variables.png"
plt.savefig(importance_path, dpi=300)
plt.close()

# Figure : Visualisation de certains arbres de décisions du modèle 
n_trees = best_params['n_estimators']
tree_indices = list(range(3)) + [n_trees // 2, n_trees - 1]

for num in tree_indices:
    plt.figure(figsize=(20, 10))
    plot_tree(final_model, num_trees=num)
    plt.title(f"Arbre de Décision n°{num + 1}")
    plt.tight_layout()
    plt.savefig(f"{arbre_path_base}/arbre_decision_{num + 1}.png", dpi=300)
    plt.close()

# Figure : Histogrammes des erreurs
plt.figure(figsize=(8, 5))
plt.hist(df_global_pred["Erreur_absolue"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution des erreurs absolues")
plt.xlabel("Erreur absolue [g]")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/hist_erreur_absolue.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df_global_pred["Précision_%"], bins=20, color="lightgreen", edgecolor="black")
plt.title("Distribution des précisions (%)")
plt.xlabel("Précision [%]")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/hist_precision.png", dpi=300)
plt.close()

# Figure : Scatter plot – masse sèche réelle vs masse sèche prédite
plt.figure(figsize=(6, 6))
for origine, couleur in zip(["Train", "Test"], ["dodgerblue", "darkorange"]):
    subset = df_global_pred[df_global_pred["Origine"] == origine]
    plt.scatter(
        subset["Masse_sèche_réelle"],
        subset["Masse_sèche_prédite"],
        alpha=0.7, label=origine, color=couleur
    )
# Ajout de la diagonale
min_val = df_global_pred["Masse_sèche_réelle"].min()
max_val = df_global_pred["Masse_sèche_réelle"].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

plt.xlabel("Masse sèche réelle [g]")
plt.ylabel("Masse sèche prédite [g]")
plt.title("Scatter plot : MS réelle vs MS prédite (Train/Test)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/scatter_reelle_vs_predite.png", dpi=300)
plt.close()

# Figure : Scatter plot - réel vs prédit avec IC 95 % sur les prédictions
plt.figure(figsize=(6, 6))
for origine, couleur in zip(["Train", "Test"], ["dodgerblue", "darkorange"]):
    subset = df_global_pred[df_global_pred["Origine"] == origine]
    plt.errorbar(
        subset["Masse_sèche_réelle"],
        subset["Masse_sèche_prédite_boot"],
        yerr=[subset["Masse_sèche_prédite_boot"] - subset["IC_inf"],
              subset["IC_sup"] - subset["Masse_sèche_prédite_boot"]],
        fmt='o',
        color=couleur,
        alpha=0.7,
        label=f"{origine} (±IC95%)",
        ecolor='black',
        capsize=4,
        capthick=1.2,
        elinewidth=1
    )
# Ajout de la diagonale
min_val = df_global_pred["Masse_sèche_réelle"].min()
max_val = df_global_pred["Masse_sèche_réelle"].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

plt.xlabel("Masse sèche réelle [g]")
plt.ylabel("Masse sèche prédite [g]")
plt.title("Prédictions avec IC 95 %")
plt.legend()
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/scatter_reelle_vs_predite_IC95.png", dpi=300)
plt.close()

# Figure : Évolution du gain cumulé par arbre (XGBoost) – Cela permet d’évaluer si le nombre d’arbres (n_estimator) utilisé pour optimiser les hyperparamètres de XGBoost avec Optuna est suffisant. 
booster = final_model.get_booster()
df_trees = booster.trees_to_dataframe()

plt.figure(figsize=(10, 6))
gain_by_tree = df_trees.groupby('Tree')['Gain'].sum()
gain_by_tree_cumsum = np.cumsum(gain_by_tree)

plt.plot(gain_by_tree.index, gain_by_tree_cumsum, marker='o', linestyle='-')
plt.xlabel('Numéro d\'arbre')
plt.ylabel('Gain cumulé')
plt.title('Évolution du gain cumulé au fil des arbres\n(Modèle XGBoost final)')
plt.grid(True)
plt.tight_layout()

# Sauvegarde de la figure
gain_cumule_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/gain_cumule_par_arbre.png"
plt.savefig(gain_cumule_path)
plt.close()

# Export du fichier Excel avec les différentes feuilles contenant les données
export_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost MS V2/Modèle 2/Resultats_XGBoost_MS_Modele2_CV.xlsx"
with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
    df_global_pred.to_excel(writer, sheet_name="Prédictions Globales", index=False)
    df_train_export.to_excel(writer, sheet_name="Prédictions Train-Test", index=False)
    df_test_export.to_excel(writer, sheet_name="Prédictions Test Final", index=False)
    cv_results.to_excel(writer, sheet_name="Résultats_CV", index=False)
    resume_df.to_excel(writer, sheet_name="Résumé_Scores", index=False)
    params_df.to_excel(writer, sheet_name="Paramètres_Optuna", index=False)
    df_bootstrap_metrics.to_excel(writer, sheet_name="Bootstrap_Metrics_Test", index=False)

# Print indiquant la fin des opérations
print(f"Modèle et résultats sauvegardés dans : {export_path}")

