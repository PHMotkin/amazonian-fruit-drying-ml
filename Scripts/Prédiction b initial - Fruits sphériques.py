#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs)
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09

# Script de modélisation XGBoost pour prédire la dimension b_mm initiale de fruits sphériques à partir de la masse initiale (Masse_tot_g) et du type de fruit (Fruit_Code). Cette fois en excluant le cacao.
# Étapes incluses :
# - Nettoyage et préparation des données
# - Optimisation des hyperparamètres via Optuna
# - Validation croisée répétée (7×4) avec visualisations
# - Sauvegarde du modèle, des prédictions et des métriques
# - Sauvegarde de figures pertinentes (arbres, histogrammes, scatter plot, importance des variables, Optuna, gain cumulé)
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


# Chemins
data_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Data Fruits Amazoniens V8.xlsx"
export_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm"
arbre_path_base = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm"
os.makedirs(export_path, exist_ok=True)

# Chargement des données
df = pd.read_excel(data_path, sheet_name="Données")
df = df.rename(columns={
    "ID Fruit": "ID_Fruit",
    "Masse tot [g]": "Masse_tot_g",
    "b [mm]": "b_mm",
    "Fruit_Code": "Fruit_Code",
    "Statut Masse": "Statut_Masse"
})

# Préparation des données
df = df[df["Statut_Masse"] == "INITIALE"]
df = df[df["Fruit_Code"] != 4]  # Exclure le cacao car ce fruit a une forme ellipsoïdale, dont les dimensions seront mal prédites par ce modèle (contrairement aux autres fruits qui ont forme géométrique proche d'une sphère) 
df = df[["Fruit_Code", "ID_Fruit", "Masse_tot_g", "b_mm"]].dropna()

# Définition de X et y après sélection des variables d’entrées et de sortie, et trie des échantillons 
X = df[["Fruit_Code", "Masse_tot_g"]]
y = df["b_mm"]

strat_col = df["Fruit_Code"]


# Découpage stratifié du jeu de données en k folds (1 fold pour le test, k-1 pour l'entraînement)
# Ce nombre de folds stratifiés est choisi par défaut égal à 10 si le fruit ayant le nombre d’échantillons le plus faible est supérieur ou égal à 10 afin d’utiliser 90% des échantillons pour l’entrainement. Dans notre cas, ce nombre est égal à 8 car le guarana anguleux ne possède que 8 échantillons.
sss = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=42)
for train_index, test_index in sss.split(X, strat_col):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

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

# Fonction objective pour Optuna : minimisation de l'erreur relative moyenne
def objective(trial):
    # Définition de l’espace de recherche des hyperparamètres
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "random_state": 42
    }

    model = XGBRegressor(**params)
    relative_errors = []  # Initialisation du vecteur des erreurs relatives

    # Validation croisée interne avec stratification
    for train_idx, test_idx in repeated_stratified_kfold_optuna(X_train, strat_col_train, n_splits=5, n_repeats=1):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # Vérification de la cohérence des longueurs
        if len(y_pred) != len(y_te):
            return np.inf  # Annulation du trial en cas de décalage

        # Calcul de l’erreur relative sécurisée
        error_relative = np.abs(y_te - y_pred) / np.maximum(np.abs(y_te), 1e-8)
        relative_errors.extend(error_relative)

    return np.mean(relative_errors)

# Optimisation des hyperparamètres avec Optuna et limite de temps
study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=300)

# Figure : Visualisation de l'évolution de l'optimisation Optuna
ax = optuna_matplotlib.plot_optimization_history(study)
ax.set_title("Évolution de l'optimisation Optuna\n(Objective Value par essai)")
ax.figure.tight_layout()
# Sauvegarde de la figure
optuna_history_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm/optuna_evolution_objective.png"
ax.figure.savefig(optuna_history_path)
plt.close(ax.figure)

# Entraînement final du modèle avec les meilleurs paramètres trouvés
best_params = study.best_params
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)
# Sauvegarde du modèle 
joblib.dump(final_model, f"{export_path}/xgboost_modele_final_b.joblib")

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
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

    model_cv = XGBRegressor(**best_params)  # Important : Le modèle est réinitialisé à chaque fold pour éviter les effets de surapprentissage cumulés

    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_cv.predict(X_test_cv)

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
final_model.fit(X_train, y_train)

# Évaluation finale sur le set de test (données jamais vu)
y_pred_test = final_model.predict(X_test)
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

for i in range(n_iterations):
    # Tirage aléatoire avec remise parmi les indices du jeu de test
    indices = rng.choice(len(X_test), size=len(X_test), replace=True)
    X_resample = X_test.iloc[indices]
    y_resample = y_test.iloc[indices]

    # Prédictions sur l'échantillon bootstrapé
    y_pred_resample = final_model.predict(X_resample)

    # Calcul des métriques R² et RMSE
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
cv_evolution_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm/evolution_cv_scores.png"
plt.savefig(cv_evolution_path, dpi=300)
plt.close()

# Figure : Évolution du score MAE – Séparée de l’autre figure car le MAE n’a pas les mêmes unités ni les mêmes ordres de grandeur que R² et RMSE
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mae_scores) + 1), mae_scores, marker='^', color='coral', linestyle='-', linewidth=2)
plt.xticks(range(1, len(mae_scores) + 1))
plt.xlabel("N° de validation croisée (Fold)")
plt.ylabel("MAE [mm]")
plt.title("Évolution du MAE (erreur absolue moyenne)\nValidation croisée répétée (7 Folds × 4 Répétitions)")
plt.grid(True)
plt.tight_layout() # Évite tout chevauchement visuel
evolution_mae_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm/evolution_cv_MAE.png"
plt.savefig(evolution_mae_path)
plt.close()


# Préparation des données à exporter dans un fichier Excel 
# Résultats des prédictions et erreurs sur les données d'entrainement (R², RMSE et MAE)
y_pred = final_model.predict(X_train)
mae = mean_absolute_error(y_train, y_pred)
rmse = np.sqrt(mean_squared_error(y_train, y_pred))
r2 = r2_score(y_train, y_pred)

# Résultats sur le jeu de test (jamais vus à l'entraînement)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

# Préparation du DataFrame complet (Train + Test)
df_merged = df.copy()
df_train_pred = X_train.copy()
df_train_pred["ID_Fruit"] = df_merged.loc[X_train.index, "ID_Fruit"].values
df_train_pred["Fruit_Code"] = df_merged.loc[X_train.index, "Fruit_Code"].values
df_train_pred["b_mm_réel"] = y_train.values
df_train_pred["b_mm_prédit"] = y_pred
df_train_pred["Origine"] = "Train"

for col in ["Masse_tot_g", "b_mm"]:
    df_train_pred[col] = df_merged.loc[X_train.index, col].values

df_test_pred = X_test.copy()
df_test_pred["ID_Fruit"] = df_merged.loc[X_test.index, "ID_Fruit"].values
df_test_pred["Fruit_Code"] = df_merged.loc[X_test.index, "Fruit_Code"].values
df_test_pred["b_mm_réel"] = y_test.values
df_test_pred["b_mm_prédit"] = y_pred_test
df_test_pred["Origine"] = "Test"

for col in ["Masse_tot_g", "b_mm"]:
    df_test_pred[col] = df_merged.loc[X_test.index, col].values

# Fusion train + test
df_global_pred = pd.concat([df_train_pred, df_test_pred], axis=0).reset_index(drop=True)
df_global_pred["Erreur_absolue"] = np.abs(df_global_pred["b_mm_prédit"] - df_global_pred["b_mm_réel"])
df_global_pred["Erreur_relative_%"] = 100 * df_global_pred["Erreur_absolue"] / df_global_pred["b_mm_réel"]
df_global_pred["Précision_%"] = 100 - df_global_pred["Erreur_relative_%"]

# BOOTSTRAP GLOBAL SUR TOUTES LES PRÉDICTIONS
bootstrap_preds_all = np.zeros((len(X), n_iterations))

for i in range(n_iterations):
    indices = rng.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]

    model_boot = XGBRegressor(**best_params)
    model_boot.fit(X_boot, y_boot)

    preds = model_boot.predict(X)
    bootstrap_preds_all[:, i] = preds

lower_bounds = np.percentile(bootstrap_preds_all, 2.5, axis=1)
upper_bounds = np.percentile(bootstrap_preds_all, 97.5, axis=1)
mean_preds = np.mean(bootstrap_preds_all, axis=1)

# Création d'un DataFrame intermédiaire avec l'alignement correct sur X
df_boot_results = df_merged.loc[X.index, ["ID_Fruit", "Fruit_Code"]].copy()
df_boot_results["b_mm_prédit_boot"] = mean_preds
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
df_global_pred = df_global_pred.rename(columns={"Masse_tot_g": "Masse_tot_g_initiale"})

colonnes_finales = ["Fruit_Code", "ID_Fruit", "Origine",
    "Masse_tot_g_initiale", "b_mm_réel", "b_mm_prédit", "b_mm_prédit_boot",
    "IC_inf", "IC_sup", "Erreur_absolue", "Erreur_relative_%", "Précision_%"
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
importance_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm/importance_variables.png"
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
plt.xlabel("Erreur absolue [mm]")
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

# Figure : Scatter plot - réel vs prédit
plt.figure(figsize=(6, 6))
for origine, couleur in zip(["Train", "Test"], ["dodgerblue", "darkorange"]):
    subset = df_global_pred[df_global_pred["Origine"] == origine]
    plt.scatter(
        subset["b_mm_réel"],
        subset["b_mm_prédit"],
        alpha=0.7, label=origine, color=couleur
    )
# Ajout de la diagonale
min_val = df_global_pred["b_mm_réel"].min()
max_val = df_global_pred["b_mm_réel"].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

plt.xlabel("b réel [mm]")
plt.ylabel("b prédit [mm]")
plt.title("Scatter plot : b réel vs b prédit (Train/Test)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{arbre_path_base}/scatter_reelle_vs_predite.png", dpi=300)
plt.close()

# Figure : Scatter plot - réel vs prédit avec IC 95 % sur les prédictions
plt.figure(figsize=(6, 6))
for origine, couleur in zip(["Train", "Test"], ["dodgerblue", "darkorange"]):
    subset = df_global_pred[df_global_pred["Origine"] == origine]
    plt.errorbar(
        subset["b_mm_réel"],
        subset["b_mm_prédit_boot"],
        yerr=[
            subset["b_mm_prédit_boot"] - subset["IC_inf"],
            subset["IC_sup"] - subset["b_mm_prédit_boot"]
        ],
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
min_val = df_global_pred["b_mm_réel"].min()
max_val = df_global_pred["b_mm_réel"].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")

plt.xlabel("b réel [mm]")
plt.ylabel("b prédit [mm]")
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
gain_cumule_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm/gain_cumule_par_arbre.png"
plt.savefig(gain_cumule_path)
plt.close()

# Export Excel
with pd.ExcelWriter(f"{export_path}/Resultats_XGBoost_b_mm.xlsx", engine="openpyxl") as writer:
    df_global_pred.to_excel(writer, sheet_name="Prédictions Globales", index=False)
    df_train_export.to_excel(writer, sheet_name="Prédictions Train-Test", index=False)
    df_test_export.to_excel(writer, sheet_name="Prédictions Test Final", index=False)
    cv_results.to_excel(writer, sheet_name="Résultats_CV", index=False)
    resume_df.to_excel(writer, sheet_name="Résumé_Scores", index=False)
    params_df.to_excel(writer, sheet_name="Paramètres_Optuna", index=False)
    df_bootstrap_metrics.to_excel(writer, sheet_name="Bootstrap_Metrics_Test", index=False)

print(f"Modèle b_mm et résultats sauvegardés dans : {export_path}")

