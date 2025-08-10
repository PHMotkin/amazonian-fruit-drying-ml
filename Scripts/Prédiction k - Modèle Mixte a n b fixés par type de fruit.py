#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs) 
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09

# Script pour prédire le paramètre k (constant cinétique de séchage) du modèle mixte (avec a, n et b fixés par type de fruit).

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor, plot_importance, plot_tree
import joblib
import optuna
import optuna.visualization.matplotlib as optuna_matplotlib

# Chemins des fichiers
data_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Midilli/Dimensions_Prédites_Modèle_3.xlsx"
midilli_params_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Midilli/parametres_mixte_anb_fixés.xlsx"
output_base_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Midilli/Paramètre k – Mixte_anb_fixés - Log"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_base_path, exist_ok=True)

# Charger la base de données principale
df = pd.read_excel(data_path, sheet_name="Données")

# Renommage des colonnes
df = df.rename(columns={
    "ID Fruit": "ID_Fruit",
    "Masse tot [g]": "Masse_tot_g",
    "Statut Masse": "Statut_Masse",
})

# Sélection des colonnes utiles
df = df[["Fruit_Code", "ID_Fruit", "Masse_tot_g", "Statut_Masse"]]

# Préparer les masses initiales
df_initiale = df[df["Statut_Masse"] == "INITIALE"][["ID_Fruit", "Masse_tot_g"]]
df_initiale = df_initiale.rename(columns={"Masse_tot_g": "Masse_tot_g_initiale"})

# Charger les dimensions initiales
df_dimensions = pd.read_excel(data_path, sheet_name="Données")
df_dimensions = df_dimensions.rename(columns={
    "ID Fruit": "ID_Fruit",
    "a [mm]": "a_mm",
    "b [mm]": "b_mm",
    "c [mm]": "c_mm",
    "Source a": "a_source",
    "Source b": "b_source",
    "Source c": "c_source"
})

# Charger les paramètres Midilli (a, b, n, k)
df_midilli = pd.read_excel(midilli_params_path)

# Fusion des données
df_merged = pd.merge(
    df_initiale[["ID_Fruit", "Masse_tot_g_initiale"]],
    df_dimensions[["ID_Fruit", "Fruit_Code", "a_mm", "b_mm", "c_mm", "a_source", "b_source", "c_source"]],
    on="ID_Fruit",
    how="inner"
)
df_merged = pd.merge(
    df_merged,
    df_midilli[["ID_Fruit", "midilli_a", "midilli_b", "midilli_n", "k"]],
    on="ID_Fruit",
    how="inner"
)

# Sélection des colonnes explicatives (sans midilli_a et midilli_b)
colonnes_X = ["Masse_tot_g_initiale", "a_mm", "b_mm", "c_mm", "midilli_a", "midilli_n", "midilli_b"]

# Supprimer les échantillons avec des valeurs manquantes dans les colonnes explicatives ou k
df_merged = df_merged.dropna(subset=colonnes_X + ["k"])

# Définition de X et y avec transformation logarithmique
X = df_merged[colonnes_X]
y = np.log(df_merged["k"])  # Transformation logarithmique de k
strat_col = df_merged["Fruit_Code"]

# Découpage stratifié (1 fold pour test, 4 folds pour entraînement)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, strat_col):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    strat_col_train = strat_col.iloc[train_index]

# Fonction personnalisée pour validation croisée stratifiée dans Optuna
def repeated_stratified_kfold_optuna(X, strat_col, n_splits=4, n_repeats=2, base_seed=42):
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + repeat)
        for train_idx, test_idx in skf.split(X, strat_col):
            yield train_idx, test_idx

# Optimisation des hyperparamètres avec Optuna
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "random_state": 42
    }
    model = XGBRegressor(**params)
    rmse_list = []

    for train_idx, test_idx in repeated_stratified_kfold_optuna(X_train, strat_col_train, n_splits=4, n_repeats=2):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = np.exp(model.predict(X_te))  # Revenir à l'échelle originale
        y_te_orig = np.exp(y_te)  # Revenir à l'échelle originale
        rmse = np.sqrt(mean_squared_error(y_te_orig, y_pred))
        rmse_list.append(rmse)

    return np.mean(rmse_list)

study = optuna.create_study(direction="minimize")
study.optimize(objective, timeout=300)  # Temps de recherche de 300 secondes = 5 minutes

# Visualisation de l'évolution de l'optimisation Optuna
ax = optuna_matplotlib.plot_optimization_history(study)
ax.set_title("Évolution de l'optimisation Optuna\n(Objective Value par essai)")
ax.figure.tight_layout()
optuna_history_path = f"{output_base_path}/optuna_evolution_objective.png"
ax.figure.savefig(optuna_history_path)
plt.close(ax.figure)

# Sauvegarde de l'étude Optuna
joblib.dump(study, f"{output_base_path}/optuna_study.pkl")

# Entraînement final avec les meilleurs paramètres
best_params = study.best_params
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Validation croisée répétée (4 folds × 4 répétitions)
r2_scores = []
rmse_scores = []
mae_scores = []

for fold_number, (train_index, test_index) in enumerate(
        repeated_stratified_kfold_optuna(X_train, strat_col_train, n_splits=4, n_repeats=4), 1):
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
    model_cv = XGBRegressor(**best_params)
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_cv = np.exp(model_cv.predict(X_test_cv))  # Revenir à l'échelle originale
    y_test_cv_orig = np.exp(y_test_cv)  # Revenir à l'échelle originale
    r2 = r2_score(y_test_cv_orig, y_pred_cv)
    rmse = np.sqrt(mean_squared_error(y_test_cv_orig, y_pred_cv))
    mae = mean_absolute_error(y_test_cv_orig, y_pred_cv)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    print(f"Fold {fold_number} - R²: {r2:.4f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

# Évaluation finale sur le set de test
y_pred_test = np.exp(final_model.predict(X_test))  # Revenir à l'échelle originale
y_test_orig = np.exp(y_test)  # Revenir à l'échelle originale
r2_test = r2_score(y_test_orig, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))
mae_test = mean_absolute_error(y_test_orig, y_pred_test)

# Bootstrap pour IC 95% sur le jeu de test
n_iterations = 1000
rng = np.random.default_rng(seed=42)
bootstrap_r2 = []
bootstrap_rmse = []
bootstrap_mae = []

for i in range(n_iterations):
    indices = rng.choice(len(X_test), size=len(X_test), replace=True)
    X_resample = X_test.iloc[indices]
    y_resample = y_test.iloc[indices]
    if np.std(y_resample) > 1e-6:
        y_pred_resample = np.exp(final_model.predict(X_resample))  # Revenir à l'échelle originale
        y_resample_orig = np.exp(y_resample)  # Revenir à l'échelle originale
        r2_boot = r2_score(y_resample_orig, y_pred_resample)
        rmse_boot = np.sqrt(mean_squared_error(y_resample_orig, y_pred_resample))
        mae_boot = mean_absolute_error(y_resample_orig, y_pred_resample)
        bootstrap_r2.append(r2_boot)
        bootstrap_rmse.append(rmse_boot)
        bootstrap_mae.append(mae_boot)

# Calcul des intervalles de confiance
def ic_95(array):
    return np.percentile(array, 2.5), np.percentile(array, 97.5)

r2_ic = ic_95(bootstrap_r2)
rmse_ic = ic_95(bootstrap_rmse)
mae_ic = ic_95(bootstrap_mae)

df_bootstrap_metrics = pd.DataFrame({
    "R2_bootstrap": bootstrap_r2,
    "RMSE_bootstrap": bootstrap_rmse,
    "MAE_bootstrap": bootstrap_mae
})
df_bootstrap_metrics.to_excel(f"{output_base_path}/bootstrap_test_metrics.xlsx", index=False)

# Résumé des scores
resume_df = pd.DataFrame({
    "Indicateur": [
        "R² (Entrainement)", "RMSE (Entrainement)", "MAE (Entrainement)",
        "R² (Test)", "RMSE (Test)", "MAE (Test)",
        "R² CV (moy ± std)", "RMSE CV (moy ± std)", "MAE CV (moy ± std)",
        "R² (Bootstrap Test)", "RMSE (Bootstrap Test)", "MAE (Bootstrap Test)"
    ],
    "Valeur": [
        r2_score(np.exp(y_train), np.exp(final_model.predict(X_train))),
        np.sqrt(mean_squared_error(np.exp(y_train), np.exp(final_model.predict(X_train)))),
        mean_absolute_error(np.exp(y_train), np.exp(final_model.predict(X_train))),
        r2_test,
        rmse_test,
        mae_test,
        f"{np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}",
        f"{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}",
        f"{np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}",
        f"{np.mean(bootstrap_r2):.4f} ± ({r2_ic[0]:.4f}, {r2_ic[1]:.4f})",
        f"{np.mean(bootstrap_rmse):.4f} ± ({rmse_ic[0]:.4f}, {rmse_ic[1]:.4f})",
        f"{np.mean(bootstrap_mae):.4f} ± ({mae_ic[0]:.4f}, {mae_ic[1]:.4f})"
    ]
})

# Figures
# Distribution bootstrap du R²
plt.figure(figsize=(8, 5))
plt.hist(bootstrap_r2, bins=30, color='cornflowerblue', edgecolor='black')
plt.axvline(r2_ic[0], color='red', linestyle='--', label=f"IC 95% Lower ({r2_ic[0]:.3f})")
plt.axvline(r2_ic[1], color='red', linestyle='--', label=f"IC 95% Upper ({r2_ic[1]:.3f})")
plt.title("Distribution bootstrap du R² sur le jeu de test")
plt.xlabel("R²")
plt.ylabel("Fréquence")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_base_path}/bootstrap_r2_distribution.png", dpi=300)
plt.close()

# Distribution bootstrap du RMSE
plt.figure(figsize=(8, 5))
plt.hist(bootstrap_rmse, bins=30, color='mediumseagreen', edgecolor='black')
plt.axvline(rmse_ic[0], color='red', linestyle='--', label=f"IC 95% Lower ({rmse_ic[0]:.6f})")
plt.axvline(rmse_ic[1], color='red', linestyle='--', label=f"IC 95% Upper ({rmse_ic[1]:.6f})")
plt.title("Distribution bootstrap du RMSE sur le jeu de test")
plt.xlabel("RMSE")
plt.ylabel("Fréquence")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_base_path}/bootstrap_rmse_distribution.png", dpi=300)
plt.close()

# Évolution des scores R² et RMSE
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(r2_scores) + 1), r2_scores, marker='o', label='R²', linestyle='-', linewidth=2)
plt.plot(range(1, len(rmse_scores) + 1), rmse_scores, marker='s', label='RMSE', linestyle='--', linewidth=2)
plt.xticks(range(1, len(r2_scores) + 1))
plt.xlabel("N° de validation croisée (Fold)")
plt.ylabel("Score")
plt.title("Évolution des scores R² et RMSE\nValidation croisée répétée (4 Folds × 4 Répétitions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_base_path}/evolution_cv_scores.png", dpi=300)
plt.close()

# Évolution du MAE
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(mae_scores) + 1), mae_scores, marker='^', color='coral', linestyle='-', linewidth=2)
plt.xticks(range(1, len(mae_scores) + 1))
plt.xlabel("N° de validation croisée (Fold)")
plt.ylabel("MAE")
plt.title("Évolution du MAE\nValidation croisée répétée (4 Folds × 4 Répétitions)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_base_path}/evolution_cv_MAE.png", dpi=300)
plt.close()

# Importance des variables
plt.figure(figsize=(8, 6))
plot_importance(final_model)
plt.title("Importance des variables (XGBoost)")
plt.tight_layout()
plt.savefig(f"{output_base_path}/importance_variables.png", dpi=300)
plt.close()

# Arbres de décision
n_trees = best_params['n_estimators']
tree_indices = list(range(3)) + [n_trees // 2, n_trees - 1]
for num in tree_indices:
    plt.figure(figsize=(20, 10))
    plot_tree(final_model, num_trees=num)
    plt.title(f"Arbre de Décision n°{num + 1}")
    plt.tight_layout()
    plt.savefig(f"{output_base_path}/arbre_decision_{num + 1}.png", dpi=300)
    plt.close()

# Préparation des prédictions
y_pred_train = np.exp(final_model.predict(X_train))  # Revenir à l'échelle originale
y_pred_test = np.exp(final_model.predict(X_test))  # Revenir à l'échelle originale
y_train_orig = np.exp(y_train)  # Revenir à l'échelle originale
y_test_orig = np.exp(y_test)  # Revenir à l'échelle originale

# DataFrame pour les prédictions d'entraînement
df_train_pred = X_train.copy()
df_train_pred["ID_Fruit"] = df_merged.loc[X_train.index, "ID_Fruit"].values
df_train_pred["Fruit_Code"] = df_merged.loc[X_train.index, "Fruit_Code"].values
df_train_pred["a_source"] = df_merged.loc[X_train.index, "a_source"].values
df_train_pred["b_source"] = df_merged.loc[X_train.index, "b_source"].values
df_train_pred["c_source"] = df_merged.loc[X_train.index, "c_source"].values
df_train_pred["midilli_a"] = df_merged.loc[X_train.index, "midilli_a"].values
df_train_pred["midilli_b"] = df_merged.loc[X_train.index, "midilli_b"].values
df_train_pred["k_ajusté"] = y_train_orig
df_train_pred["k_predit"] = y_pred_train
df_train_pred["Origine"] = "Train"
df_train_pred["Erreur_absolue"] = np.abs(df_train_pred["k_predit"] - df_train_pred["k_ajusté"])
df_train_pred["Erreur_relative_%"] = 100 * df_train_pred["Erreur_absolue"] / df_train_pred["k_ajusté"].replace(0, np.nan)
df_train_pred["Précision_%"] = 100 - df_train_pred["Erreur_relative_%"]

# DataFrame pour les prédictions de test
df_test_pred = X_test.copy()
df_test_pred["ID_Fruit"] = df_merged.loc[X_test.index, "ID_Fruit"].values
df_test_pred["Fruit_Code"] = df_merged.loc[X_test.index, "Fruit_Code"].values
df_test_pred["a_source"] = df_merged.loc[X_test.index, "a_source"].values
df_test_pred["b_source"] = df_merged.loc[X_test.index, "b_source"].values
df_test_pred["c_source"] = df_merged.loc[X_test.index, "c_source"].values
df_test_pred["midilli_a"] = df_merged.loc[X_test.index, "midilli_a"].values
df_test_pred["midilli_b"] = df_merged.loc[X_test.index, "midilli_b"].values
df_test_pred["k_ajusté"] = y_test_orig
df_test_pred["k_predit"] = y_pred_test
df_test_pred["Origine"] = "Test"
df_test_pred["Erreur_absolue"] = np.abs(df_test_pred["k_predit"] - df_test_pred["k_ajusté"])
df_test_pred["Erreur_relative_%"] = 100 * df_test_pred["Erreur_absolue"] / df_test_pred["k_ajusté"].replace(0, np.nan)
df_test_pred["Précision_%"] = 100 - df_test_pred["Erreur_relative_%"]

# DataFrame global
df_global_pred = pd.concat([df_train_pred, df_test_pred], axis=0).reset_index(drop=True)

# Bootstrap global pour IC 95% des prédictions
bootstrap_preds_all = np.zeros((len(X), n_iterations))
for i in range(n_iterations):
    indices = rng.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]
    model_boot = XGBRegressor(**best_params)
    model_boot.fit(X_boot, y_boot)
    preds = np.exp(model_boot.predict(X))  # Revenir à l'échelle originale
    bootstrap_preds_all[:, i] = preds

lower_bounds = np.percentile(bootstrap_preds_all, 2.5, axis=1)
upper_bounds = np.percentile(bootstrap_preds_all, 97.5, axis=1)
mean_preds = np.mean(bootstrap_preds_all, axis=1)

df_boot_results = df_merged.loc[X.index, ["ID_Fruit", "Fruit_Code"]].copy()
df_boot_results["k_predit_boot"] = mean_preds
df_boot_results["IC_inf"] = lower_bounds
df_boot_results["IC_sup"] = upper_bounds

# Fusion avec les résultats bootstrap pour df_global_pred
df_global_pred = df_global_pred.merge(df_boot_results, on=["ID_Fruit", "Fruit_Code"], how="left")

# Fusion avec les résultats bootstrap pour df_train_pred et df_test_pred
df_train_pred = df_train_pred.merge(
    df_boot_results[["ID_Fruit", "Fruit_Code", "k_predit_boot", "IC_inf", "IC_sup"]],
    on=["ID_Fruit", "Fruit_Code"],
    how="left"
)
df_test_pred = df_test_pred.merge(
    df_boot_results[["ID_Fruit", "Fruit_Code", "k_predit_boot", "IC_inf", "IC_sup"]],
    on=["ID_Fruit", "Fruit_Code"],
    how="left"
)

# Réorganisation des colonnes dans l'ordre souhaité
colonnes_finales = [
    "Fruit_Code", "ID_Fruit", "Origine", "Masse_tot_g_initiale", "a_mm", "a_source",
    "b_mm", "b_source", "c_mm", "c_source", "midilli_a", "midilli_b", "midilli_n",
    "k_ajusté", "k_predit", "k_predit_boot", "IC_inf", "IC_sup",
    "Erreur_absolue", "Erreur_relative_%", "Précision_%"
]
df_global_pred = df_global_pred[colonnes_finales].sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)
df_train_pred = df_train_pred[colonnes_finales].sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)
df_test_pred = df_test_pred[colonnes_finales].sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)

# Analyse des performances par Fruit_Code
print("\nPerformances par Fruit_Code :")
for fruit in df_global_pred["Fruit_Code"].unique():
    subset = df_global_pred[df_global_pred["Fruit_Code"] == fruit]
    r2 = r2_score(subset["k_ajusté"], subset["k_predit"])
    rmse = np.sqrt(mean_squared_error(subset["k_ajusté"], subset["k_predit"]))
    mae = mean_absolute_error(subset["k_ajusté"], subset["k_predit"])
    print(f"Fruit {fruit} - R²: {r2:.4f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")

# Figures des prédictions
plt.figure(figsize=(6, 6))
for origine, couleur in zip(["Train", "Test"], ["dodgerblue", "darkorange"]):
    subset = df_global_pred[df_global_pred["Origine"] == origine]
    plt.scatter(subset["k_ajusté"], subset["k_predit"], alpha=0.7, label=origine, color=couleur)
min_val = df_global_pred["k_ajusté"].min()
max_val = df_global_pred["k_ajusté"].max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
plt.xlabel("k ajusté")
plt.ylabel("k prédit")
plt.title("Scatter plot : k ajusté vs k prédit (Train/Test)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_base_path}/scatter_ajusté_vs_predit.png", dpi=300)
plt.close()

# Scatter plot avec IC 95%
plt.figure(figsize=(6, 6))
for origine, couleur in zip(["Train", "Test"], ["dodgerblue", "darkorange"]):
    subset = df_global_pred[df_global_pred["Origine"] == origine]
    plt.errorbar(
        subset["k_ajusté"],
        subset["k_predit_boot"],
        yerr=[subset["k_predit_boot"] - subset["IC_inf"], subset["IC_sup"] - subset["k_predit_boot"]],
        fmt='o',
        color=couleur,
        alpha=0.7,
        label=f"{origine} (±IC95%)",
        ecolor='black',
        capsize=4,
        capthick=1.2,
        elinewidth=1
    )
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="y = x")
plt.xlabel("k ajusté")
plt.ylabel("k prédit")
plt.title("Prédictions de k avec IC 95%")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_base_path}/scatter_ajusté_vs_predit_IC95.png", dpi=300)
plt.close()

# Histogramme des erreurs
plt.figure(figsize=(8, 5))
plt.hist(df_global_pred["Erreur_absolue"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution des erreurs absolues pour k")
plt.xlabel("Erreur absolue")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(f"{output_base_path}/hist_erreur_absolue.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df_global_pred["Précision_%"], bins=20, color="lightgreen", edgecolor="black")
plt.title("Distribution des précisions (%) pour k")
plt.xlabel("Précision [%]")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(f"{output_base_path}/hist_precision.png", dpi=300)
plt.close()

# Gain cumulé par arbre
booster = final_model.get_booster()
df_trees = booster.trees_to_dataframe()
gain_by_tree = df_trees.groupby('Tree')['Gain'].sum()
gain_by_tree_cumsum = np.cumsum(gain_by_tree)
plt.figure(figsize=(10, 6))
plt.plot(gain_by_tree.index, gain_by_tree_cumsum, marker='o', linestyle='-')
plt.xlabel("Numéro d'arbre")
plt.ylabel("Gain cumulé")
plt.title("Évolution du gain cumulé au fil des arbres\n(Modèle XGBoost final)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_base_path}/gain_cumule_par_arbre.png", dpi=300)
plt.close()

# Exportation Excel
export_path = f"{output_base_path}/Resultats_XGBoost_k_Modele1_CV.xlsx"
with pd.ExcelWriter(export_path, engine="openpyxl") as writer:
    df_global_pred.to_excel(writer, sheet_name="Prédictions Globales", index=False)
    df_train_pred.to_excel(writer, sheet_name="Prédictions Train-Test", index=False)
    df_test_pred.to_excel(writer, sheet_name="Prédictions Test Final", index=False)
    pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(r2_scores))] + ["R² moyen", "R² écart-type", "RMSE moyen", "RMSE écart-type", "MAE moyen", "MAE écart-type"],
        "R2": r2_scores + [np.mean(r2_scores), np.std(r2_scores), np.nan, np.nan, np.nan, np.nan],
        "RMSE": rmse_scores + [np.nan, np.nan, np.mean(rmse_scores), np.std(rmse_scores), np.nan, np.nan],
        "MAE": mae_scores + [np.nan, np.nan, np.nan, np.nan, np.mean(mae_scores), np.std(mae_scores)]
    }).to_excel(writer, sheet_name="Résultats_CV", index=False)
    resume_df.to_excel(writer, sheet_name="Résumé_Scores", index=False)
    pd.DataFrame(list(best_params.items()), columns=["Paramètre", "Valeur"]).to_excel(writer, sheet_name="Paramètres_Optuna", index=False)
    df_bootstrap_metrics.to_excel(writer, sheet_name="Bootstrap_Metrics_Test", index=False)

# Sauvegarde du modèle
joblib.dump(final_model, f"{output_base_path}/xgboost_k_modele_1.joblib")

print(f"Modèle et résultats sauvegardés dans : {output_base_path}")

