#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs)
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09

# Script pour appliquer un modèle conditionnel pour prédire la masse finale de fruits amazoniens. Le but de ce dernier est d’améliorer la précision de prédiction de la masse finale est combinant les qualités du modèle 1 et du modèle 2. 
# La base de données utilisée pour appliquer le modèle conditionnel est la même que celle utilisée pour entraîner les modèles 1 et 2. Elle contient les caractéristiques initiales des fruits amazoniens (masse initiale, dimensions a, b, c) ainsi que leur masse finale réelle après étuve.
# Ce fichier est structuré et provient du traitement des données expérimentales originales.
# Ce script repose donc sur deux modèles préalablement entraînés :
# - Modèle 1 : prédiction directe de la masse finale. Cela impacte davantage l’erreur absolue entre la masse initiale et la masse finale, et donc est favorable pour les fruits dont la masse est moyenne à élevée.
# - Modèle 2 : prédiction du facteur masse initiale / masse finale, permettant de reconstituer la masse finale par division. Cela impacte davantage l’erreur relative entre la masse initiale et la masse finale, et donc est favorable pour les fruits dont la masse est faible.
# Le modèle conditionnel utilise automatiquement le modèle 1 ou 2 en fonction de la masse initiale par rapport à une masse critique M_critique = Erreur Absolue moyenne/Erreur Relative seuil. Le seuil de l’erreur relative est ici de 10%.
# Étapes :
# - Chargement des modèles pré-entraînés indépendamment
# - Chargement des données utilisées pour la prédiction
# - Calcul automatique de la masse critique M_critique utilisée comme condition d’utilisation du modèle 1 ou 2
# - Application du modèle 1 ou 2 selon la masse initiale
# - Évaluation finale (R², RMSE, MAE) spécifique au modèle conditionnel, adapté en fonction du modèle 1 ou 2 considéré, car la prédiction effectuée est différente pour obtenir la masse finale.
# - Visualisation finale
# - Export des résultats dans un fichier Excel structuré

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# Chargement du fichier Excel contenant les données utilisées pour le modèle 2
# Le fichier doit contenir au moins : Masse_tot_g_initiale, Masse_tot_g_finale
df = pd.read_excel("D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle 2/Resultats_XGBoost_Mf_Modele2_CV.xlsx", sheet_name="Prédictions Train-Test")

# Chargement des deux modèles
# Chargement du modèle 1 (prédiction directe de la masse finale)
modele_1_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle 1/xgboost_Mf_modele_1.joblib"
modele_1 = joblib.load(modele_1_path)

# Chargement du modèle 2 (prédiction du facteur de réduction)
modele_2_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle 2/xgboost_Mf_modele_2.joblib"
modele_2 = joblib.load(modele_2_path)

# Calcul automatique de la masse critique : M_critique
# Erreur absolue moyenne sur les données du modèle 2
EA_moyenne = np.mean(df["Erreur_absolue"])
ER_seuil = 0.1  # 10 % d'erreur relative autorisée

# Calcul de M_critique
M_critique = EA_moyenne / ER_seuil
print(f"Valeur calculée de M_critique : {M_critique:.4f} g")

# Application du modèle conditionnel
# Préparation des variables nécessaires
X = df[["Masse_tot_g_initiale", "a_mm", "b_mm", "c_mm"]]
masse_initiale_all = df["Masse_tot_g_initiale"].values
masse_finale_reelle = df["Masse_finale_réelle"].values

# Initialisation des listes
masse_finale_predite_meta = []
IC_inf_meta = []
IC_sup_meta = []

# Chargement des valeurs de bootstrap (IC95 %) pour les deux modèles
df_boot_1 = pd.read_excel("D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle 1/Resultats_XGBoost_Mf_Modele1_CV.xlsx", sheet_name="Prédictions Train-Test")
df_boot_2 = pd.read_excel("D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle 2/Resultats_XGBoost_Mf_Modele2_CV.xlsx", sheet_name="Prédictions Train-Test")

# Fusion avec la base principale selon Fruit_Code et ID_Fruit
df_merged_1 = pd.merge(df, df_boot_1[["Fruit_Code", "ID_Fruit", "IC_inf", "IC_sup"]], on=["Fruit_Code", "ID_Fruit"], how="left", suffixes=('', '_mod1'))
df_merged_2 = pd.merge(df, df_boot_2[["Fruit_Code", "ID_Fruit", "IC_inf", "IC_sup"]], on=["Fruit_Code", "ID_Fruit"], how="left", suffixes=('', '_mod2'))

# Boucle sur chaque échantillon
for i in range(len(X)):
    masse_init = masse_initiale_all[i]

    if masse_init > M_critique:
        # Modèle 1 : prédiction directe + IC depuis le merged
        masse_finale_pred = modele_1.predict(X.iloc[i:i+1])[0]
        ic_inf = df_merged_1.loc[i, "IC_inf"]
        ic_sup = df_merged_1.loc[i, "IC_sup"]
    else:
        # Modèle 2 : prédiction via facteur + conversion des IC
        facteur_pred = modele_2.predict(X.iloc[i:i+1])[0]
        if facteur_pred == 0:
            masse_finale_pred = np.nan
            ic_inf = np.nan
            ic_sup = np.nan
        else:
            masse_finale_pred = masse_init / facteur_pred
            ic_inf = df_merged_2.loc[i, "IC_inf"]
            ic_sup = df_merged_2.loc[i, "IC_sup"]

    # Ajout des valeurs à chaque liste
    masse_finale_predite_meta.append(masse_finale_pred)
    IC_inf_meta.append(ic_inf)
    IC_sup_meta.append(ic_sup)
# Conversion en array numpy pour s’assurer que les calculs du R², du RMSE et du MAE soient corrects
masse_finale_predite_meta = np.array(masse_finale_predite_meta)

# Évaluation du modèle conditionnel
r2_meta = r2_score(masse_finale_reelle, masse_finale_predite_meta)
rmse_meta = np.sqrt(mean_squared_error(masse_finale_reelle, masse_finale_predite_meta))
mae_meta = mean_absolute_error(masse_finale_reelle, masse_finale_predite_meta)

print("\nPerformance du Modèle Conditionnel :")
print(f"R² : {r2_meta:.4f}")
print(f"RMSE : {rmse_meta:.4f} g")
print(f"MAE : {mae_meta:.4f} g")

# Calcul des erreurs pour le modèle conditionnel
erreur_absolue_meta = np.abs(masse_finale_reelle - masse_finale_predite_meta)
precision_meta = 100 - (100 * erreur_absolue_meta / masse_finale_reelle)

# Figure : Scatter plot : masse finale réelle vs masse finale prédite
plt.figure(figsize=(6, 6))
plt.scatter(masse_finale_reelle, masse_finale_predite_meta, alpha=0.7, color="dodgerblue")
# Ajout de la diagonale
plt.plot([masse_finale_reelle.min(), masse_finale_reelle.max()],
         [masse_finale_reelle.min(), masse_finale_reelle.max()],
         'r--')
plt.xlabel("Masse finale réelle [g]")
plt.ylabel("Masse finale prédite [g]")
plt.title("Scatter plot : Masse finale réelle vs prédite\n(Modèle conditionnel)")
plt.annotate(f"M_critique = {M_critique:.3f} g", xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round", fc="w")) # Afficher M_critique
plt.grid(True)
plt.tight_layout()

# Sauvegarde de la figure
scatter_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle Conditionnel/scatter_reelle_vs_predite_Modele_Conditionnel.png"
plt.savefig(scatter_path, dpi=300)
plt.close()

# Figure : Scatter plot : masse finale réelle vs masse finale prédite avec IC95%
# Conversion en tableau numpy 
masse_finale_predite_meta = np.array(masse_finale_predite_meta)
IC_inf_meta = np.array(IC_inf_meta)
IC_sup_meta = np.array(IC_sup_meta)

# Correction des éventuels inversions de bornes
IC_inf_corr = np.minimum(IC_inf_meta, masse_finale_predite_meta)
IC_sup_corr = np.maximum(IC_sup_meta, masse_finale_predite_meta)

# Recalcul des demi-largeurs
yerr_inf = masse_finale_predite_meta - IC_inf_corr
yerr_sup = IC_sup_corr - masse_finale_predite_meta
yerr = [yerr_inf, yerr_sup]

# Création du scatter plot avec barres d'erreur
plt.figure(figsize=(6, 6))
plt.errorbar(masse_finale_reelle, masse_finale_predite_meta, 
yerr=yerr, 
fmt='o',
        color="dodgerblue",
        alpha=0.7,
        label=f"{origine} (±IC95%)",
        ecolor='black',
        capsize=4,
        capthick=1.2,
        elinewidth=1
)
# Ajout de la diagonale
plt.plot([masse_finale_reelle.min(), masse_finale_reelle.max()],
         [masse_finale_reelle.min(), masse_finale_reelle.max()],
         'r--')
plt.xlabel("Masse finale réelle [g]")
plt.ylabel("Masse finale prédite (modèle conditionnel) [g]")
plt.title("Scatter plot : Masse finale réelle vs prédite\n(Modèle conditionnel)")
plt.annotate(f"M_critique = {M_critique:.3f} g", xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round", fc="w"))
plt.grid(True)
plt.tight_layout()

# Sauvegarde de la figure
scatter_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle Conditionnel/scatter_reelle_vs_predite_Modele_Conditionnel_IC95.png"
plt.savefig(scatter_path, dpi=300)
plt.close()


# Figure : Histogramme des erreurs absolues [g]
plt.figure(figsize=(8, 5))
plt.hist(erreur_absolue_meta, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution des erreurs absolues\n(Modèle conditionnel)")
plt.xlabel("Erreur absolue [g]")
plt.ylabel("Fréquence")
plt.grid(True)
plt.tight_layout()

# Sauvegarde de la figure
hist_erreur_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle Conditionnel/hist_erreur_absolue_Modele_Conditionnel.png"
plt.savefig(hist_erreur_path, dpi=300)
plt.close()

# Figure : Histogramme des précisions [%]
plt.figure(figsize=(8, 5))
plt.hist(precision_meta, bins=20, color="lightgreen", edgecolor="black")
plt.title("Distribution des précisions (%)\n(Modèle conditionnel)")
plt.xlabel("Précision [%]")
plt.ylabel("Fréquence")
plt.grid(True)
plt.tight_layout()

# Sauvegarde de la figure
hist_precision_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle Conditionnel/hist_precision_Modele_Conditionnel.png"
plt.savefig(hist_precision_path, dpi=300)
plt.close()

# Export structuré des résultats
# Création du DataFrame final avec les colonnes classiques
results_meta_df = pd.DataFrame({
    "Fruit_Code": df["Fruit_Code"].values,
    "ID_Fruit": df["ID_Fruit"].values,
    "Masse_tot_g_initiale": df["Masse_tot_g_initiale"].values,
    "a_mm": df["a_mm"].values,
    "b_mm": df["b_mm"].values,
    "c_mm": df["c_mm"].values,
    "Masse_finale_réelle": masse_finale_reelle,
    "Masse_finale_prédite_meta": masse_finale_predite_meta,
})
# Ajout des erreurs
results_meta_df["IC_inf"] = IC_inf_meta
results_meta_df["IC_sup"] = IC_sup_meta
results_meta_df["Erreur_absolue"] = np.abs(results_meta_df["Masse_finale_réelle"] - results_meta_df["Masse_finale_prédite_meta"])
results_meta_df["Erreur_relative_%"] = 100 * results_meta_df["Erreur_absolue"] / results_meta_df["Masse_finale_réelle"]
results_meta_df["Précision_%"] = 100 - results_meta_df["Erreur_relative_%"]
# Tri par type de fruit et ID
results_meta_df = results_meta_df.sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)

# Export dans un nouveau fichier Excel
export_meta_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Mf V2/Modèle Conditionnel/Resultats_Modele_Conditionnel.xlsx"
with pd.ExcelWriter(export_meta_path, engine="openpyxl") as writer:
    # Première feuille : prédictions
    results_meta_df.to_excel(writer, sheet_name="Prédictions_Modèle_Conditionnel", index=False)
    # Deuxième feuille : performances globales
    perf_meta_df = pd.DataFrame({
        "Métrique": ["R²", "RMSE [g]", "MAE [g]"],
        "Valeur": [r2_meta, rmse_meta, mae_meta]
    })
    perf_meta_df.to_excel(writer, sheet_name="Performance_Modèle_Conditionnel", index=False)
    # Sauvegarde de M_critique
    seuil_df = pd.DataFrame({
        "M_critique (g)": [M_critique],
        "Erreur absolue moyenne (g)": [EA_moyenne],
        "Seuil erreur relative (%)": [ER_seuil * 100]
    })
    seuil_df.to_excel(writer, sheet_name="Seuil_M_critique", index=False)

# Print indiquant la fin des opérations
print(f"\nRésultats du modèle conditionnel sauvegardés dans : {export_meta_path}")

