#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs)
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09

# Script pour compléter les dimensions ai bi ci manquantes des fruits sphériques à partir de leur masse initiale et de la variable catégorielle Fruit_Code.
# Ces préditions sont réalisées grâce aux 3 modèles XGBoost pour prédire ai, bi et ci.

import pandas as pd
import numpy as np
import joblib
import os

# === Chemins ===
data_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Data Fruits Amazoniens V8.xlsx"
result_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Dimensions_Prédites_Modèle_3.xlsx"

modele3_paths = {
    "a": "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_a_mm/xgboost_modele_final_a.joblib",
    "b": "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_b_mm/xgboost_modele_final_b.joblib",
    "c": "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/XGBoost Dimensions V2/Modèle_3/Modèle_3_c_mm/xgboost_modele_final_c.joblib"
}

# === Chargement des modèles ===
models = {dim: joblib.load(path) for dim, path in modele3_paths.items()}

# === Chargement des données ===
df = pd.read_excel(data_path, sheet_name="Données")
df = df.rename(columns={
    "Masse tot [g]": "Masse_tot_g",
    "ID Fruit": "ID_Fruit",
    "Fruit_Code": "Fruit_Code",
    "Statut Masse": "Statut_Masse",
    "a [mm]": "a_mm",
    "b [mm]": "b_mm",
    "c [mm]": "c_mm"
})

# === Identifier les ID_Fruit ayant atteint le statut FINALE
fruits_complets = df[df["Statut_Masse"] == "FINALE"]["ID_Fruit"].unique()

# === Extraire les données INITIALES pour ces fruits
df_init = df[(df["Statut_Masse"] == "INITIALE") & (df["ID_Fruit"].isin(fruits_complets))]
df_init = df_init[["ID_Fruit", "Fruit_Code", "Masse_tot_g", "a_mm", "b_mm", "c_mm"]].dropna(subset=["Masse_tot_g"])

# === Préparer le DataFrame final
results_df = df_init[["ID_Fruit", "Fruit_Code", "Masse_tot_g"]].copy()

# === Prédiction conditionnelle ou utilisation directe
for dim in ["a", "b", "c"]:
    col_predicted = f"{dim}_mm"
    col_source = f"{dim}_source"

    results_df[col_predicted] = np.nan
    results_df[col_source] = np.nan

    for idx, row in df_init.iterrows():
        fruit_id = row["ID_Fruit"]
        fruit_code = row["Fruit_Code"]
        masse = row["Masse_tot_g"]
        value_measured = row[f"{dim}_mm"]

        if pd.notna(value_measured):
            results_df.loc[results_df["ID_Fruit"] == fruit_id, col_predicted] = value_measured
            results_df.loc[results_df["ID_Fruit"] == fruit_id, col_source] = "Mesurée"
        elif fruit_code != 4:  # Pas de prédiction pour cacao

    
            X_input = pd.DataFrame([[fruit_code, masse]], columns=["Fruit_Code", "Masse_tot_g"])
            pred_mod3 = models[dim].predict(X_input)[0]
            
            results_df.loc[results_df["ID_Fruit"] == fruit_id, col_predicted] = pred_mod3
            results_df.loc[results_df["ID_Fruit"] == fruit_id, col_source] = "Prédite"
        

# === Tri final des colonnes
ordered_cols = [
    "Fruit_Code", "ID_Fruit", "Masse_tot_g",
    "a_mm", "a_source",
    "b_mm", "b_source",
    "c_mm", "c_source"
]
results_df = results_df[ordered_cols].sort_values(by=["Fruit_Code", "ID_Fruit"]).reset_index(drop=True)

# === Sauvegarde
results_df.to_excel(result_path, index=False)
print(f"Fichier final trié enregistré dans : {result_path}")

