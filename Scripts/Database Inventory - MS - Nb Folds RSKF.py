#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs)
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09
 
# Scripts pour afficher l'inventaire de la base de données et le choix du nombre de folds pour la validation croisée répétée stratifiée en k-plis (RSKF CV).
# Les spécifications de ce script concernent uniquement la prédiction de la masse sèche (MS).

import pandas as pd
import numpy as np

# Chargement des données
file_path = "D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Data Fruits Amazoniens V8.xlsx"
df = pd.read_excel(file_path, sheet_name="Données")

# Inventaire général (tous les échantillons sont présents)
total_all_samples = df[df["Statut Masse"] == "INITIALE"]["Fruit_Code"].value_counts().sort_index()
total_all_count = df[df["Statut Masse"] == "INITIALE"].shape[0]

print("\n=== INVENTAIRE COMPLET DE LA BASE DE DONNÉES ===")
print(f"• Nombre total d'échantillons INITIAUX dans la base : {total_all_count}")
print("• Répartition des échantillons par fruit :")
fruit_names = {
    0: "Açai",
    1: "Guarana anguleux",
    2: "Guarana sphérique",
    3: "Poivre",
    4: "Cacao"
}
for code, total in total_all_samples.items():
    print(f"   - {fruit_names[code]} : {total}")

# Filtrage pour la prédiction de la Masse Sèche
init_df = df[df["Statut Masse"] == "INITIALE"].copy()
seche_df = df[df["Statut Masse"] == "SECHE"].copy()

merged_df = pd.merge(
    init_df[["ID Fruit", "Fruit_Code", "a [mm]", "b [mm]", "c [mm]", "Masse tot [g]"]],
    seche_df[["ID Fruit", "Masse tot [g]"]],
    on="ID Fruit",
    suffixes=("_init", "_seche")
)

required_columns = ["a [mm]", "b [mm]", "c [mm]", "Masse tot [g]_init", "Masse tot [g]_seche"]
clean_df = merged_df.dropna(subset=required_columns).copy()

# Inventaire pour la Masse Sèche
print("\n=== INVENTAIRE POUR LA PRÉDICTION DE LA MASSE SÈCHE ===")
print(f"• Nombre total d’échantillons utilisables : {len(clean_df)}")
print("• Répartition des échantillons par fruit :")

used_counts = clean_df["Fruit_Code"].value_counts().sort_index()
min_class_size = used_counts.min()

for code in used_counts.index:
    used = used_counts[code]
    total = total_all_samples.get(code, 0)
    ratio = 100 * used / total if total > 0 else 0
    print(f"   - {fruit_names[code]} : {used} / {total} ({ratio:.1f} %)")

print(f"• Fruit limitant (le plus rare pour CV) : {fruit_names[used_counts.idxmin()]} ({min_class_size})")

# Plan de validation
total_folds = min_class_size
reserved_test_fold = 1
folds_for_cv = total_folds - reserved_test_fold

# Détermination du nombre de validation pour RSKF CV
min_validations = 25
max_validations = 35

best_combo = None
min_gap = float('inf')
for n_repeats in range(1, 21):
    total_validations = folds_for_cv * n_repeats
    if min_validations <= total_validations <= max_validations:
        gap = abs(total_validations - (min_validations + max_validations) / 2)
        if gap < min_gap:
            min_gap = gap
            best_combo = (folds_for_cv, n_repeats)

# Affichage du plan de validation
if best_combo:
    n_splits_final, n_repeats_final = best_combo
    print("\n=== PLAN DE VALIDATION PROPOSÉ ===")
    print(f"• Nombre total de folds disponibles (limité par le fruit rare) : {total_folds}")
    print(f"• Folds réservés pour test final : {reserved_test_fold}")
    print(f"• Folds restants pour la validation croisée répétée stratifiée interne (RSKF CV) : {folds_for_cv}")
    print(f"• Nombre optimal de folds (n_splits) pour RSKF CV interne : {n_splits_final}")
    print(f"• Nombre recommandé de répétitions (n_repeats) : {n_repeats_final}")
    print(f"• Total de validations RSKF CV internes : {n_splits_final * n_repeats_final}")
    print(f"• Ratio train/test dans chaque split RSKF CV : {(n_splits_final - 1) / n_splits_final:.0%} / {100 / n_splits_final:.0f}%")
else:
    print("\nAucune combinaison (n_splits × n_repeats) ne respecte les contraintes définies.")

print(f"\n Taux final d’échantillons totalement exploitables actuellement : {len(clean_df)} / {total_all_count} ({100 * len(clean_df) / total_all_count:.1f} % de la base complète)")


# In[ ]:




