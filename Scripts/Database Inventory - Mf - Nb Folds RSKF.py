#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs)
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09
 
# Scripts pour afficher l'inventaire de la base de données et le choix du nombre de folds pour la validation croisée répétée stratifiée en k-plis (RSKF CV).
# Les spécifications de ce script concernent uniquement la prédiction de la masse finale (Mf), obtenue en fin d'expérience au temps final tf.

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

# Filtrage pour la prédiction de la Masse Finale
init_df = df[df["Statut Masse"] == "INITIALE"].copy()
finale_df = df[df["Statut Masse"] == "FINALE"].copy()

merged_df = pd.merge(
    init_df[["ID Fruit", "Fruit_Code", "a [mm]", "b [mm]", "c [mm]", "Masse tot [g]"]],
    finale_df[["ID Fruit", "Masse tot [g]"]],
    on="ID Fruit",
    suffixes=("_init", "_finale")
)

required_columns = ["a [mm]", "b [mm]", "c [mm]", "Masse tot [g]_init", "Masse tot [g]_finale"]
clean_df = merged_df.dropna(subset=required_columns).copy()

# Vérification des dimensions manquantes dans les données initiales fusionnées
nb_missing_dims = merged_df[["a [mm]", "b [mm]", "c [mm]"]].isnull().any(axis=1).sum()

if nb_missing_dims > 0:
    print(f"\n⚠️  Attention : {nb_missing_dims} échantillons disposent d'une masse finale mais ont au moins une dimension initiale / variable d'entrée manquante (a, b ou c).")
    print("   → Ces valeurs devront être comblées avant modélisation (ex. : via prédiction à partir de Fruit_Code et Masse initiale).")
    
    # Mise en garde sur le plan de validation
    print("\n⚠️  Remarque : le plan de validation croisée proposé ci-dessous est basé sur l’inventaire actuel (échantillons complets).")
    print("   → Si les dimensions manquantes sont comblées par prédiction ou autre méthode, il conviendra de relancer ce script.")
    print("   → Cela permettra de recalculer la configuration optimale des folds selon la nouvelle distribution des échantillons.")
else:
    print("\n✔ Toutes les dimensions a, b, c sont renseignées pour les échantillons disposant d'une masse finale.")

# Inventaire pour la Masse Finale
print("\n=== INVENTAIRE POUR LA PRÉDICTION DE LA MASSE FINALE ===")
# Nombre d'échantillons ayant à la fois une masse initiale et une masse finale
id_initiaux = set(df[df["Statut Masse"] == "INITIALE"]["ID Fruit"])
id_finales = set(df[df["Statut Masse"] == "FINALE"]["ID Fruit"])
id_communs = id_initiaux.intersection(id_finales)
nb_init_et_finale = len(id_communs)
print(f"• Nombre d’échantillons ayant à la fois une masse initiale et une masse finale : {nb_init_et_finale} / {total_all_count} ({100 * nb_init_et_finale / total_all_count:.1f} %)")

# Nombre d’échantillons exploitables après exclusion des valeurs manquantes (a, b, c)
print(f"• Nombre total d’échantillons utilisables (possédant toutes les variables d'entrées requises pour la prédiction) : {len(clean_df)} / {nb_init_et_finale} ({100 * len(clean_df) / nb_init_et_finale:.1f} %)")

print("• Répartition des échantillons par fruit :")

used_counts = clean_df["Fruit_Code"].value_counts().sort_index()
min_class_size = used_counts.min()

print("• Répartition des échantillons par fruit :")

for code in fruit_names:
    used = used_counts.get(code, 0)
    total_base = total_all_samples.get(code, 0)
    total_clean = len(clean_df)
    
    ratio_base = 100 * used / total_base if total_base > 0 else 0
    ratio_clean = 100 * used / total_clean if total_clean > 0 else 0

    print(f"   - {fruit_names[code]} : {used} / {total_base} ({ratio_base:.1f} % base totale),    {used} / {total_clean} ({ratio_clean:.1f} % des échantillons exploitables)")

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

