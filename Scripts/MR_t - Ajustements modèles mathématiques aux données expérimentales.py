#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Auteur : Pierre-Henri Motkin
# Affiliation : Université Libre de Bruxelles (ULB)
# Service : Transfers, Interfaces & Processes (TIPs) 
# Mémoire de master | Année académique 2024–2025
# Titre : Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens
# Version : 1.0.0 | Date : 2025-08-09

# Script pour ajuster les modèles mathématiques aux données expérimentales de MR(t). Plusieurs modèles sont présentés : Henderson et Pabis, Midilli, et toutes les variantes du modèle mixte développées.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

# =============================================================================
# Configuration des chemins d'accès pour les données et les sorties
# =============================================================================
input_path = r'D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Data Fruits Amazoniens V8.xlsx'
export_path = r'D:/Mémoire ULB 2024-2025/1 - Avancées Travail Personnel/Machine Learning Final/MR_t_V3'

# Création des dossiers pour stocker les figures
figures_exp_path = os.path.join(export_path, 'Figures_MR_exp')
figures_henderson_path = os.path.join(export_path, 'Figures_MR_Henderson')
figures_aghbashlo_path = os.path.join(export_path, 'Figures_MR_Aghbashlo')
figures_midilli_path = os.path.join(export_path, 'Figures_MR_Midilli')
figures_mixte_path = os.path.join(export_path, 'Figures_MR_Mixte - Lineaire et Midilli')
figures_mixte_a_fixé_path = os.path.join(export_path, 'Figures_MR_Mixte_a_fixé - Lineaire et Midilli')
figures_mixte_a_b_fixés_path = os.path.join(export_path, 'Figures_MR_Mixte_a_b_fixés - Lineaire et Midilli')
figures_mixte_a_n_b_fixés_path = os.path.join(export_path, 'Figures_MR_Mixte_a_n_b_fixés - Lineaire et Midilli')
figures_residus_mixte_path = os.path.join(export_path, 'Figures_Residus_Mixte')
figures_residus_mixte_a_fixé_path = os.path.join(export_path, 'Figures_Residus_Mixte_a_fixé')
figures_residus_mixte_a_b_fixés_path = os.path.join(export_path, 'Figures_Residus_Mixte_a_b_fixés')
figures_residus_mixte_a_n_b_fixés_path = os.path.join(export_path, 'Figures_Residus_Mixte_a_n_b_fixés')

for path in [
    figures_exp_path,
    figures_henderson_path,
    figures_aghbashlo_path,
    figures_midilli_path,
    figures_mixte_path,
    figures_mixte_a_fixé_path,
    figures_mixte_a_b_fixés_path,
    figures_mixte_a_n_b_fixés_path,
    figures_residus_mixte_path,
    figures_residus_mixte_a_fixé_path,
    figures_residus_mixte_a_b_fixés_path,
    figures_residus_mixte_a_n_b_fixés_path
]:
    os.makedirs(path, exist_ok=True)

# =============================================================================
# Chargement et prétraitement des données
# =============================================================================
df = pd.read_excel(input_path, sheet_name='Données')

# Calcul des valeurs initiales (X0) et finales (Xf)
df['X0'] = df[df['Statut Masse'] == 'INITIALE'].groupby('ID Fruit')['Xi tot [g eau/g MS]'].transform('first')
df['Xf'] = df[df['Statut Masse'] == 'FINALE'].groupby('ID Fruit')['Xi tot [g eau/g MS]'].transform('first')
df['X0'] = df.groupby('ID Fruit')['X0'].transform('first')
df['Xf'] = df.groupby('ID Fruit')['Xf'].transform('first')

# Calcul du Moisture Ratio (MR(t))
df['MR(t)'] = np.where(
    (df['X0'].notna()) & (df['Xf'].notna()),
    (df['Xi tot [g eau/g MS]'] - df['Xf']) / (df['X0'] - df['Xf']),
    np.nan
)

# Nettoyage des données
df_clean = df.dropna(subset=['MR(t)', 'Temps [min]']).query("`Statut Masse` != 'SECHE'")

# Création des sous-dossiers pour chaque type de fruit
fruit_types = df_clean['ID Fruit'].apply(lambda x: x.split('_')[0] if '_' in x else x).unique()
for fruit_type in fruit_types:
    os.makedirs(os.path.join(figures_residus_mixte_path, fruit_type), exist_ok=True)
    os.makedirs(os.path.join(figures_residus_mixte_a_fixé_path, fruit_type), exist_ok=True)
    os.makedirs(os.path.join(figures_residus_mixte_a_b_fixés_path, fruit_type), exist_ok=True)
    os.makedirs(os.path.join(figures_residus_mixte_a_n_b_fixés_path, fruit_type), exist_ok=True)

# =============================================================================
# Définition des paramètres et des modèles
# =============================================================================
# Temps t1 à partir duquel l'équation de Midilli doit être ajustée pour le modèle mixte
t1_fixe = {
    0: 10,  # Açai
    1: 10,  # Guarana Anguleux
    2: 10,  # Guarana Sphérique
    3: 10,  # Poivre
    4: 20   # Cacao
}
# Valeurs de a fixées pour le modèle mixte
a_fixé_values = {
    '0': 0.9491,  # Açai
    '1': 0.9445,  # Guarana A
    '2': 0.9419,  # Guarana S
    '3': 0.8557,  # Poivre
    '4': 0.7511   # Cacao
}
# Valeurs de n fixées pour le modèle mixte (seront calculées à partir des valeurs médianes par type de fruit)
n_fixe_values = {}
# Valeurs de b fixées pour le modèle mixte (seront calculées à partir des valeurs médianes par type de fruit)
b_fixe_values = {}

def henderson_pabis(t, a, k):
    """Modèle Henderson-Pabis : MR(t) = a * exp(-k * t)"""
    return a * np.exp(-k * t)

def aghbashlo(t, k1, k2):
    """Modèle Aghbashlo : MR(t) = exp(-k1 * t / (1 + k2 * t))"""
    return np.exp(-k1 * t / (1 + k2 * t))

def midilli(t, a, k, n, b):
    """Modèle Midilli : MR(t) = a * exp(-k * t^n) + b * t"""
    return a * np.exp(-k * t**n) + b * t

def modele_mixte(t, t0, t1, pente, intercept, a, k, n, b):
    """Modèle mixte : linéaire entre t0 et t1, puis Midilli de t1 à tf fin de séchage (MR(tf)=0)"""
    return np.where(
        (t >= t0) & (t <= t1),
        pente * t + intercept,
        midilli(t, a, k, n, b)
    )

def midilli_mixte_a_fixé(t, t0, t1, pente, intercept, k, n, b, a_fixé):
    """Modèle mixte avec a fixé : linéaire entre t0 et t1, puis Midilli avec a fixé"""
    return np.where(
        (t >= t0) & (t <= t1),
        pente * t + intercept,
        midilli(t, a_fixé, k, n, b)
    )

def midilli_mixte_a_b_fixés(t, t0, t1, pente, intercept, k, n, a_fixé, b_fixé):
    """Modèle mixte avec a, b fixés : linéaire entre t0 et t1, puis Midilli avec a, b fixés"""
    return np.where(
        (t >= t0) & (t <= t1),
        pente * t + intercept,
        midilli(t, a_fixé, k, n, b_fixé)
    )

def midilli_mixte_a_n_b_fixés(t, t0, t1, pente, intercept, k, a_fixé, n_fixé, b_fixé):
    """Modèle mixte avec a, n, b fixés : linéaire entre t0 et t1, puis Midilli avec a, n, b fixés"""
    return np.where(
        (t >= t0) & (t <= t1),
        pente * t + intercept,
        midilli(t, a_fixé, k, n_fixé, b_fixé)
    )

# =============================================================================
# Fonction pour les résidus
# =============================================================================
def residuals_unweighted(model_func, t, MR, *args):
    """Calcule les résidus non pondérés"""
    def res(params): return MR - model_func(t, *params, *args)
    return res

# =============================================================================
# Initialisation des structures pour stocker les résultats
# =============================================================================
resultats_parametres = []
resultats_qualite_fit = []
analyse_residus = []
droites_lineaires = []
tous_r2_rmse_mixte = []
parametres_midilli_2 = []
predictions_mixte = []
t95_henderson_vals = []
t95_aghbashlo_vals = []
t95_midilli_vals = []
t95_mixte_vals = []
t95_mixte_a_fixé_vals = []
t95_mixte_a_b_fixés_vals = []
t95_mixte_a_n_b_fixés_vals = []
all_results = []

fruit_ids = df_clean['ID Fruit'].unique()

def multistart_optimization(residuals_func, bounds, n_starts=100):
    """Effectue une optimisation par moindres carrés non linéaires avec multistart et homoscédasticité"""
    lb, ub = bounds
    best_params = None
    best_loss = np.inf
    results_log = []

    for i in range(n_starts):
        x0 = np.random.uniform(lb, ub)
        try:
            res = least_squares(residuals_func, x0=x0, bounds=bounds)
            loss = np.sum(res.fun ** 2)
            if res.success and not np.any(np.isnan(res.x)) and not np.any(np.isinf(res.x)):
                on_boundary = np.any(np.isclose(res.x, lb)) or np.any(np.isclose(res.x, ub))
                results_log.append({
                    'start_index': i,
                    'x0': x0.tolist(),
                    'params': res.x.tolist(),
                    'loss': float(loss),
                    'success': res.success,
                    'on_boundary': on_boundary
                })
                if loss < best_loss and not on_boundary:
                    best_loss = loss
                    best_params = res.x
            else:
                results_log.append({
                    'start_index': i,
                    'x0': x0.tolist(),
                    'params': None,
                    'loss': None,
                    'success': False,
                    'on_boundary': False,
                    'error': 'Non-converged or invalid parameters'
                })
        except Exception as e:
            results_log.append({
                'start_index': i,
                'x0': x0.tolist(),
                'params': None,
                'loss': None,
                'success': False,
                'on_boundary': False,
                'error': str(e)
            })

    if best_params is None:
        valid_results = [r for r in results_log if r['success'] and r['loss'] is not None]
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['loss'])
            best_params = np.array(best_result['params'])
            best_loss = best_result['loss']
        else:
            best_params = np.full(len(lb), np.nan)
            best_loss = np.inf

    return best_params, results_log

# =============================================================================
# Première passe : Ajustement de tous les modèles jusqu'à midilli_mixte_a_fixé
# =============================================================================
for fruit in fruit_ids:
    subset = df_clean[df_clean['ID Fruit'] == fruit]
    t = subset['Temps [min]'].values
    MR = subset['MR(t)'].values

    if len(t) < 5:
        print(f"Échantillon {fruit} ignoré (nombre insuffisant de points)")
        continue

    t_sorted = np.sort(t)
    t0 = t_sorted[0]
    fruit_code = str(subset['Fruit_Code'].iloc[0])
    t1 = t1_fixe.get(int(fruit_code), 10)
    a_fixé = a_fixé_values.get(fruit_code, 1.0)
    X0, Xf = subset['X0'].iloc[0], subset['Xf'].iloc[0]

    # Génération des figures expérimentales
    plt.figure(figsize=(8, 6))
    plt.scatter(t, MR, color='black', label='Expérimental')
    plt.xlabel('Temps [min]')
    plt.ylabel('MR(t)')
    plt.title(f'Expérimental - {fruit}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_exp_path, f'{fruit}_MR_Exp.png'), dpi=300)
    plt.close()

    params_dict = {}
    pred_dict = {}
    r2_dict = {}
    rmse_dict = {}

    # Ajustements standards avec multistart non pondéré (homoscédasticité)
    for model, bounds, var in [
        (henderson_pabis, ([0.5, 0.0001], [1.5, 0.1]), 'henderson'),
        (aghbashlo, ([0.00001, 0.000001], [0.1, 0.1]), 'aghbashlo'),
        (midilli, ([0.5, 0.0001, 0.5, -0.01], [1.5, 0.1, 2.0, 0.01]), 'midilli')
    ]:
        lb, ub = bounds
        try:
            def residuals(params): return MR - model(t, *params)
            popt, log = multistart_optimization(residuals, bounds=(lb, ub), n_starts=100)
            for entry in log:
                all_results.append({
                    'ID Fruit': fruit,
                    'Fruit_Code': fruit_code,
                    'Modèle': var,
                    **entry
                })
            MR_pred = model(t, *popt)
            if np.any(np.isnan(MR_pred)) or np.any(np.isinf(MR_pred)):
                raise ValueError(f"MR_pred contains NaN or inf for {var}")
            rmse = np.sqrt(mean_squared_error(MR, MR_pred))
            r2 = r2_score(MR, MR_pred)
        except Exception as e:
            print(f"Ajustement échoué pour {var} sur {fruit} : {e}")
            popt = [np.nan] * len(lb)
            MR_pred = np.full_like(t, np.nan)
            rmse = np.nan
            r2 = np.nan
        params_dict[var] = popt
        pred_dict[var] = MR_pred
        rmse_dict[var] = rmse
        r2_dict[var] = r2

    # Ajustement mixte - Phase après t1
    t_mid = t[t > t1]
    MR_mid = MR[t > t1]

    if len(t_mid) >= 2:
        # Calcul de MR(t1) selon la droite linéaire (pour forcer la continuité)
        MR_t1_target = 1 + (t1 - t0) * ((MR[t >= t1][0] - 1) / (t[t >= t1][0] - t0)) if t1 > t0 else 1

        # Fonction résiduelle avec a2 fixé à MR(t1) expérimental (pour forcer la continuité)
        def residuals_midilli_constrained(params):
            k2, n2, b2 = params
            try:
                a2 = MR[t == t1][0]  # a2 fixé à la valeur expérimentale exacte de MR(t1)
                MR_pred = midilli(t_mid, a2, k2, n2, b2)
            except:
                return np.full_like(t_mid, 1e6)
            return MR_mid - MR_pred

        # Bornes pour les trois paramètres libres (a2 est déterminé par la contrainte)
        bounds_k2n2b2 = ([0.0001, 0.5, -0.01], [0.1, 2.0, 0.01])

        try:
            result_midilli, log = multistart_optimization(residuals_midilli_constrained, bounds_k2n2b2, n_starts=100)
            for entry in log:
                all_results.append({
                    'ID Fruit': fruit,
                    'Fruit_Code': fruit_code,
                    'Modèle': 'midilli_mixte',
                    **entry
                })
            k2, n2, b2 = result_midilli
            a2 = MR_t1_target / (np.exp(-k2 * t1 ** n2) + b2 * t1) if (np.exp(-k2 * t1 ** n2) + b2 * t1) != 0 else np.nan
            pente = (MR_t1_target - 1) / (t1 - t0) if t1 > t0 else 0
            intercept = 1
            MR_pred_mixte = modele_mixte(t, t0, t1, pente, intercept, a2, k2, n2, b2)
            r2_mixte = r2_score(MR, MR_pred_mixte)
            rmse_mixte = np.sqrt(mean_squared_error(MR, MR_pred_mixte))
        except Exception as e:
            print(f"Erreur dans l’ajustement mixte pour {fruit} : {e}")
            a2, k2, n2, b2 = np.nan, np.nan, np.nan, np.nan
            pente, intercept = np.nan, np.nan
            r2_mixte, rmse_mixte = np.nan, np.nan
            MR_pred_mixte = np.full_like(t, np.nan)
    else:
        print(f"Pas assez de points pour mixte sur {fruit}: t_mid={len(t_mid)}")
        a2, k2, n2, b2 = np.nan, np.nan, np.nan, np.nan
        pente, intercept = np.nan, np.nan
        r2_mixte, rmse_mixte = np.nan, np.nan
        MR_pred_mixte = np.full_like(t, np.nan)

    # Ajustement mixte avec a fixé - Phase après t1
    if len(t_mid) >= 2:
        def residuals_midilli_a_fixé(params, t_mid, MR_mid, a_fixé):
            k2, n2, b2 = params
            try:
                MR_pred = midilli(t_mid, a_fixé, k2, n2, b2)
                return MR_mid - MR_pred
            except:
                return np.full_like(t_mid, 1e6)

        try:
            result_midilli_a_fixé, log = multistart_optimization(
                lambda params: residuals_midilli_a_fixé(params, t_mid, MR_mid, a_fixé),
                bounds=([0.0001, 0.5, -0.01], [0.1, 2.0, 0.01]),
                n_starts=100
            )
            for entry in log:
                all_results.append({
                    'ID Fruit': fruit,
                    'Fruit_Code': fruit_code,
                    'Modèle': 'midilli_mixte_a_fixé',
                    **entry
                })
            k2_a_fixé, n2_a_fixé, b2_a_fixé = result_midilli_a_fixé
            MR_t1_target = midilli(t1, a_fixé, k2_a_fixé, n2_a_fixé, b2_a_fixé)
            pente_a_fixé = (MR_t1_target - 1) / (t1 - t0) if t1 > t0 else 0
            intercept_a_fixé = 1
            MR_pred_mixte_a_fixé = midilli_mixte_a_fixé(t, t0, t1, pente_a_fixé, intercept_a_fixé, k2_a_fixé, n2_a_fixé, b2_a_fixé, a_fixé)
            r2_mixte_a_fixé = r2_score(MR, MR_pred_mixte_a_fixé)
            rmse_mixte_a_fixé = np.sqrt(mean_squared_error(MR, MR_pred_mixte_a_fixé))
        except Exception as e:
            print(f"Erreur dans l’ajustement mixte avec a fixé pour {fruit} : {e}")
            k2_a_fixé, n2_a_fixé, b2_a_fixé = np.nan, np.nan, np.nan
            pente_a_fixé, intercept_a_fixé = np.nan, np.nan
            r2_mixte_a_fixé, rmse_mixte_a_fixé = np.nan, np.nan
            MR_pred_mixte_a_fixé = np.full_like(t, np.nan)
    else:
        print(f"Pas assez de points pour mixte avec a fixé sur {fruit}: t_mid={len(t_mid)}")
        k2_a_fixé, n2_a_fixé, b2_a_fixé = np.nan, np.nan, np.nan
        pente_a_fixé, intercept_a_fixé = np.nan, np.nan
        r2_mixte_a_fixé, rmse_mixte_a_fixé = np.nan, np.nan
        MR_pred_mixte_a_fixé = np.full_like(t, np.nan)

    # Résidus pour les modèles ajustés jusqu'à présent
    residus = {}
    try:
        residus = {
            'Henderson': MR - pred_dict.get('henderson', np.full_like(MR, np.nan)),
            'Aghbashlo': MR - pred_dict.get('aghbashlo', np.full_like(MR, np.nan)),
            'Midilli': MR - pred_dict.get('midilli', np.full_like(MR, np.nan)),
            'Mixte': MR - MR_pred_mixte,
            'Mixte_a_fixé': MR - MR_pred_mixte_a_fixé
        }
    except Exception as e:
        print(f"Erreur lors du calcul des résidus pour {fruit} : {e}")
        residus = {k: np.full_like(MR, np.nan) for k in ['Henderson', 'Aghbashlo', 'Midilli', 'Mixte', 'Mixte_a_fixé']}

    # Stockage des résultats
    resultats_parametres.append({
        'Fruit_Code': fruit_code,
        'ID Fruit': fruit,
        'X0': X0,
        'Xf': Xf,
        'a_henderson': params_dict.get('henderson', [np.nan, np.nan])[0],
        'k_henderson': params_dict.get('henderson', [np.nan, np.nan])[1],
        'k1_aghbashlo': params_dict.get('aghbashlo', [np.nan, np.nan])[0],
        'k2_aghbashlo': params_dict.get('aghbashlo', [np.nan, np.nan])[1],
        'a_midilli': params_dict.get('midilli', [np.nan, np.nan, np.nan, np.nan])[0],
        'k_midilli': params_dict.get('midilli', [np.nan, np.nan, np.nan, np.nan])[1],
        'n_midilli': params_dict.get('midilli', [np.nan, np.nan, np.nan, np.nan])[2],
        'b_midilli': params_dict.get('midilli', [np.nan, np.nan, np.nan, np.nan])[3],
        'a2_midilli_mixte': a2,
        'k2_midilli_mixte': k2,
        'n2_midilli_mixte': n2,
        'b2_midilli_mixte': b2,
        'a2_midilli_mixte_a_fixé': a_fixé,
        'k2_midilli_mixte_a_fixé': k2_a_fixé,
        'n2_midilli_mixte_a_fixé': n2_a_fixé,
        'b2_midilli_mixte_a_fixé': b2_a_fixé
    })

    resultats_qualite_fit.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        'R2_henderson': r2_dict.get('henderson', np.nan),
        'RMSE_henderson': rmse_dict.get('henderson', np.nan),
        'R2_aghbashlo': r2_dict.get('aghbashlo', np.nan),
        'RMSE_aghbashlo': rmse_dict.get('aghbashlo', np.nan),
        'R2_midilli': r2_dict.get('midilli', np.nan),
        'RMSE_midilli': rmse_dict.get('midilli', np.nan),
        'R2_mixte': r2_mixte,
        'RMSE_mixte': rmse_mixte,
        'R2_mixte_a_fixé': r2_mixte_a_fixé,
        'RMSE_mixte_a_fixé': rmse_mixte_a_fixé
    })

    droites_lineaires.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't0': t0,
        't1': t1,
        'Pente': pente,
        'Ordonnée': intercept,
        'Pente_a_fixé': pente_a_fixé,
        'Ordonnée_a_fixé': intercept_a_fixé
    })

    tous_r2_rmse_mixte.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        'R2_mixte': r2_mixte,
        'RMSE_mixte': rmse_mixte,
        'R2_mixte_a_fixé': r2_mixte_a_fixé,
        'RMSE_mixte_a_fixé': rmse_mixte_a_fixé
    })

    parametres_midilli_2.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        'a2': a2,
        'k2': k2,
        'n2': n2,
        'b2': b2,
        'a_fixé': a_fixé,
        'k2_a_fixé': k2_a_fixé,
        'n2_a_fixé': n2_a_fixé,
        'b2_a_fixé': b2_a_fixé
    })

    predictions_mixte.append(pd.DataFrame({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        'Temps [min]': t,
        'MR(t)_exp': MR,
        'MR(t)_mixte': MR_pred_mixte,
        'MR(t)_mixte_a_fixé': MR_pred_mixte_a_fixé
    }))

    # Calcul des t95
    try:
        if params_dict.get('henderson', [np.nan, np.nan])[1] > 0:
            t95_henderson = np.log(0.05) / (-params_dict['henderson'][1]) if params_dict['henderson'][0] > 0 else np.nan
        else:
            t95_henderson = np.nan
    except:
        t95_henderson = np.nan
    t95_henderson_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Henderson [min]': t95_henderson,
        'R2_henderson': r2_dict.get('henderson', np.nan)
    })

    try:
        ln_05 = np.log(0.05)
        if params_dict.get('aghbashlo', [np.nan, np.nan])[0] > 0:
            t95_aghbashlo = -ln_05 / (params_dict['aghbashlo'][0] + params_dict['aghbashlo'][1] * ln_05)
        else:
            t95_aghbashlo = np.nan
    except:
        t95_aghbashlo = np.nan
    t95_aghbashlo_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Aghbashlo [min]': t95_aghbashlo,
        'R2_aghbashlo': r2_dict.get('aghbashlo', np.nan)
    })

    try:
        def midilli_func(t, a, k, n, b):
            return a * np.exp(-k * t**n) + b * t - 0.05
        if not np.any(np.isnan(params_dict.get('midilli', [np.nan, np.nan, np.nan, np.nan]))):
            res = least_squares(
                lambda x: midilli_func(x, *params_dict['midilli']),
                x0=[max(t)],
                bounds=(1e-8, 1e4)
            )
            t95_midilli = res.x[0] if res.success and res.x[0] > 0 else np.nan
        else:
            t95_midilli = np.nan
    except:
        t95_midilli = np.nan
    t95_midilli_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Midilli [min]': t95_midilli,
        'R2_midilli': r2_dict.get('midilli', np.nan)
    })

    try:
        if not any(np.isnan([a2, k2, n2, b2])):
            func = lambda t_var: midilli(t_var, a2, k2, n2, b2) - 0.05
            res_mixte = least_squares(func, x0=[max(t_mid)], bounds=(t1, 1e4))
            t95_val = res_mixte.x[0] if res_mixte.success else np.nan
        else:
            t95_val = np.nan
    except:
        t95_val = np.nan
    t95_mixte_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Mixte [min]': t95_val,
        'R2_mixte': r2_mixte
    })

    try:
        if not any(np.isnan([a_fixé, k2_a_fixé, n2_a_fixé, b2_a_fixé])):
            func = lambda t_var: midilli(t_var, a_fixé, k2_a_fixé, n2_a_fixé, b2_a_fixé) - 0.05
            res_mixte_a_fixé = least_squares(func, x0=[max(t_mid)], bounds=(t1, 1e4))
            t95_val_a_fixé = res_mixte_a_fixé.x[0] if res_mixte_a_fixé.success else np.nan
        else:
            t95_val_a_fixé = np.nan
    except:
        t95_val_a_fixé = np.nan
    t95_mixte_a_fixé_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Mixte_a_fixé [min]': t95_val_a_fixé,
        'R2_mixte_a_fixé': r2_mixte_a_fixé
    })

    # Génération des figures
    t_line = np.linspace(min(t), max(t), 200)
    plt.figure(figsize=(8, 6))
    plt.scatter(t, MR, color='black', label='Expérimental')
    plt.plot(t_line, henderson_pabis(t_line, *params_dict.get('henderson', [np.nan, np.nan])), '-', color='blue', label='Henderson')
    plt.text(0.98, 0.3, f'R²={r2_dict.get("henderson", np.nan):.3f}\nRMSE={rmse_dict.get("henderson", np.nan):.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
    plt.xlabel('Temps [min]')
    plt.ylabel('MR(t)')
    plt.title(f'Henderson - {fruit}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_henderson_path, f'{fruit}_MR_Henderson.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(t, MR, color='black', label='Expérimental')
    plt.plot(t_line, aghbashlo(t_line, *params_dict.get('aghbashlo', [np.nan, np.nan])), '-', color='blue', label='Aghbashlo')
    plt.text(0.98, 0.3, f'R²={r2_dict.get("aghbashlo", np.nan):.3f}\nRMSE={rmse_dict.get("aghbashlo", np.nan):.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
    plt.xlabel('Temps [min]')
    plt.ylabel('MR(t)')
    plt.title(f'Aghbashlo - {fruit}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_aghbashlo_path, f'{fruit}_MR_Aghbashlo.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(t, MR, color='black', label='Expérimental')
    plt.plot(t_line, midilli(t_line, *params_dict.get('midilli', [np.nan, np.nan, np.nan, np.nan])), '-', color='blue', label='Midilli')
    plt.text(0.98, 0.3, f'R²={r2_dict.get("midilli", np.nan):.3f}\nRMSE={rmse_dict.get("midilli", np.nan):.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
    plt.xlabel('Temps [min]')
    plt.ylabel('MR(t)')
    plt.title(f'Midilli - {fruit}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(figures_midilli_path, f'{fruit}_MR_Midilli.png'), dpi=300)
    plt.close()

    if not np.any(np.isnan(MR_pred_mixte)):
        t_line_lin = np.linspace(t0, t1, 100)
        t_line_mid = t_line[t_line > t1]
        MR_line_mixte = modele_mixte(t_line, t0, t1, pente, intercept, a2, k2, n2, b2)
        plt.figure(figsize=(8, 6))
        plt.scatter(t, MR, color='black', label='Expérimental')
        plt.plot(t_line_lin, pente * t_line_lin + intercept, '-', color='red', label='Linéaire')
        plt.plot(t_line_mid, midilli(t_line_mid, a2, k2, n2, b2), '-', color='blue', label='Midilli')
        plt.axvline(t1, color='green', ls='--', label=f'Intersection={int(t1)} min')
        plt.text(0.98, 0.3, f'R²={r2_mixte:.3f}\nRMSE={rmse_mixte:.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        plt.xlabel('Temps [min]')
        plt.ylabel('MR(t)')
        plt.title(f'Mixte - {fruit}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(figures_mixte_path, f'{fruit}_MR_Mixte.png'), dpi=300)
        plt.close()

        # Sauvegarde figure des résidus mixte
        plt.figure(figsize=(8, 6))
        plt.scatter(t, residus["Mixte"], color='blue', label='Résidus Mixte')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Temps [min]')
        plt.ylabel('Résidus (MR_exp - MR_mixte)')
        plt.title(f'Résidus Mixte - {fruit}')
        plt.legend()
        plt.grid()
        save_path = os.path.join(figures_residus_mixte_path, fruit_type, f'Residus_Mixte_{fruit}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

    if not np.any(np.isnan(MR_pred_mixte_a_fixé)):
        t_line_lin = np.linspace(t0, t1, 100)
        t_line_mid = t_line[t_line > t1]
        MR_line_mixte_a_fixé = midilli_mixte_a_fixé(t_line, t0, t1, pente_a_fixé, intercept_a_fixé, k2_a_fixé, n2_a_fixé, b2_a_fixé, a_fixé)
        plt.figure(figsize=(8, 6))
        plt.scatter(t, MR, color='black', label='Expérimental')
        plt.plot(t_line_lin, pente_a_fixé * t_line_lin + intercept_a_fixé, '-', color='red', label='Linéaire')
        plt.plot(t_line_mid, midilli(t_line_mid, a_fixé, k2_a_fixé, n2_a_fixé, b2_a_fixé), '-', color='blue', label='Midilli a fixé')
        plt.axvline(t1, color='green', ls='--', label=f'Intersection={int(t1)} min')
        plt.text(0.98, 0.3, f'R²={r2_mixte_a_fixé:.3f}\nRMSE={rmse_mixte_a_fixé:.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        plt.xlabel('Temps [min]')
        plt.ylabel('MR(t)')
        plt.title(f'Mixte a fixé - {fruit}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(figures_mixte_a_fixé_path, f'{fruit}_MR_Mixte_a_fixé.png'), dpi=300)
        plt.close()

        # Sauvegarde figure des résidus mixte avec a fixé
        plt.figure(figsize=(8, 6))
        plt.scatter(t, residus["Mixte_a_fixé"], color='purple', label='Résidus Mixte a fixé')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Temps [min]')
        plt.ylabel('Résidus (MR_exp - MR_mixte_a_fixé)')
        plt.title(f'Résidus Mixte a fixé - {fruit}')
        plt.legend()
        plt.grid()
        save_path = os.path.join(figures_residus_mixte_a_fixé_path, fruit_type, f'Residus_Mixte_a_fixé_{fruit}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

# =============================================================================
# Calcul des médianes pour b_fixe_values
# =============================================================================
df_parametres_midilli_2 = pd.DataFrame(parametres_midilli_2)
median_values = df_parametres_midilli_2.groupby('Fruit_Code')[['b2_a_fixé']].median().reset_index()
for _, row in median_values.iterrows():
    fruit_code = str(row['Fruit_Code'])
    b_fixe_values[fruit_code] = row['b2_a_fixé'] if not pd.isna(row['b2_a_fixé']) else 0.0

# Affichage des valeurs médianes pour vérification
print("Valeurs médianes pour b_fixe_values:", b_fixe_values)

# =============================================================================
# Deuxième passe : Ajustement du modèle midilli_mixte_a_b_fixés
# =============================================================================
for fruit in fruit_ids:
    subset = df_clean[df_clean['ID Fruit'] == fruit]
    t = subset['Temps [min]'].values
    MR = subset['MR(t)'].values

    if len(t) < 5:
        print(f"Échantillon {fruit} ignoré (nombre insuffisant de points)")
        continue

    t_sorted = np.sort(t)
    t0 = t_sorted[0]
    fruit_code = str(subset['Fruit_Code'].iloc[0])
    t1 = t1_fixe.get(int(fruit_code), 10)
    a_fixé = a_fixé_values.get(fruit_code, 1.0)
    b_fixé = b_fixe_values.get(fruit_code, 0.0)
    X0, Xf = subset['X0'].iloc[0], subset['Xf'].iloc[0]

    # Ajustement mixte avec a et b fixés - Phase après t1
    t_mid = t[t > t1]
    MR_mid = MR[t > t1]

    if len(t_mid) >= 2:
        def residuals_midilli_a_b_fixés(params, t_mid, MR_mid, a_fixé, b_fixé):
            k2, n2 = params
            try:
                MR_pred = midilli(t_mid, a_fixé, k2, n2, b_fixé)
                return MR_mid - MR_pred
            except:
                return np.full_like(t_mid, 1e6)

        try:
            result_midilli_a_b_fixés, log = multistart_optimization(
                lambda params: residuals_midilli_a_b_fixés(params, t_mid, MR_mid, a_fixé, b_fixé),
                bounds=([0.0001, 0.5], [0.1, 2.0]),
                n_starts=100
            )
            for entry in log:
                all_results.append({
                    'ID Fruit': fruit,
                    'Fruit_Code': fruit_code,
                    'Modèle': 'midilli_mixte_a_b_fixés',
                    **entry
                })
            k2_a_b_fixés, n2_a_b_fixés = result_midilli_a_b_fixés
            MR_t1_target = midilli(t1, a_fixé, k2_a_b_fixés, n2_a_b_fixés, b_fixé)
            pente_a_b_fixés = (MR_t1_target - 1) / (t1 - t0) if t1 > t0 else 0
            intercept_a_b_fixés = 1
            MR_pred_mixte_a_b_fixés = midilli_mixte_a_b_fixés(t, t0, t1, pente_a_b_fixés, intercept_a_b_fixés, k2_a_b_fixés, n2_a_b_fixés, a_fixé, b_fixé)
            r2_mixte_a_b_fixés = r2_score(MR, MR_pred_mixte_a_b_fixés)
            rmse_mixte_a_b_fixés = np.sqrt(mean_squared_error(MR, MR_pred_mixte_a_b_fixés))
        except Exception as e:
            print(f"Erreur dans l’ajustement mixte avec a, b fixés pour {fruit} : {e}")
            k2_a_b_fixés, n2_a_b_fixés = np.nan, np.nan
            pente_a_b_fixés, intercept_a_b_fixés = np.nan, np.nan
            r2_mixte_a_b_fixés, rmse_mixte_a_b_fixés = np.nan, np.nan
            MR_pred_mixte_a_b_fixés = np.full_like(t, np.nan)
    else:
        print(f"Pas assez de points pour mixte avec a, b fixés sur {fruit}: t_mid={len(t_mid)}")
        k2_a_b_fixés, n2_a_b_fixés = np.nan, np.nan
        pente_a_b_fixés, intercept_a_b_fixés = np.nan, np.nan
        r2_mixte_a_b_fixés, rmse_mixte_a_b_fixés = np.nan, np.nan
        MR_pred_mixte_a_b_fixés = np.full_like(t, np.nan)

    # Mise à jour des résidus
    residus['Mixte_a_b_fixés'] = MR - MR_pred_mixte_a_b_fixés
    for idx, res in enumerate(analyse_residus):
        if res['ID Fruit'] == fruit:
            analyse_residus[idx]['Residus_Mixte_a_b_fixés'] = residus['Mixte_a_b_fixés'].tolist()
            break

    # Mise à jour des paramètres
    for idx, param in enumerate(resultats_parametres):
        if param['ID Fruit'] == fruit:
            resultats_parametres[idx].update({
                'a2_midilli_mixte_a_b_fixés': a_fixé,
                'k2_midilli_mixte_a_b_fixés': k2_a_b_fixés,
                'n2_midilli_mixte_a_b_fixés': n2_a_b_fixés,
                'b2_midilli_mixte_a_b_fixés': b_fixé
            })
            break

    # Mise à jour de la qualité de l'ajustement
    for idx, qual in enumerate(resultats_qualite_fit):
        if qual['ID Fruit'] == fruit:
            resultats_qualite_fit[idx].update({
                'R2_mixte_a_b_fixés': r2_mixte_a_b_fixés,
                'RMSE_mixte_a_b_fixés': rmse_mixte_a_b_fixés
            })
            break

    # Mise à jour des droites linéaires
    for idx, droite in enumerate(droites_lineaires):
        if droite['ID Fruit'] == fruit:
            droites_lineaires[idx].update({
                'Pente_a_b_fixés': pente_a_b_fixés,
                'Ordonnée_a_b_fixés': intercept_a_b_fixés
            })
            break

    # Mise à jour de tous_r2_rmse_mixte
    for idx, r2_rmse in enumerate(tous_r2_rmse_mixte):
        if r2_rmse['ID Fruit'] == fruit:
            tous_r2_rmse_mixte[idx].update({
                'R2_mixte_a_b_fixés': r2_mixte_a_b_fixés,
                'RMSE_mixte_a_b_fixés': rmse_mixte_a_b_fixés
            })
            break

    # Mise à jour des paramètres Midilli
    for idx, param_mid in enumerate(parametres_midilli_2):
        if param_mid['ID Fruit'] == fruit:
            parametres_midilli_2[idx].update({
                'k2_a_b_fixés': k2_a_b_fixés,
                'n2_a_b_fixés': n2_a_b_fixés,
                'a2_a_b_fixés': a_fixé,
                'b2_a_b_fixés': b_fixé
            })
            break

    # Mise à jour des prédictions
    for idx, pred in enumerate(predictions_mixte):
        if pred['ID Fruit'].iloc[0] == fruit:
            predictions_mixte[idx]['MR(t)_mixte_a_b_fixés'] = MR_pred_mixte_a_b_fixés
            break

    # Calcul de t95 pour mixte a, b fixés
    try:
        if not any(np.isnan([a_fixé, k2_a_b_fixés, n2_a_b_fixés, b_fixé])):
            func = lambda t_var: midilli(t_var, a_fixé, k2_a_b_fixés, n2_a_b_fixés, b_fixé) - 0.05
            res_mixte_a_b_fixés = least_squares(func, x0=[max(t_mid)], bounds=(t1, 1e4))
            t95_val_a_b_fixés = res_mixte_a_b_fixés.x[0] if res_mixte_a_b_fixés.success else np.nan
        else:
            t95_val_a_b_fixés = np.nan
    except:
        t95_val_a_b_fixés = np.nan
    t95_mixte_a_b_fixés_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Mixte_a_b_fixés [min]': t95_val_a_b_fixés,
        'R2_mixte_a_b_fixés': r2_mixte_a_b_fixés
    })

    # Génération de la figure pour mixte a, b fixés
    if not np.any(np.isnan(MR_pred_mixte_a_b_fixés)):
        t_line = np.linspace(min(t), max(t), 200)
        t_line_lin = np.linspace(t0, t1, 100)
        t_line_mid = t_line[t_line > t1]
        MR_line_mixte_a_b_fixés = midilli_mixte_a_b_fixés(t_line, t0, t1, pente_a_b_fixés, intercept_a_b_fixés, k2_a_b_fixés, n2_a_b_fixés, a_fixé, b_fixé)
        plt.figure(figsize=(8, 6))
        plt.scatter(t, MR, color='black', label='Expérimental')
        plt.plot(t_line_lin, pente_a_b_fixés * t_line_lin + intercept_a_b_fixés, '-', color='red', label='Linéaire')
        plt.plot(t_line_mid, midilli(t_line_mid, a_fixé, k2_a_b_fixés, n2_a_b_fixés, b_fixé), '-', color='blue', label='Midilli a, b fixés')
        plt.axvline(t1, color='green', ls='--', label=f'Intersection={int(t1)} min')
        plt.text(0.98, 0.3, f'R²={r2_mixte_a_b_fixés:.3f}\nRMSE={rmse_mixte_a_b_fixés:.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        plt.xlabel('Temps [min]')
        plt.ylabel('MR(t)')
        plt.title(f'Mixte a, b fixés - {fruit}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(figures_mixte_a_b_fixés_path, f'{fruit}_MR_Mixte_a_b_fixés.png'), dpi=300)
        plt.close()

        # Sauvegarde figure des résidus mixte avec a, b fixés
        plt.figure(figsize=(8, 6))
        plt.scatter(t, residus["Mixte_a_b_fixés"], color='orange', label='Résidus Mixte a, b fixés')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Temps [min]')
        plt.ylabel('Résidus (MR_exp - MR_mixte_a_b_fixés)')
        plt.title(f'Résidus Mixte a, b fixés - {fruit}')
        plt.legend()
        plt.grid()
        save_path = os.path.join(figures_residus_mixte_a_b_fixés_path, fruit_type, f'Residus_Mixte_a_b_fixés_{fruit}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

# =============================================================================
# Calcul des médianes pour n_fixe_values
# =============================================================================
df_parametres_midilli_2 = pd.DataFrame(parametres_midilli_2)
median_values = df_parametres_midilli_2.groupby('Fruit_Code')[['n2_a_b_fixés']].median().reset_index()
for _, row in median_values.iterrows():
    fruit_code = str(row['Fruit_Code'])
    n_fixe_values[fruit_code] = row['n2_a_b_fixés'] if not pd.isna(row['n2_a_b_fixés']) else 1.0

# Affichage des valeurs médianes pour vérification
print("Valeurs médianes pour n_fixe_values:", n_fixe_values)

# =============================================================================
# Troisième passe : Ajustement du modèle midilli_mixte_a_n_b_fixés
# =============================================================================
for fruit in fruit_ids:
    subset = df_clean[df_clean['ID Fruit'] == fruit]
    t = subset['Temps [min]'].values
    MR = subset['MR(t)'].values

    if len(t) < 5:
        print(f"Échantillon {fruit} ignoré (nombre insuffisant de points)")
        continue

    t_sorted = np.sort(t)
    t0 = t_sorted[0]
    fruit_code = str(subset['Fruit_Code'].iloc[0])
    t1 = t1_fixe.get(int(fruit_code), 10)
    a_fixé = a_fixé_values.get(fruit_code, 1.0)
    n_fixé = n_fixe_values.get(fruit_code, 1.0)
    b_fixé = b_fixe_values.get(fruit_code, 0.0)
    X0, Xf = subset['X0'].iloc[0], subset['Xf'].iloc[0]

    # Ajustement mixte avec a, n, b fixés - Phase après t1
    t_mid = t[t > t1]
    MR_mid = MR[t > t1]

    if len(t_mid) >= 2:
        def residuals_midilli_a_n_b_fixés(params, t_mid, MR_mid, a_fixé, n_fixé, b_fixé):
            k2 = params[0]
            try:
                MR_pred = midilli(t_mid, a_fixé, k2, n_fixé, b_fixé)
                return MR_mid - MR_pred
            except:
                return np.full_like(t_mid, 1e6)

        try:
            result_midilli_a_n_b_fixés, log = multistart_optimization(
                lambda params: residuals_midilli_a_n_b_fixés(params, t_mid, MR_mid, a_fixé, n_fixé, b_fixé),
                bounds=([0.0001], [0.1]),
                n_starts=100
            )
            for entry in log:
                all_results.append({
                    'ID Fruit': fruit,
                    'Fruit_Code': fruit_code,
                    'Modèle': 'midilli_mixte_a_n_b_fixés',
                    **entry
                })
            k2_a_n_b_fixés = result_midilli_a_n_b_fixés[0]
            MR_t1_target = midilli(t1, a_fixé, k2_a_n_b_fixés, n_fixé, b_fixé)
            pente_a_n_b_fixés = (MR_t1_target - 1) / (t1 - t0) if t1 > t0 else 0
            intercept_a_n_b_fixés = 1
            MR_pred_mixte_a_n_b_fixés = midilli_mixte_a_n_b_fixés(t, t0, t1, pente_a_n_b_fixés, intercept_a_n_b_fixés, k2_a_n_b_fixés, a_fixé, n_fixé, b_fixé)
            r2_mixte_a_n_b_fixés = r2_score(MR, MR_pred_mixte_a_n_b_fixés)
            rmse_mixte_a_n_b_fixés = np.sqrt(mean_squared_error(MR, MR_pred_mixte_a_n_b_fixés))
        except Exception as e:
            print(f"Erreur dans l’ajustement mixte avec a, n, b fixés pour {fruit} : {e}")
            k2_a_n_b_fixés = np.nan
            pente_a_n_b_fixés, intercept_a_n_b_fixés = np.nan, np.nan
            r2_mixte_a_n_b_fixés, rmse_mixte_a_n_b_fixés = np.nan, np.nan
            MR_pred_mixte_a_n_b_fixés = np.full_like(t, np.nan)
    else:
        print(f"Pas assez de points pour mixte avec a, n, b fixés sur {fruit}: t_mid={len(t_mid)}")
        k2_a_n_b_fixés = np.nan
        pente_a_n_b_fixés, intercept_a_n_b_fixés = np.nan, np.nan
        r2_mixte_a_n_b_fixés, rmse_mixte_a_n_b_fixés = np.nan, np.nan
        MR_pred_mixte_a_n_b_fixés = np.full_like(t, np.nan)

    # Mise à jour des résidus
    residus['Mixte_a_n_b_fixés'] = MR - MR_pred_mixte_a_n_b_fixés
    for idx, res in enumerate(analyse_residus):
        if res['ID Fruit'] == fruit:
            analyse_residus[idx]['Residus_Mixte_a_n_b_fixés'] = residus['Mixte_a_n_b_fixés'].tolist()
            break

    # Mise à jour des paramètres
    for idx, param in enumerate(resultats_parametres):
        if param['ID Fruit'] == fruit:
            resultats_parametres[idx].update({
                'a2_midilli_mixte_a_n_b_fixés': a_fixé,
                'k2_midilli_mixte_a_n_b_fixés': k2_a_n_b_fixés,
                'n2_midilli_mixte_a_n_b_fixés': n_fixé,
                'b2_midilli_mixte_a_n_b_fixés': b_fixé
            })
            break

    # Mise à jour de la qualité de l'ajustement
    for idx, qual in enumerate(resultats_qualite_fit):
        if qual['ID Fruit'] == fruit:
            resultats_qualite_fit[idx].update({
                'R2_mixte_a_n_b_fixés': r2_mixte_a_n_b_fixés,
                'RMSE_mixte_a_n_b_fixés': rmse_mixte_a_n_b_fixés
            })
            break

    # Mise à jour des droites linéaires
    for idx, droite in enumerate(droites_lineaires):
        if droite['ID Fruit'] == fruit:
            droites_lineaires[idx].update({
                'Pente_a_n_b_fixés': pente_a_n_b_fixés,
                'Ordonnée_a_n_b_fixés': intercept_a_n_b_fixés
            })
            break

    # Mise à jour de tous_r2_rmse_mixte
    for idx, r2_rmse in enumerate(tous_r2_rmse_mixte):
        if r2_rmse['ID Fruit'] == fruit:
            tous_r2_rmse_mixte[idx].update({
                'R2_mixte_a_n_b_fixés': r2_mixte_a_n_b_fixés,
                'RMSE_mixte_a_n_b_fixés': rmse_mixte_a_n_b_fixés
            })
            break

    # Mise à jour des paramètres Midilli
    for idx, param_mid in enumerate(parametres_midilli_2):
        if param_mid['ID Fruit'] == fruit:
            parametres_midilli_2[idx].update({
                'k2_a_n_b_fixés': k2_a_n_b_fixés,
                'a2_a_n_b_fixés': a_fixé,
                'n2_a_n_b_fixés': n_fixé,
                'b2_a_n_b_fixés': b_fixé
            })
            break

    # Mise à jour des prédictions
    for idx, pred in enumerate(predictions_mixte):
        if pred['ID Fruit'].iloc[0] == fruit:
            predictions_mixte[idx]['MR(t)_mixte_a_n_b_fixés'] = MR_pred_mixte_a_n_b_fixés
            break

    # Calcul de t95 pour mixte a, n, b fixés
    try:
        if not any(np.isnan([a_fixé, k2_a_n_b_fixés, n_fixé, b_fixé])):
            func = lambda t_var: midilli(t_var, a_fixé, k2_a_n_b_fixés, n_fixé, b_fixé) - 0.05
            res_mixte_a_n_b_fixés = least_squares(func, x0=[max(t_mid)], bounds=(t1, 1e4))
            t95_val_a_n_b_fixés = res_mixte_a_n_b_fixés.x[0] if res_mixte_a_n_b_fixés.success else np.nan
        else:
            t95_val_a_n_b_fixés = np.nan
    except:
        t95_val_a_n_b_fixés = np.nan
    t95_mixte_a_n_b_fixés_vals.append({
        'ID Fruit': fruit,
        'Fruit_Code': fruit_code,
        't95_Mixte_a_n_b_fixés [min]': t95_val_a_n_b_fixés,
        'R2_mixte_a_n_b_fixés': r2_mixte_a_n_b_fixés
    })

    # Génération de la figure pour mixte a, n, b fixés
    if not np.any(np.isnan(MR_pred_mixte_a_n_b_fixés)):
        t_line = np.linspace(min(t), max(t), 200)
        t_line_lin = np.linspace(t0, t1, 100)
        t_line_mid = t_line[t_line > t1]
        MR_line_mixte_a_n_b_fixés = midilli_mixte_a_n_b_fixés(t_line, t0, t1, pente_a_n_b_fixés, intercept_a_n_b_fixés, k2_a_n_b_fixés, a_fixé, n_fixé, b_fixé)
        plt.figure(figsize=(8, 6))
        plt.scatter(t, MR, color='black', label='Expérimental')
        plt.plot(t_line_lin, pente_a_n_b_fixés * t_line_lin + intercept_a_n_b_fixés, '-', color='red', label='Linéaire')
        plt.plot(t_line_mid, midilli(t_line_mid, a_fixé, k2_a_n_b_fixés, n_fixé, b_fixé), '-', color='blue', label='Midilli a, n, b fixés')
        plt.axvline(t1, color='green', ls='--', label=f'Intersection={int(t1)} min')
        plt.text(0.98, 0.3, f'R²={r2_mixte_a_n_b_fixés:.3f}\nRMSE={rmse_mixte_a_n_b_fixés:.4f}', transform=plt.gca().transAxes, ha='right', va='bottom')
        plt.xlabel('Temps [min]')
        plt.ylabel('MR(t)')
        plt.title(f'Mixte a, n, b fixés - {fruit}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(figures_mixte_a_n_b_fixés_path, f'{fruit}_MR_Mixte_a_n_b_fixés.png'), dpi=300)
        plt.close()

        # Sauvegarde figure des résidus mixte avec a, n, b fixés
        plt.figure(figsize=(8, 6))
        plt.scatter(t, residus["Mixte_a_n_b_fixés"], color='green', label='Résidus Mixte a, n, b fixés')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Temps [min]')
        plt.ylabel('Résidus (MR_exp - MR_mixte_a_n_b_fixés)')
        plt.title(f'Résidus Mixte a, n, b fixés - {fruit}')
        plt.legend()
        plt.grid()
        save_path = os.path.join(figures_residus_mixte_a_n_b_fixés_path, fruit_type, f'Residus_Mixte_a_n_b_fixés_{fruit}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()

# =============================================================================
# Exportation des résultats
# =============================================================================
with pd.ExcelWriter(os.path.join(export_path, 'MR_Analyse_Complète.xlsx')) as writer:
    pd.DataFrame(resultats_parametres).to_excel(writer, sheet_name='Paramètres', index=False)
    pd.DataFrame(resultats_qualite_fit).to_excel(writer, sheet_name='Qualité_ajustement', index=False)
    pd.DataFrame(analyse_residus).to_excel(writer, sheet_name='Résidus', index=False)
    pd.DataFrame(droites_lineaires).to_excel(writer, sheet_name='Droites_linéaires', index=False)
    pd.DataFrame(tous_r2_rmse_mixte).to_excel(writer, sheet_name='R2_RMSE_Mixte', index=False)
    pd.DataFrame(parametres_midilli_2).to_excel(writer, sheet_name='Paramètres_midilli_2', index=False)
    pd.concat(predictions_mixte).to_excel(writer, sheet_name='Valeurs_Ajustées', index=False)
    pd.DataFrame(t95_henderson_vals).to_excel(writer, sheet_name='t95_Henderson', index=False)
    pd.DataFrame(t95_aghbashlo_vals).to_excel(writer, sheet_name='t95_Aghbashlo', index=False)
    pd.DataFrame(t95_midilli_vals).to_excel(writer, sheet_name='t95_Midilli', index=False)
    pd.DataFrame(t95_mixte_vals).to_excel(writer, sheet_name='t95_Mixte', index=False)
    pd.DataFrame(t95_mixte_a_fixé_vals).to_excel(writer, sheet_name='t95_Mixte_a_fixé', index=False)
    pd.DataFrame(t95_mixte_a_b_fixés_vals).to_excel(writer, sheet_name='t95_Mixte_a_b_fixés', index=False)
    pd.DataFrame(t95_mixte_a_n_b_fixés_vals).to_excel(writer, sheet_name='t95_Mixte_a_n_b_fixés', index=False)
    
    # Exportation des résumés par modèle
    resume_henderson_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_henderson', 'RMSE_henderson']].rename(columns={'R2_henderson': 'R2', 'RMSE_henderson': 'RMSE'})
    resume_aghbashlo_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_aghbashlo', 'RMSE_aghbashlo']].rename(columns={'R2_aghbashlo': 'R2', 'RMSE_aghbashlo': 'RMSE'})
    resume_midilli_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_midilli', 'RMSE_midilli']].rename(columns={'R2_midilli': 'R2', 'RMSE_midilli': 'RMSE'})
    resume_mixte_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_mixte', 'RMSE_mixte']].rename(columns={'R2_mixte': 'R2', 'RMSE_mixte': 'RMSE'})
    resume_mixte_a_fixé_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_mixte_a_fixé', 'RMSE_mixte_a_fixé']].rename(columns={'R2_mixte_a_fixé': 'R2', 'RMSE_mixte_a_fixé': 'RMSE'})
    resume_mixte_a_b_fixés_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_mixte_a_b_fixés', 'RMSE_mixte_a_b_fixés']].rename(columns={'R2_mixte_a_b_fixés': 'R2', 'RMSE_mixte_a_b_fixés': 'RMSE'})
    resume_mixte_a_n_b_fixés_final = pd.DataFrame(resultats_qualite_fit)[['ID Fruit', 'Fruit_Code', 'R2_mixte_a_n_b_fixés', 'RMSE_mixte_a_n_b_fixés']].rename(columns={'R2_mixte_a_n_b_fixés': 'R2', 'RMSE_mixte_a_n_b_fixés': 'RMSE'})

    resume_henderson_final.to_excel(writer, sheet_name='Résumé_Henderson', index=False)
    resume_aghbashlo_final.to_excel(writer, sheet_name='Résumé_Aghbashlo', index=False)
    resume_midilli_final.to_excel(writer, sheet_name='Résumé_Midilli', index=False)
    resume_mixte_final.to_excel(writer, sheet_name='Résumé_Mixte', index=False)
    resume_mixte_a_fixé_final.to_excel(writer, sheet_name='Résumé_Mixte_a_fixé', index=False)
    resume_mixte_a_b_fixés_final.to_excel(writer, sheet_name='Résumé_Mixte_a_b_fixés', index=False)
    resume_mixte_a_n_b_fixés_final.to_excel(writer, sheet_name='Résumé_Mixte_a_n_b_fixés', index=False)

    # Exportation des résultats multistart par modèle
    modeles = ['henderson', 'aghbashlo', 'midilli', 'midilli_mixte', 'midilli_mixte_a_fixé', 'midilli_mixte_a_b_fixés', 'midilli_mixte_a_n_b_fixés']
    df_multistart = pd.DataFrame(all_results)
    for modele in modeles:
        df_modele = df_multistart[df_multistart['Modèle'] == modele]
        if not df_modele.empty:
            sheet_name = f"Multistart_{modele.replace('_', '')}"[:31]
            df_modele.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Résultats multistart pour {modele} exportés dans la feuille {sheet_name}")
        else:
            print(f"Aucun résultat multistart trouvé pour {modele}")

print(f"Fichier Excel sauvegardé dans : {os.path.join(export_path, 'MR_Analyse_Complète.xlsx')}")

