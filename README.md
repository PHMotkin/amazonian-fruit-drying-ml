# amazonian-fruit-drying-ml
Drying mathematical modelling and XGBoost predictions for Amazonian fruits (açaí, guaraná, black pepper, cocoa): _MR_(t) fitting, prediction of final and dry mass, kinetic constant k, and initial spherical-fruit dimensions.

**Project Summary**

This repository contains the code used in my master’s thesis (ULB, 2024–2025) on mathematical modelling and prediction of drying profiles of Amazonian fruits (acai, guarana—angulate and spherical—black pepper, cocoa). This work is part of an academic collaboration between **Université Libre de Bruxelles (ULB) — Transfers, Interfaces & Processes (TIPs) —** and **Universidade Federal do Pará (UFPA)**.

**Thesis title:** _Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens._

**How to cite:** Motkin, P.-H. (2025). _Mise en place d'outils statistiques et de machine learning visant à la compréhension du séchage de fruits amazoniens._ Master's thesis - Université Libre de Bruxelles.

**This repository includes:**

•	Parametric **fitting of _MR_(t)** with **Henderson & Pabis**, **Midilli**, and **an innovative piecewise mixed model** created by the author (this part contains a rigorous approach with multi-start strategy and homoscedasticity assumption),

•	**XGBoost models** to predict **initial dimensions of spherical fruits** (_a_, _b_, _c_), **dry mass** (_MS_) and **final mass** (_Mf_) from initial features, and the **kinetic drying constant _k_** of the mixed model created,

•	**Rigorous evaluation:** **stratification by type of fruit** (to deal with heterogeneity of database), **Repeated Stratified k-Fold CV**, **Optuna** hyperparameter optimization, **95% bootstrap CIs**, and structured Excel exports with figures.
