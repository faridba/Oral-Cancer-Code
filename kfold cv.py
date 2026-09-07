#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     LeaveOneOut, GridSearchCV, train_test_split, learning_curve)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score, jaccard_score, classification_report,
                             brier_score_loss)
from sklearn.calibration import calibration_curve
from imblearn.metrics import classification_report_imbalanced

# Feature Selection Imports
from mrmr import mrmr_classif
from sklearn.feature_selection import RFE

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

# Configuration
warnings.filterwarnings('ignore')
os.environ["SCIPY_ARRAY_API"] = "1"

# Define Classifiers and Param Grids
CLASSIFIERS = {
    'SVM_linear': (SVC(probability=True, kernel='linear', class_weight='balanced'),
                   {'C': [0.001, 0.01, 0.1, 0.5, 1, 2, 5]}),
    'SVM_rbf': (SVC(probability=True, kernel='rbf', class_weight='balanced'),
                {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 0.5, 1, 2, 5]}),
    'RandomForest': (RandomForestClassifier(class_weight='balanced'),
                     {'n_estimators': [50, 100, 200]}),
    'LogisticRegression': (LogisticRegression(max_iter=10000, class_weight='balanced'),
                           {'C': [0.1, 1, 10]}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    'GradientBoosting': (GradientBoostingClassifier(),
                         {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [100, 200, 500]}),
    'ExtraTrees': (ExtraTreesClassifier(class_weight='balanced'),
                   {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    'LinearDiscriminantAnalysis': (LinearDiscriminantAnalysis(), {}),
    'XGBoost': (XGBClassifier(eval_metric='logloss'),
                {'n_estimators': [50, 100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2]})
}


def ensure_dir(path):
    """Helper to create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compare_models_stats(scores_model_A, scores_model_B, n_bootstrap=10000):
    scores_A = np.array(scores_model_A)
    scores_B = np.array(scores_model_B)

    if len(scores_A) != len(scores_B):
        return {'p_val_t': np.nan, 'p_val_bootstrap': np.nan, 'diff_mean': np.nan}

    t_stat, p_val_t = stats.ttest_rel(scores_A, scores_B)

    obs_diff = np.mean(scores_A) - np.mean(scores_B)
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(scores_A), size=len(scores_A), replace=True)
        diff = np.mean(scores_A[indices]) - np.mean(scores_B[indices])
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)
    p_val_boot = np.mean(bootstrap_diffs <= 0)

    return {
        't_stat': t_stat,
        'p_val_t': p_val_t,
        'p_val_bootstrap': p_val_boot,
        'diff_mean': obs_diff
    }


def get_metrics(y_true, y_pred, y_proba, num_classes=2):
    """Helper to calculate metrics."""
    cri = classification_report_imbalanced(y_true, y_pred, output_dict=True)

    if num_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba.shape[1] > 1 else 0.5
        pos_probs = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
        brier = brier_score_loss(y_true, pos_probs)
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
        brier = np.nan

    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': cri['avg_f1'],
        'precision': cri['avg_pre'],
        'recall': cri['avg_rec'],
        'specificity': cri['avg_spe'],
        'AUC': auc,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'jaccard': jaccard_score(y_true, y_pred, average='weighted'),
        'brier': brier  # <-- ADD THIS
    }


def plot_calibration_curve_custom(y_true, y_proba, title, save_path):
    """Generates and saves a calibration curve/reliability diagram."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='s', color='blue', label='Model')

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_learning_curve(estimator, title, X, y, save_path, cv=5):
    """
    Generates and saves a learning curve plot to demonstrate lack of overfitting.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='roc_auc'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("ROC AUC Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(save_path)
    plt.close()


def perform_feature_selection(X, y, method, n_features, save_path):
    """
    Selects features using mRMR or SVM-RFE and saves the list.
    """
    if os.path.exists(save_path):
        return list(np.load(save_path, allow_pickle=True))

    selected_features = []

    if method == 'mrmr':
        selected_features = mrmr_classif(X=X, y=y, K=n_features)

    elif method == 'SVM-RFE':
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_].tolist()

    print(f"    Saving {len(selected_features)} features ({method}) to disk...")
    np.save(save_path, selected_features)

    return selected_features


def run_experiment_cv(X, qmics, y, path_save, num_feat, feat_type, mode='late'):
    """
    Unified function to run either Late Fusion or Early Fusion CV.
    mode: 'late' or 'early'
    """
    num_classes = len(np.unique(y))
    k_fold_num = 5
    results_stats = []

    mode_dir = 'Late_Fusion' if mode == 'late' else 'Early_Fusion'

    path_fs = os.path.join(path_save, 'Feature_Selection_Results', mode_dir, feat_type, str(num_feat))
    ensure_dir(path_fs)
    idx_path = os.path.join(path_save, 'Index_train_test')
    ensure_dir(idx_path)
    model_path = os.path.join(path_save, 'Models', mode_dir)
    ensure_dir(model_path)
    ensure_dir(os.path.join(path_save, 'Learning_Curves', mode_dir))
    ensure_dir(os.path.join(path_save, 'Calibration_Curves', mode_dir))

    raw_fold_scores = {name: [] for name in CLASSIFIERS.keys()}

    for name, (classifier, params) in CLASSIFIERS.items():
        print(f"  [{mode.upper()}] Classifier: {name}")

        fold_metrics = {'AUC': [], 'acc': [], 'f1': [], 'mcc': [], 'precision': [],
                        'recall': [], 'specificity': [], 'kappa': [], 'jaccard': [],
                        'brier': []}
        all_y_true_fold = []
        all_y_proba_fold = []

        if k_fold_num > 1:
            kf = RepeatedStratifiedKFold(n_splits=k_fold_num, n_repeats=5, random_state=42)
        else:
            kf = LeaveOneOut()

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            f_train = os.path.join(idx_path, f'train_indx_{fold_num + 1}.npy')
            f_test = os.path.join(idx_path, f'test_indx_{fold_num + 1}.npy')
            f_train_val = os.path.join(idx_path, f'train_val_indx_{fold_num + 1}.npy')
            f_val = os.path.join(idx_path, f'val_indx_{fold_num + 1}.npy')

            if os.path.isfile(f_train):
                train_idx, test_idx = np.load(f_train), np.load(f_test)
                X_train_full, X_test = X.loc[train_idx], X.loc[test_idx]
                y_train_full, y_test = y.loc[train_idx], y.loc[test_idx].values.ravel()
                qmics_train_full, qmics_test = qmics.loc[train_idx], qmics.loc[test_idx]
            else:
                X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx].values.ravel()
                qmics_train_full, qmics_test = qmics.iloc[train_idx], qmics.iloc[test_idx]
                np.save(f_train, list(X_train_full.index))
                np.save(f_test, list(X_test.index))

            # ==============================================================================
            # EARLY FUSION LOGIC
            # ==============================================================================
            if mode == 'early':
                path_ef_model = os.path.join(model_path, feat_type, name, str(num_feat))
                ensure_dir(path_ef_model)

                imp = SimpleImputer(strategy='median')
                X_train_imp = pd.DataFrame(imp.fit_transform(X_train_full), columns=X_train_full.columns,
                                           index=X_train_full.index)
                X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

                scaler_gene = StandardScaler()
                X_train_scl = pd.DataFrame(scaler_gene.fit_transform(X_train_imp), columns=X_train_full.columns,
                                           index=X_train_full.index)
                X_test_scl = pd.DataFrame(scaler_gene.transform(X_test_imp), columns=X_test.columns, index=X_test.index)

                fs_filename = f'selected_features_fold_num_{fold_num}.npy'
                fs_file_path = os.path.join(path_fs, fs_filename)
                list_best_feat = perform_feature_selection(X_train_scl, y_train_full, feat_type, num_feat, fs_file_path)

                scaler_qmics = StandardScaler()
                qmics_train_scl = pd.DataFrame(scaler_qmics.fit_transform(qmics_train_full),
                                               index=qmics_train_full.index, columns=qmics_train_full.columns)
                qmics_test_scl = pd.DataFrame(scaler_qmics.transform(qmics_test), index=qmics_test.index,
                                              columns=qmics_test.columns)

                X_train_final = pd.concat([X_train_scl[list_best_feat], qmics_train_scl], axis=1)
                X_test_final = pd.concat([X_test_scl[list_best_feat], qmics_test_scl], axis=1)

                ef_model_name = os.path.join(path_ef_model, f'ef_model_fold_{fold_num}.pkl')

                pipeline = GridSearchCV(classifier, params, n_jobs=-1, cv=5, scoring='roc_auc')

                if not os.path.isfile(ef_model_name):
                    pipeline.fit(X_train_final, y_train_full.values.ravel())
                    with open(ef_model_name, 'wb') as f:
                        pickle.dump(pipeline, f)
                else:
                    with open(ef_model_name, 'rb') as f:
                        pipeline = pickle.load(f)

                if fold_num == 0:
                    lc_path = os.path.join(path_save, 'Learning_Curves', mode_dir,
                                           f'{name}_{feat_type}_{num_feat}_lc.png')
                    if not os.path.exists(lc_path):
                        plot_learning_curve(classifier, f"LC {mode}: {name}", X_train_final,
                                            y_train_full.values.ravel(), lc_path)

                y_pred = pipeline.predict(X_test_final)
                y_pred_proba = pipeline.predict_proba(X_test_final)
                metrics = get_metrics(y_test, y_pred, y_pred_proba, num_classes)

                if num_classes == 2:
                    pos_probs = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba
                    all_y_true_fold.extend(y_test)
                    all_y_proba_fold.extend(pos_probs)

                for key in fold_metrics:
                    if key in metrics:
                        fold_metrics[key].append(metrics[key])

                raw_fold_scores[name].append(metrics['AUC'])

            # ==============================================================================
            # LATE FUSION LOGIC
            # ==============================================================================
            elif mode == 'late':
                from sklearn.model_selection import cross_val_predict

                path_base_model = os.path.join(model_path, feat_type, name, str(num_feat), 'Base Models')
                ensure_dir(path_base_model)
                path_meta_model = os.path.join(model_path, feat_type, name, str(num_feat), 'Meta Models')
                ensure_dir(path_meta_model)

                imp = SimpleImputer(strategy='median')
                X_train_imp = pd.DataFrame(imp.fit_transform(X_train_full), columns=X_train_full.columns,
                                           index=X_train_full.index)
                X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

                scaler_gene = StandardScaler()
                X_train_scl = pd.DataFrame(scaler_gene.fit_transform(X_train_imp), columns=X_train_full.columns,
                                           index=X_train_full.index)
                X_test_scl = pd.DataFrame(scaler_gene.transform(X_test_imp), columns=X_test.columns, index=X_test.index)

                scaler_qmics = StandardScaler()
                qmics_train_scl = pd.DataFrame(scaler_qmics.fit_transform(qmics_train_full),
                                               index=qmics_train_full.index)
                qmics_test_scl = pd.DataFrame(scaler_qmics.transform(qmics_test), index=qmics_test.index)

                fs_filename = f'selected_features_fold_num_{fold_num}.npy'
                fs_file_path = os.path.join(path_fs, fs_filename)
                list_best_feat = perform_feature_selection(X_train_scl, y_train_full, feat_type, num_feat, fs_file_path)

                X_train_scl = X_train_scl[list_best_feat]
                X_test_scl = X_test_scl[list_best_feat]

                base_model_name = os.path.join(path_base_model, f'base_model_fold_{fold_num}.pkl')
                pipeline_base = GridSearchCV(classifier, params, n_jobs=-1, cv=3, scoring='roc_auc')

                if not os.path.isfile(base_model_name):
                    pipeline_base.fit(X_train_scl, np.ravel(y_train_full))
                    with open(base_model_name, 'wb') as f:
                        pickle.dump(pipeline_base, f)
                else:
                    with open(base_model_name, 'rb') as f:
                        pipeline_base = pickle.load(f)

                if fold_num == 0:
                    lc_path = os.path.join(path_save, 'Learning_Curves', mode_dir,
                                           f'{name}_{feat_type}_{num_feat}_lc.png')
                    if not os.path.exists(lc_path):
                        cv_lc = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        plot_learning_curve(classifier, f"LC {mode}: {name}", X_train_scl, np.ravel(y_train_full),
                                            lc_path, cv=cv_lc)

                min_class_count = np.min(np.bincount(np.ravel(y_train_full).astype(int)))

                try:
                    if min_class_count >= 3:
                        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                        oof_probs = cross_val_predict(
                            pipeline_base.best_estimator_,
                            X_train_scl,
                            np.ravel(y_train_full),
                            cv=cv_strategy,
                            method='predict_proba',
                            n_jobs=-1
                        )
                        probs_train_vals = oof_probs[:, 1]
                    else:
                        probs_train_vals = pipeline_base.predict_proba(X_train_scl)[:, 1]
                except (IndexError, ValueError):
                    probs_train_vals = pipeline_base.predict_proba(X_train_scl)[:, 1]

                probs_train = pd.DataFrame(probs_train_vals, index=qmics_train_scl.index, columns=['Gene_Prob'])
                probs_test = pd.DataFrame(pipeline_base.predict_proba(X_test_scl)[:, 1], index=qmics_test.index,
                                          columns=['Gene_Prob'])

                X_stack_train = pd.concat([probs_train, qmics_train_scl], axis=1)
                X_stack_test = pd.concat([probs_test, qmics_test_scl], axis=1)

                meta_scaler = StandardScaler()
                X_stack_train = meta_scaler.fit_transform(X_stack_train.values)
                X_stack_test = meta_scaler.transform(X_stack_test.values)

                meta_model_name = os.path.join(path_meta_model, f'meta_model_fold_{fold_num}.pkl')
                meta_classifier = LogisticRegression(class_weight='balanced')
                meta_params = {'C': [0.01, 0.1, 1, 10]}

                pipeline_meta = GridSearchCV(meta_classifier, meta_params, n_jobs=-1, cv=3, scoring='roc_auc')

                if not os.path.isfile(meta_model_name):
                    pipeline_meta.fit(X_stack_train, np.ravel(y_train_full))
                    with open(meta_model_name, 'wb') as f:
                        pickle.dump(pipeline_meta, f)
                else:
                    with open(meta_model_name, 'rb') as f:
                        pipeline_meta = pickle.load(f)

                y_pred = pipeline_meta.predict(X_stack_test)
                y_pred_proba = pipeline_meta.predict_proba(X_stack_test)
                metrics = get_metrics(y_test, y_pred, y_pred_proba, num_classes)

                if num_classes == 2:
                    pos_probs = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba
                    all_y_true_fold.extend(y_test)
                    all_y_proba_fold.extend(pos_probs)

                for key in fold_metrics:
                    if key in metrics:
                        fold_metrics[key].append(metrics[key])

                raw_fold_scores[name].append(metrics['AUC'])

        if num_classes == 2 and all_y_true_fold:
            calib_filename = f'{name}_{feat_type}_{num_feat}_calib.png'
            calib_path = os.path.join(path_save, 'Calibration_Curves', mode_dir, calib_filename)
            plot_title = f"Calibration ({mode.upper()}): {name}"
            plot_calibration_curve_custom(all_y_true_fold, all_y_proba_fold, plot_title, calib_path)

        mean_auc = np.mean(fold_metrics['AUC'])
        std_auc = np.std(fold_metrics['AUC'])
        ci_auc = 1.96 * (std_auc / np.sqrt(len(fold_metrics['AUC'])))

        mean_acc = np.mean(fold_metrics['acc'])
        std_acc = np.std(fold_metrics['acc'])
        ci_acc = 1.96 * (std_acc / np.sqrt(len(fold_metrics['acc'])))

        mean_brier = np.mean(fold_metrics['brier'])
        std_brier = np.std(fold_metrics['brier'])
        ci_brier = 1.96 * (std_brier / np.sqrt(len(fold_metrics['brier'])))

        row = {
            'Classifier': name,
            'Feature_Count': num_feat,
            'Fusion': mode,
            'Selection': feat_type,
            'Mean_AUC': mean_auc,
            'Std_AUC': std_auc,
            'CI_AUC': ci_auc,
            'Mean_Acc': mean_acc,
            'Std_Acc': std_acc,
            'CI_Acc': ci_acc,
            'Mean_F1': np.mean(fold_metrics['f1']),
            'Mean_MCC': np.mean(fold_metrics['mcc']),
            'Mean_Brier': mean_brier,
            'Std_Brier': std_brier,
            'CI_Brier': ci_brier
        }
        print(f"    Mean AUC: {mean_auc:.4f}")
        results_stats.append(row)

    stat_rows = []

    return results_stats, stat_rows, raw_fold_scores


def load_cohort_data(file_path, sheet_name=None, header=0,
                     col_idx_gene_start=19, col_idx_gene_end=-2,
                     qmics_col='qV2', type_col='Type',
                     limit_rows=None, is_uk=False):
    """Modular function to load and clean data for different cohorts."""
    if sheet_name:
        df = pd.read_excel(file_path, index_col=0, header=header, sheet_name=sheet_name)
    else:
        df = pd.read_excel(file_path, index_col=0)

    if is_uk:
        gene_ids = df.iloc[1, :].dropna().values
        df.columns = df.iloc[6, :]
        df = df.iloc[7:, :]

        df['qV2'] = pd.to_numeric(df['qV2'], errors='coerce')
        df = df.dropna(subset=['qV2'])

        qmics = pd.DataFrame(df['qV2'])

        y_raw = df['Normal or Cancer'].values.astype(int)
        y = pd.DataFrame([0 if i == 1 else 1 for i in y_raw], index=df.index)

        X = df.iloc[:, col_idx_gene_start:col_idx_gene_end].astype(float)
        X.columns = gene_ids

        print(f"  UK Data Loaded. Shape: {X.shape}")
        return X, y, qmics

    if limit_rows:
        df = df.iloc[:limit_rows, :]

    if qmics_col == 'q(II)':
        qmics = pd.DataFrame(df['q(II)'])
        qmics.columns = ['qV2']
    else:
        qmics = pd.DataFrame(df[qmics_col])

    if isinstance(df.iloc[0, 1], (int, float, np.integer)):
        y_raw = df.iloc[:, 1].values.astype(int)
    else:
        y_raw = df[type_col].values.astype(int)

    y = pd.DataFrame([0 if i == 1 else 1 for i in y_raw], index=df.index)
    X = df.iloc[:, col_idx_gene_start:col_idx_gene_end].astype(float).fillna(0)
    return X, y, qmics


################################################################################
# MAIN EXECUTION
################################################################################

EXCEL_PATH = 'C:/Users/farid/oral cancer paper/qV2 HNSCC Tissue Cohorts (all).xlsx'
BASE_SAVE_PATH = 'C:/Users/farid/oral cancer paper'

cohorts = [
    {'name': 'UK', 'sheet': None, 'header': 0, 'limit': None, 'col_end': -2, 'qmics_col': 'qV2', 'test_size': 0.25},
    {'name': 'India 1', 'sheet': 'HNSCC (IN-KGMU)', 'header': 7, 'limit': 48, 'col_end': -1, 'qmics_col': 'qV2',
     'test_size': 0.4},
    {'name': 'India 2', 'sheet': 'HNSCC (IN-MU)', 'header': 7, 'limit': 33, 'col_end': -2, 'qmics_col': 'q(II)',
     'test_size': 0.4},
{'name': 'China 1', 'sheet': 'HNSCC (CN)', 'header': 7, 'limit': 35, 'col_end': -2, 'qmics_col': 'q(II)', 'test_size': 0.4}
]

feature_counts = [2, 3, 5, 7, 10, 12, 14]
BASE_FEAT = 14
feat_types = ['mrmr', 'SVM-RFE']
global_raw_scores = {}
for cohort in cohorts:
    print(f"\nProcessing Cohort: {cohort['name']}")
    path_save = os.path.join(BASE_SAVE_PATH, cohort['name'])
    ensure_dir(path_save)

    if cohort['name'] == 'UK':
        X_c, y_c, qmics_c = load_cohort_data(EXCEL_PATH, is_uk=True)
    else:
        X_c, y_c, qmics_c = load_cohort_data(
            EXCEL_PATH,
            sheet_name=cohort['sheet'],
            header=cohort['header'],
            limit_rows=cohort['limit'],
            col_idx_gene_end=cohort['col_end'],
            qmics_col=cohort['qmics_col']
        )

    if cohort['name'] not in global_raw_scores:
        global_raw_scores[cohort['name']] = {}

    all_metrics_rows = []
    comparison_rows = []
    fusion_comp_rows = [] # Added new list for Fusion Stats

    for f_type in feat_types:
        for n_feat in feature_counts:
            print(f"  --- {f_type} | {n_feat} feats ---")

            res_ef, stats_ef, raw_ef = run_experiment_cv(X_c, qmics_c, y_c, path_save, n_feat, f_type, mode='early')
            all_metrics_rows.extend(res_ef)

            if 'early' not in global_raw_scores[cohort['name']]: global_raw_scores[cohort['name']]['early'] = {}
            if f_type not in global_raw_scores[cohort['name']]['early']: global_raw_scores[cohort['name']]['early'][f_type] = {}
            global_raw_scores[cohort['name']]['early'][f_type][n_feat] = raw_ef

            res_lf, stats_lf, raw_lf = run_experiment_cv(X_c, qmics_c, y_c, path_save, n_feat, f_type, mode='late')
            all_metrics_rows.extend(res_lf)

            if 'late' not in global_raw_scores[cohort['name']]: global_raw_scores[cohort['name']]['late'] = {}
            if f_type not in global_raw_scores[cohort['name']]['late']: global_raw_scores[cohort['name']]['late'][f_type] = {}
            global_raw_scores[cohort['name']]['late'][f_type][n_feat] = raw_lf

    print(f"  Calculating Statistics vs {BASE_FEAT} Features...")
    c_data = global_raw_scores[cohort['name']]

    for mode in ['early', 'late']:
        for f_type in feat_types:
            if BASE_FEAT not in c_data[mode][f_type]:
                continue

            base_scores_dict = c_data[mode][f_type][BASE_FEAT]

            for n_feat in feature_counts:
                if n_feat == BASE_FEAT:
                    continue

                current_scores_dict = c_data[mode][f_type][n_feat]

                for clf_name in current_scores_dict.keys():
                    if clf_name not in base_scores_dict: continue

                    scores_current = current_scores_dict[clf_name]
                    scores_base = base_scores_dict[clf_name]

                    stats_res = compare_models_stats(scores_current, scores_base)

                    comparison_rows.append({
                        'Cohort': cohort['name'],
                        'Classifier': clf_name,
                        'Fusion': mode,
                        'Selection': f_type,
                        'Comparison': f'{n_feat} vs {BASE_FEAT} (Base)',
                        'Diff_Mean_AUC': stats_res['diff_mean'],
                        'p_value_t': stats_res['p_val_t'],
                        'Significant': 'Yes' if stats_res['p_val_t'] < 0.05 else 'No'
                    })

    # --- NEW BLOCK: Calculating Statistics: Late vs Early Fusion ---
    print(f"  Calculating Statistics: Late vs Early Fusion...")
    for f_type in feat_types:
        for n_feat in feature_counts:
            # Verify both fusion modes completed for this feature count
            if n_feat in c_data['early'][f_type] and n_feat in c_data['late'][f_type]:
                early_clf_dict = c_data['early'][f_type][n_feat]
                late_clf_dict = c_data['late'][f_type][n_feat]

                for clf_name in early_clf_dict.keys():
                    if clf_name in late_clf_dict:
                        scores_early = early_clf_dict[clf_name]
                        scores_late = late_clf_dict[clf_name]

                        # Calculate statistics (Model A = Late, Model B = Early)
                        stats_fusion = compare_models_stats(scores_late, scores_early)

                        fusion_comp_rows.append({
                            'Cohort': cohort['name'],
                            'Classifier': clf_name,
                            'Selection': f_type,
                            'Feature_Count': n_feat,
                            'Comparison': 'Late vs Early',
                            'Late_Mean_AUC': np.mean(scores_late),
                            'Early_Mean_AUC': np.mean(scores_early),
                            'Diff_Mean_AUC (Late - Early)': stats_fusion['diff_mean'],
                            'p_value_t': stats_fusion['p_val_t'],
                            'Significant': 'Yes' if stats_fusion['p_val_t'] < 0.05 else 'No'
                        })
    # ----------------------------------------------------------------

    excel_path = os.path.join(path_save, f"{cohort['name']}_Full_Results.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(all_metrics_rows).to_excel(writer, sheet_name='Performance_Metrics', index=False)
        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            comp_df = comp_df.sort_values(by=['Fusion', 'Selection', 'Classifier'])
            comp_df.to_excel(writer, sheet_name='Stats_vs_14_Features', index=False)
        # --- NEW BLOCK: Save Fusion Stats to Excel ---
        if fusion_comp_rows:
            fusion_df = pd.DataFrame(fusion_comp_rows)
            fusion_df = fusion_df.sort_values(by=['Selection', 'Feature_Count', 'Classifier'])
            fusion_df.to_excel(writer, sheet_name='Stats_Late_vs_Early', index=False)
        # ---------------------------------------------

print("Processing Complete.")


# In[ ]:




