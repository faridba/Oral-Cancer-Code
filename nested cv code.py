import pandas as pd
import numpy as np
import os
import warnings
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve

# --------------------------------
import os

os.environ["PYTHONWARNINGS"] = "ignore"
import warnings

warnings.filterwarnings('ignore')
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef, f1_score)

# Feature Selection
from mrmr import mrmr_classif
from sklearn.feature_selection import RFE

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
os.environ["SCIPY_ARRAY_API"] = "1"

# ==========================================
# CONFIGURATION
# ==========================================
EXCEL_PATH = 'C:/Users/farid/oral cancer paper/qV2 HNSCC Tissue Cohorts (all).xlsx'
BASE_SAVE_PATH = 'C:/Users/farid/oral cancer paper/Inter_Cohort_Results_Stats'

# Define Classifiers
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
    os.makedirs(path, exist_ok=True)


# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_calibration(y_true, y_proba, title, save_path):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=5)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='s', color='blue', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


def plot_lc(estimator, title, X, y, save_path):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='roc_auc'
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(7, 5))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="CV Score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")

    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("ROC AUC")
    plt.grid(True)
    plt.legend(loc="best")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')


# ==========================================
# METRICS & FEATURE SELECTION
# ==========================================
def get_qmids_bootstrapped_metrics(y_true, qmics_proba, n_boot=1000):
    """NEW: Calculates bootstrapped AUC distribution for the QMIDS baseline."""
    y_true = np.array(y_true)
    qmics_proba = np.array(qmics_proba).ravel()
    boot_aucs = []

    for i in range(n_boot):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2: continue

        auc = roc_auc_score(y_true[indices], qmics_proba[indices])
        boot_aucs.append(auc)

    return {
        'Mean_AUC': np.mean(boot_aucs) if boot_aucs else 0,
        'raw_aucs': boot_aucs
    }


def get_bootstrapped_metrics(y_true, y_pred, y_proba, n_boot=1000):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    boot_aucs, boot_accs, boot_f1s, boot_mccs = [], [], [], []

    for i in range(n_boot):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2: continue

        auc = roc_auc_score(y_true[indices], y_proba[indices, 1]) if y_proba.shape[1] > 1 else 0.5
        boot_aucs.append(auc)
        boot_accs.append(accuracy_score(y_true[indices], y_pred[indices]))
        boot_f1s.append(f1_score(y_true[indices], y_pred[indices]))
        boot_mccs.append(matthews_corrcoef(y_true[indices], y_pred[indices]))

    def calc_stats(scores):
        if not scores: return 0, 0, 0
        mean = np.mean(scores)
        std = np.std(scores)
        ci = 1.96 * std
        return mean, std, ci

    m_auc, s_auc, ci_auc = calc_stats(boot_aucs)
    m_acc, s_acc, ci_acc = calc_stats(boot_accs)
    m_f1, _, _ = calc_stats(boot_f1s)
    m_mcc, _, _ = calc_stats(boot_mccs)

    return {
        'Mean_AUC': m_auc, 'Std_AUC': s_auc, 'CI_AUC': ci_auc,
        'Mean_Acc': m_acc, 'Std_Acc': s_acc, 'CI_Acc': ci_acc,
        'Mean_F1': m_f1, 'Mean_MCC': m_mcc, 'raw_aucs': boot_aucs
    }


def perform_feature_selection(X, y, method, n_features):
    if method == 'mrmr':
        return mrmr_classif(X=X, y=y, K=n_features)
    elif method == 'SVM-RFE':
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        return X.columns[rfe.support_].tolist()
    return []


# ==========================================
# DATA LOADING
# ==========================================
def load_cohort_data(file_path, sheet_name=None, header=0, col_idx_gene_start=19, col_idx_gene_end=-2, qmics_col='qV2',
                     type_col='Type', limit_rows=None, is_uk=False):
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
        return X, y, qmics

    if limit_rows: df = df.iloc[:limit_rows, :]
    actual_qmics_col = 'q(II)' if qmics_col == 'q(II)' else qmics_col
    df[actual_qmics_col] = pd.to_numeric(df[actual_qmics_col], errors='coerce')
    df = df.dropna(subset=[actual_qmics_col])
    qmics = pd.DataFrame(df[actual_qmics_col])
    qmics.columns = ['qV2']

    if isinstance(df.iloc[0, 1], (int, float, np.integer)):
        y_raw = df.iloc[:, 1].values.astype(int)
    else:
        y_raw = df[type_col].values.astype(int)
    y = pd.DataFrame([0 if i == 1 else 1 for i in y_raw], index=df.index)
    X = df.iloc[:, col_idx_gene_start:col_idx_gene_end].astype(float).fillna(0)
    return X, y, qmics


# ==========================================
# EXPERIMENT ENGINE
# ==========================================
def run_inter_cohort(train_data, test_data, train_name, test_name, n_feat, feat_type):
    X_train, y_train, qmics_train = train_data
    X_test, y_test, qmics_test = test_data

    plot_dir = os.path.join(BASE_SAVE_PATH, 'Plots', f"{train_name}_to_{test_name}")
    ensure_dir(plot_dir)

    imp = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

    scaler_gene = StandardScaler()
    X_train_scl = pd.DataFrame(scaler_gene.fit_transform(X_train_imp), columns=X_train.columns, index=X_train.index)
    X_test_scl = pd.DataFrame(scaler_gene.transform(X_test_imp), columns=X_test.columns, index=X_test.index)

    selected_feats = perform_feature_selection(X_train_scl, y_train.values.ravel(), feat_type, n_feat)
    X_train_sel = X_train_scl[selected_feats]
    X_test_sel = X_test_scl[selected_feats]

    scaler_qmics = StandardScaler()
    qmics_train_scl = pd.DataFrame(scaler_qmics.fit_transform(qmics_train), index=qmics_train.index, columns=['qV2'])
    qmics_test_scl = pd.DataFrame(scaler_qmics.transform(qmics_test), index=qmics_test.index, columns=['qV2'])

    # --- NEW: Get QMIDS baseline bootstrapped metrics for the TEST cohort ---
    qmids_stats = get_qmids_bootstrapped_metrics(y_test.values.ravel(), qmics_test.values)
    qmids_raw_aucs = qmids_stats['raw_aucs']
    qmids_mean_auc = qmids_stats['Mean_AUC']

    results_rows = []

    y_train_flat = y_train.values.ravel()
    min_class_count = np.min(np.bincount(y_train_flat.astype(int)))

    n_splits_outer = max(2, min(5, min_class_count))
    cv_outer = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

    for clf_name, (clf, params) in CLASSIFIERS.items():
        base_filename = f"{clf_name}_{feat_type}_{n_feat}f"

        # --- A. NO FUSION ---
        pipeline_none = GridSearchCV(clf, params, cv=cv_outer, scoring='roc_auc', n_jobs=-1)
        pipeline_none.fit(X_train_sel, y_train_flat)
        probs_none = pipeline_none.predict_proba(X_test_sel)

        plot_lc(pipeline_none.best_estimator_, f"LC (None): {clf_name}", X_train_sel, y_train_flat,
                os.path.join(plot_dir, f"{base_filename}_None_LC.png"))
        if probs_none.shape[1] > 1:
            plot_calibration(y_test.values.ravel(), probs_none[:, 1], f"Calib (None): {clf_name}",
                             os.path.join(plot_dir, f"{base_filename}_None_Calib.png"))

        stats_none = get_bootstrapped_metrics(y_test.values.ravel(), pipeline_none.predict(X_test_sel), probs_none)
        results_rows.append({'Train_Cohort': train_name, 'Test_Cohort': test_name, 'Model': clf_name, 'Fusion': 'None',
                             'Features': n_feat, 'Selection': feat_type,
                             'qmids_Mean_AUC': qmids_mean_auc, 'qmids_raw_aucs': qmids_raw_aucs, **stats_none})

        # --- B. EARLY FUSION ---
        X_train_ef = pd.concat([X_train_sel, qmics_train_scl], axis=1)
        X_test_ef = pd.concat([X_test_sel, qmics_test_scl], axis=1)

        pipeline_ef = GridSearchCV(clf, params, cv=cv_outer, scoring='roc_auc', n_jobs=-1)
        pipeline_ef.fit(X_train_ef, y_train_flat)
        probs_ef = pipeline_ef.predict_proba(X_test_ef)

        plot_lc(pipeline_ef.best_estimator_, f"LC (Early): {clf_name}", X_train_ef, y_train_flat,
                os.path.join(plot_dir, f"{base_filename}_Early_LC.png"))
        if probs_ef.shape[1] > 1:
            plot_calibration(y_test.values.ravel(), probs_ef[:, 1], f"Calib (Early): {clf_name}",
                             os.path.join(plot_dir, f"{base_filename}_Early_Calib.png"))

        stats_ef = get_bootstrapped_metrics(y_test.values.ravel(), pipeline_ef.predict(X_test_ef), probs_ef)
        results_rows.append({'Train_Cohort': train_name, 'Test_Cohort': test_name, 'Model': clf_name, 'Fusion': 'Early',
                             'Features': n_feat, 'Selection': feat_type,
                             'qmids_Mean_AUC': qmids_mean_auc, 'qmids_raw_aucs': qmids_raw_aucs, **stats_ef})

        # --- C. LATE FUSION ---
        train_proba_list = []
        n_splits_inner = max(2, min(3 if len(y_train) < 50 else 5, min_class_count))
        cv_inner = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=42)

        for tr_idx, val_idx in cv_outer.split(X_train_sel, y_train):
            X_tr_fold, X_val_fold = X_train_sel.iloc[tr_idx], X_train_sel.iloc[val_idx]
            y_tr_fold = y_train.iloc[tr_idx].values.ravel()

            if len(np.unique(y_tr_fold)) < 2: continue

            base_cv = GridSearchCV(clf, params, cv=cv_inner, scoring='roc_auc', n_jobs=-1)
            base_cv.fit(X_tr_fold, y_tr_fold)
            probs_val = base_cv.predict_proba(X_val_fold)
            train_proba_list.append(pd.DataFrame(probs_val, index=X_train_sel.index[val_idx]))

        if train_proba_list:
            df_oof_probs = pd.concat(train_proba_list).sort_index()
            qmics_train_aligned = qmics_train_scl.loc[df_oof_probs.index]
            X_stack_train = pd.concat([df_oof_probs, qmics_train_aligned], axis=1)
            X_stack_train.columns = range(X_stack_train.shape[1])

            meta_learner = LogisticRegression(class_weight='balanced')
            meta_y_train = y_train.loc[df_oof_probs.index].values.ravel()
            meta_learner.fit(X_stack_train, meta_y_train)

            base_final = GridSearchCV(clf, params, cv=cv_outer, scoring='roc_auc', n_jobs=-1)
            base_final.fit(X_train_sel, y_train_flat)

            probs_test = pd.DataFrame(base_final.predict_proba(X_test_sel), index=X_test.index)
            X_stack_test = pd.concat([probs_test, qmics_test_scl], axis=1)
            X_stack_test.columns = range(X_stack_test.shape[1])

            probs_lf = meta_learner.predict_proba(X_stack_test)

            plot_lc(meta_learner, f"LC (Late): Meta-LogReg", X_stack_train, meta_y_train,
                    os.path.join(plot_dir, f"{base_filename}_Late_LC.png"))
            if probs_lf.shape[1] > 1:
                plot_calibration(y_test.values.ravel(), probs_lf[:, 1], f"Calib (Late): Meta-LogReg",
                                 os.path.join(plot_dir, f"{base_filename}_Late_Calib.png"))

            stats_lf = get_bootstrapped_metrics(y_test.values.ravel(), meta_learner.predict(X_stack_test), probs_lf)
            results_rows.append(
                {'Train_Cohort': train_name, 'Test_Cohort': test_name, 'Model': clf_name, 'Fusion': 'Late',
                 'Features': n_feat, 'Selection': feat_type,
                 'qmids_Mean_AUC': qmids_mean_auc, 'qmids_raw_aucs': qmids_raw_aucs, **stats_lf})

    return results_rows


def calculate_p_value(dist_a, dist_b):
    """Calculates p-value by finding the fraction of bootstrap differences <= 0."""
    min_len = min(len(dist_a), len(dist_b))

    if len(dist_a) != len(dist_b):
        print(f"Warning: Mismatched lengths ({len(dist_a)} vs {len(dist_b)}). Truncating to {min_len}.")

    a_arr = np.array(dist_a)[:min_len]
    b_arr = np.array(dist_b)[:min_len]

    diffs = a_arr - b_arr
    return np.mean(diffs <= 0)


def generate_comparisons(df_results):
    """NEW: Exclusively compares every model variation against the QMIDS baseline."""
    comparison_rows = []

    for _, row in df_results.iterrows():
        model_dist = row['raw_aucs']
        qmids_dist = row['qmids_raw_aucs']

        # Calculate p-value (bootstrap method)
        p_val = calculate_p_value(model_dist, qmids_dist)

        comparison_rows.append({
            'Train_Cohort': row['Train_Cohort'],
            'Test_Cohort': row['Test_Cohort'],
            'Model': row['Model'],
            'Selection': row['Selection'],
            'Features': row['Features'],
            'Fusion': row['Fusion'],
            'Model_Mean_AUC': row['Mean_AUC'],
            'QMIDS_Mean_AUC': row['qmids_Mean_AUC'],
            'Diff_Mean_AUC': row['Mean_AUC'] - row['qmids_Mean_AUC'],
            'p_value_bootstrap': p_val,
            'Significant_vs_QMIDS': 'Yes' if p_val < 0.05 else 'No'
        })

    return pd.DataFrame(comparison_rows)


# ==========================================
# MAIN EXECUTION
# ==========================================
print("Loading all cohorts...")
COHORT_DEFS = [
    {'name': 'UK', 'sheet': None, 'header': 0, 'limit': None, 'col_end': -2, 'qmics_col': 'qV2'},
    {'name': 'India1', 'sheet': 'HNSCC (IN-KGMU)', 'header': 7, 'limit': 48, 'col_end': -1, 'qmics_col': 'qV2'},
    {'name': 'India2', 'sheet': 'HNSCC (IN-MU)', 'header': 7, 'limit': 33, 'col_end': -2, 'qmics_col': 'q(II)'},
    {'name': 'China1', 'sheet': 'HNSCC (CN)', 'header': 7, 'limit': 35, 'col_end': -2, 'qmics_col': 'q(II)'}
]

data_store = {}
reference_cols = None

for c_def in COHORT_DEFS:
    if c_def['name'] == 'UK':
        X, y, q = load_cohort_data(EXCEL_PATH, is_uk=True)
        reference_cols = X.columns
    else:
        X, y, q = load_cohort_data(EXCEL_PATH, sheet_name=c_def['sheet'], header=c_def['header'],
                                   limit_rows=c_def['limit'], col_idx_gene_end=c_def['col_end'],
                                   qmics_col=c_def['qmics_col'])
        X.columns = reference_cols
    data_store[c_def['name']] = (X, y, q)

ensure_dir(BASE_SAVE_PATH)
all_results = []
feature_counts = [2, 3, 5, 7, 10, 14]
feat_types = ['mrmr', 'SVM-RFE']

print("\nStarting Experiments...")

for train_name in data_store.keys():
    for test_name in data_store.keys():
        if train_name == test_name: continue

        print(f"  {train_name} -> {test_name}")
        for f_type in feat_types:
            for n_feat in feature_counts:
                res = run_inter_cohort(data_store[train_name], data_store[test_name], train_name, test_name, n_feat,
                                       f_type)
                all_results.extend(res)

df_main = pd.DataFrame(all_results)

print("\nGenerating Comparisons vs QMIDS Baseline...")
df_comp = generate_comparisons(df_main)

# Drop the raw arrays so the Excel file doesn't crash or become massive
df_main_clean = df_main.drop(columns=['raw_aucs', 'qmids_raw_aucs'])

save_file = os.path.join(BASE_SAVE_PATH, 'Inter_Cohort_Final_Stats.xlsx')
with pd.ExcelWriter(save_file) as writer:
    df_main_clean.to_excel(writer, sheet_name='Metrics', index=False)
    df_comp.to_excel(writer, sheet_name='Stats_vs_QMIDS_Baseline', index=False)

print(f"\nProcessing Complete. File saved to: {save_file}")