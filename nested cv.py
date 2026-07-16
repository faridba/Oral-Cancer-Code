import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold,
                                     LeaveOneOut, GridSearchCV, train_test_split, learning_curve)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef,
                             cohen_kappa_score, jaccard_score, classification_report)
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
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')

    return {
        'acc': accuracy_score(y_true, y_pred),
        'f1': cri['avg_f1'],
        'precision': cri['avg_pre'],
        'recall': cri['avg_rec'],
        'specificity': cri['avg_spe'],
        'AUC': auc,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'jaccard': jaccard_score(y_true, y_pred, average='weighted')
    }


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

    # Plot the shaded area for variance (Stability)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # Plot the mean scores
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
        # Load existing features to save time
        return list(np.load(save_path, allow_pickle=True))

    selected_features = []

    if method == 'mrmr':
        # mRMR requires a dataframe X and series/array y
        selected_features = mrmr_classif(X=X, y=y, K=n_features)

    elif method == 'SVM-RFE':
        # SVM-RFE requires a linear kernel to get coefficients
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_].tolist()

    # Save the selected features
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
    results_stats = []  # To store row data for Excel

    # Create Sub-directories
    mode_dir = 'Late_Fusion' if mode == 'late' else 'Early_Fusion'

    path_fs = os.path.join(path_save, 'Feature_Selection_Results', mode_dir, feat_type, str(num_feat))
    ensure_dir(path_fs)
    idx_path = os.path.join(path_save, 'Index_train_test')
    ensure_dir(idx_path)
    model_path = os.path.join(path_save, 'Models', mode_dir)
    ensure_dir(model_path)
    ensure_dir(os.path.join(path_save, 'Learning_Curves', mode_dir))

    # Store raw scores for statistical comparison later
    raw_fold_scores = {name: [] for name in CLASSIFIERS.keys()}

    for name, (classifier, params) in CLASSIFIERS.items():
        print(f"  [{mode.upper()}] Classifier: {name}")

        fold_metrics = {'AUC': [], 'acc': [], 'f1': [], 'mcc': [], 'precision': [],
                        'recall': [], 'specificity': [], 'kappa': [], 'jaccard': []}

        if k_fold_num > 1:
            kf = RepeatedStratifiedKFold(n_splits=k_fold_num, n_repeats=5, random_state=42)
        else:
            kf = LeaveOneOut()

        for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            # --- Data Split Management ---
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

                # Preprocessing
                imp = SimpleImputer(strategy='median')
                X_train_imp = pd.DataFrame(imp.fit_transform(X_train_full), columns=X_train_full.columns,
                                           index=X_train_full.index)
                X_test_imp = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)

                scaler_gene = StandardScaler()
                X_train_scl = pd.DataFrame(scaler_gene.fit_transform(X_train_imp), columns=X_train_full.columns,
                                           index=X_train_full.index)
                X_test_scl = pd.DataFrame(scaler_gene.transform(X_test_imp), columns=X_test.columns, index=X_test.index)

                # Feature Selection
                fs_filename = f'selected_features_fold_num_{fold_num}.npy'
                fs_file_path = os.path.join(path_fs, fs_filename)
                list_best_feat = perform_feature_selection(X_train_scl, y_train_full, feat_type, num_feat, fs_file_path)

                # Qmics Scaling (Important!)
                scaler_qmics = StandardScaler()
                qmics_train_scl = pd.DataFrame(scaler_qmics.fit_transform(qmics_train_full),
                                               index=qmics_train_full.index, columns=qmics_train_full.columns)
                qmics_test_scl = pd.DataFrame(scaler_qmics.transform(qmics_test), index=qmics_test.index,
                                              columns=qmics_test.columns)

                # Concatenation (Genes + Qmics)
                X_train_final = pd.concat([X_train_scl[list_best_feat], qmics_train_scl], axis=1)
                X_test_final = pd.concat([X_test_scl[list_best_feat], qmics_test_scl], axis=1)

                ef_model_name = os.path.join(path_ef_model, f'ef_model_fold_{fold_num}.pkl')

                # Training
                pipeline = GridSearchCV(classifier, params, n_jobs=-1, cv=5, scoring='roc_auc')

                if not os.path.isfile(ef_model_name):
                    # FIX: Train on X_train_final (Merged), NOT X_train_scl
                    pipeline.fit(X_train_final, y_train_full.values.ravel())
                    with open(ef_model_name, 'wb') as f:
                        pickle.dump(pipeline, f)
                else:
                    with open(ef_model_name, 'rb') as f:
                        pipeline = pickle.load(f)

                # Learning Curve (Fold 0 only)
                if fold_num == 0:
                    lc_path = os.path.join(path_save, 'Learning_Curves', mode_dir,
                                           f'{name}_{feat_type}_{num_feat}_lc.png')
                    if not os.path.exists(lc_path):
                        plot_learning_curve(classifier, f"LC {mode}: {name}", X_train_final, y_train_full.values.ravel(), lc_path)

                # FIX: Prediction & Metrics Storage (Was missing in your code)
                y_pred = pipeline.predict(X_test_final)
                y_pred_proba = pipeline.predict_proba(X_test_final)
                metrics = get_metrics(y_test, y_pred, y_pred_proba, num_classes)

                # Store Metrics
                for key in fold_metrics:
                    if key in metrics:
                        fold_metrics[key].append(metrics[key])

                raw_fold_scores[name].append(metrics['AUC'])

            # ==============================================================================
            # LATE FUSION LOGIC
            # ==============================================================================
            elif mode == 'late':
                # Inner Split for Stacking
                if os.path.isfile(f_train_val):
                    train_val_idx = np.load(f_train_val)
                    val_idx = np.load(f_val)
                    X_train = X_train_full.loc[train_val_idx]
                    X_val = X_train_full.loc[val_idx]
                    y_train = y_train_full.loc[train_val_idx].values.ravel()
                    y_val = y_train_full.loc[val_idx].values.ravel()
                    qmics_train = qmics_train_full.loc[train_val_idx]
                    qmics_val = qmics_train_full.loc[val_idx]
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_full, y_train_full, test_size=0.4, stratify=y_train_full, random_state=42
                    )
                    y_train, y_val = y_train.values.ravel(), y_val.values.ravel()
                    qmics_train = qmics_train_full.loc[X_train.index]
                    qmics_val = qmics_train_full.loc[X_val.index]
                    np.save(f_train_val, list(X_train.index))
                    np.save(f_val, list(X_val.index))

                # --- Preprocessing Genes ---
                imp = SimpleImputer(strategy='median')
                X_train = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns, index=X_test.index)
                X_val = pd.DataFrame(imp.transform(X_val), columns=X_val.columns, index=X_val.index)

                scaler_gene = StandardScaler()
                X_train_scl = pd.DataFrame(scaler_gene.fit_transform(X_train), columns=X_train.columns,
                                           index=X_train.index)
                X_test_scl = pd.DataFrame(scaler_gene.transform(X_test), columns=X_test.columns, index=X_test.index)
                X_val_scl = pd.DataFrame(scaler_gene.transform(X_val), columns=X_val.columns, index=X_val.index)

                # --- Feature Selection ---
                fs_filename = f'selected_features_fold_num_{fold_num}.npy'
                fs_file_path = os.path.join(path_fs, fs_filename)
                list_best_feat = perform_feature_selection(X_train_scl, y_train, feat_type, num_feat, fs_file_path)

                X_train_scl = X_train_scl[list_best_feat]
                X_test_scl = X_test_scl[list_best_feat]
                X_val_scl = X_val_scl[list_best_feat]

                # Directories
                path_base_model = os.path.join(model_path, feat_type, name, str(num_feat), 'Base Models')
                ensure_dir(path_base_model)
                path_meta_model = os.path.join(model_path, feat_type, name, str(num_feat), 'Meta Models')
                ensure_dir(path_meta_model)

                # Base Learner
                base_model_name = os.path.join(path_base_model, f'base_model_fold_{fold_num}.pkl')

                if not os.path.isfile(base_model_name):
                    pipeline_base = GridSearchCV(classifier, params, n_jobs=-1, cv=5, scoring='roc_auc')
                    pipeline_base.fit(X_train_scl, y_train)
                    with open(base_model_name, 'wb') as f:
                        pickle.dump(pipeline_base, f)
                else:
                    with open(base_model_name, 'rb') as f:
                        pipeline_base = pickle.load(f)

                # Learning Curve
                if fold_num == 0:
                    lc_path = os.path.join(path_save, 'Learning_Curves', mode_dir,
                                           f'{name}_{feat_type}_{num_feat}_lc.png')
                    if not os.path.exists(lc_path):
                        plot_learning_curve(classifier, f"LC {mode}: {name}", X_train_scl, y_train, lc_path)

                # Generate Probabilities for Stacking
                probs_val = pd.DataFrame(pipeline_base.predict_proba(X_val_scl), index=qmics_val.index)
                probs_test = pd.DataFrame(pipeline_base.predict_proba(X_test_scl), index=qmics_test.index)

                # --- 4. Stacking / Late Fusion ---
                # Check alignment before concat
                X_stack_val = pd.concat([probs_val, qmics_val], axis=1)
                X_stack_test = pd.concat([probs_test, qmics_test], axis=1)

                meta_scaler = StandardScaler()
                X_stack_val = meta_scaler.fit_transform(X_stack_val.values)
                X_stack_test = meta_scaler.transform(X_stack_test.values)

                # List to store results of all potential Meta-Learners for this fold
                fold_meta_candidates = {}

                for name_lf, (meta_classifier, meta_params) in CLASSIFIERS.items():
                    path_model_lf_specific = os.path.join(path_meta_model, name_lf)
                    ensure_dir(path_model_lf_specific)
                    meta_model_name = os.path.join(path_model_lf_specific, f'meta_model_fold_{fold_num}.pkl')

                    pipeline_meta = GridSearchCV(meta_classifier, meta_params, n_jobs=-1, cv=5, scoring='roc_auc')

                    if not os.path.isfile(meta_model_name):
                        pipeline_meta.fit(X_stack_val, y_val)
                        with open(meta_model_name, 'wb') as f:
                            pickle.dump(pipeline_meta, f)
                    else:
                        with open(meta_model_name, 'rb') as f:
                            pipeline_meta = pickle.load(f)

                    val_cv_score = pipeline_meta.best_score_

                    # 2. Store this score
                    fold_meta_candidates[name_lf] = {
                        'model': pipeline_meta,
                        'name': name_lf,
                        'val_score': val_cv_score
                    }
                meta_results = pd.DataFrame(fold_meta_candidates).T
                best_meta_clf = meta_results.iloc[np.argmax(meta_results['val_score'].values)]['name']
                best_meta_model = meta_results.iloc[np.argmax(meta_results['val_score'].values)]['model']
                best_val_auc = meta_results.iloc[np.argmax(meta_results['val_score'].values)]['val_score']


                y_pred = best_meta_model.predict(X_stack_test)
                y_pred_proba = best_meta_model.predict_proba(X_stack_test)

                metrics = get_metrics(y_test, y_pred, y_pred_proba, num_classes)

                # Add to fold metrics
                for key in fold_metrics:
                    if key in metrics:
                        fold_metrics[key].append(metrics[key])

                raw_fold_scores[name].append(metrics['AUC'])

        # --- Aggregating Results for Excel ---
        mean_auc = np.mean(fold_metrics['AUC'])
        std_auc = np.std(fold_metrics['AUC'])
        ci_auc = 1.96 * (std_auc / np.sqrt(len(fold_metrics['AUC'])))

        mean_acc = np.mean(fold_metrics['acc'])
        std_acc = np.std(fold_metrics['acc'])
        ci_acc = 1.96 * (std_acc / np.sqrt(len(fold_metrics['acc'])))

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
            'Mean_MCC': np.mean(fold_metrics['mcc'])
        }
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

        # 2a. Drop NaNs in Qmics (UK specific column is 'qV2')
        # We check for NaNs in 'qV2' and keep only valid rows
        # coercion ensures non-numeric garbage becomes NaN, then we drop
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

EXCEL_PATH = 'qV2 HNSCC Tissue Cohorts (all).xlsx'
BASE_SAVE_PATH = 'oral cancer paper'

cohorts = [
    {'name': 'UK', 'sheet': None, 'header': 0, 'limit': None, 'col_end': -2, 'qmics_col': 'qV2', 'test_size': 0.25},
    {'name': 'India 1', 'sheet': 'HNSCC (IN-KGMU)', 'header': 7, 'limit': 48, 'col_end': -1, 'qmics_col': 'qV2',
     'test_size': 0.4},
    {'name': 'India 2', 'sheet': 'HNSCC (IN-MU)', 'header': 7, 'limit': 33, 'col_end': -2, 'qmics_col': 'q(II)',
     'test_size': 0.4},
    #{'name': 'China 1', 'sheet': 'HNSCC (CN)', 'header': 7, 'limit': 35, 'col_end': -2, 'qmics_col': 'q(II)',
     #'test_size': 0.4}
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

    # We will collect the comparison stats here
    comparison_rows = []

    for f_type in feat_types:
        for n_feat in feature_counts:
            print(f"  --- {f_type} | {n_feat} feats ---")

            # --- RUN EARLY FUSION ---
            # Note: We unpack 3 values now
            res_ef, stats_ef, raw_ef = run_experiment_cv(X_c, qmics_c, y_c, path_save, n_feat, f_type, mode='early')
            all_metrics_rows.extend(res_ef)

            # Store Raw Scores for comparison
            # Structure: global[Cohort]['early'][f_type][n_feat] = raw_scores_dict
            if 'early' not in global_raw_scores[cohort['name']]: global_raw_scores[cohort['name']]['early'] = {}
            if f_type not in global_raw_scores[cohort['name']]['early']: global_raw_scores[cohort['name']]['early'][
                f_type] = {}
            global_raw_scores[cohort['name']]['early'][f_type][n_feat] = raw_ef

            # --- RUN LATE FUSION ---
            res_lf, stats_lf, raw_lf = run_experiment_cv(X_c, qmics_c, y_c, path_save, n_feat, f_type, mode='late')
            all_metrics_rows.extend(res_lf)

            # Store Raw Scores
            if 'late' not in global_raw_scores[cohort['name']]: global_raw_scores[cohort['name']]['late'] = {}
            if f_type not in global_raw_scores[cohort['name']]['late']: global_raw_scores[cohort['name']]['late'][
                f_type] = {}
            global_raw_scores[cohort['name']]['late'][f_type][n_feat] = raw_lf

    # ==============================================================================
    # NEW STEP: PERFORM STATISTICAL COMPARISON VS BASE_FEAT (14)
    # ==============================================================================
    print(f"  Calculating Statistics vs {BASE_FEAT} Features...")

    # Iterate through saved scores to compare N vs 14
    c_data = global_raw_scores[cohort['name']]

    for mode in ['early', 'late']:
        for f_type in feat_types:
            # Check if our base feature count exists in the results
            if BASE_FEAT not in c_data[mode][f_type]:
                print(f"Warning: Base feature count {BASE_FEAT} not found for {mode}/{f_type}")
                continue

            # Get the baseline scores (dict of classifiers)
            base_scores_dict = c_data[mode][f_type][BASE_FEAT]

            for n_feat in feature_counts:
                if n_feat == BASE_FEAT:
                    continue  # Don't compare 14 vs 14

                # Get current scores
                current_scores_dict = c_data[mode][f_type][n_feat]

                # Compare every classifier
                for clf_name in current_scores_dict.keys():
                    if clf_name not in base_scores_dict: continue

                    scores_current = current_scores_dict[clf_name]
                    scores_base = base_scores_dict[clf_name]

                    # Run your existing stats function
                    stats_res = compare_models_stats(scores_current, scores_base)

                    comparison_rows.append({
                        'Cohort': cohort['name'],
                        'Classifier': clf_name,
                        'Fusion': mode,
                        'Selection': f_type,
                        'Comparison': f'{n_feat} vs {BASE_FEAT} (Base)',
                        'Diff_Mean_AUC': stats_res['diff_mean'],  # Positive means n_feat is better than 14
                        'p_value_t': stats_res['p_val_t'],
                        'Significant': 'Yes' if stats_res['p_val_t'] < 0.05 else 'No'
                    })

    # --- SAVE TO EXCEL ---
    excel_path = os.path.join(path_save, f"{cohort['name']}_Full_Results.xlsx")

    with pd.ExcelWriter(excel_path) as writer:
        # Sheet 1: Performance Metrics
        pd.DataFrame(all_metrics_rows).to_excel(writer, sheet_name='Performance_Metrics', index=False)

        # Sheet 2: Comparison vs 14 Features
        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            # Sort for easier reading
            comp_df = comp_df.sort_values(by=['Fusion', 'Selection', 'Classifier'])
            comp_df.to_excel(writer, sheet_name='Stats_vs_14_Features', index=False)

print("Processing Complete.")
