import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

clients_cols = [
    'avg_inflow_30d',
    'has_mobile_app',
    'txn_count_30d',
    'spend_food_share'
]

risk_cols = [
    'loan_amount',
    'monthly_payment',
    'credit_history_months',
    'debt_to_income_known',
    'loan_term_months',
    'past_due_count_12m',
    'pred_inflow',
    'pti_ratio',
    'loan_to_inflow_ratio'
]



def add_risk_features(credit, inflow_pred, clients_ids):
    dt = pd.DataFrame({
        'client_id': clients_ids, 
        'pred_inflow': inflow_pred
    })

    df = credit.merge(dt, on='client_id', how='left')
    df['pred_inflow'] = np.clip(df['pred_inflow'], 1, None)
    df['pti_ratio'] = df['monthly_payment'] / df['pred_inflow']
    df['loan_to_inflow_ratio'] = df['loan_amount'] / df['pred_inflow']

    return df



def train_inflow(train_dir, artifacts_dir, config_path=None):
    c_train = pd.read_csv(train_dir / 'clients_train.csv')
    alpha = 10

    if config_path and config_path.exists():
        with open(config_path) as f:
            alpha = json.load(f).get('alpha', alpha)

    sc = StandardScaler()
    X_train = sc.fit_transform(c_train[clients_cols])
    y_train = c_train['inflow_next_30d'] / 1000
    
    model = HuberRegressor(alpha=alpha)
    model.fit(X_train, y_train)
    
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'model': model, 
        'scaler': sc, 
        'cols': clients_cols, 
        'scale': 1000
    }, artifacts_dir / 'inflow_model.joblib')

    manifest = {
        "task": "inflow",
        "model_file": "inflow_model.joblib",
        "feature_columns": clients_cols,
        "effective_config": {"alpha": alpha}
    }
    (artifacts_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Inflow модель")


def predict_inflow(test_dir, artifacts_dir, out_path):
    bundle = joblib.load(artifacts_dir / 'inflow_model.joblib')
    c_test = pd.read_csv(test_dir / 'clients_test.csv')
    
    pred = bundle['model'].predict(bundle['scaler'].transform(c_test[bundle['cols']])) * bundle['scale']
    pred = np.clip(pred, 0, None)
    
    out = pd.DataFrame({
        'client_id': c_test['client_id'], 
        'predicted_inflow': pred
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(out_path, index=False)
    print("Inflow предсказания")



def train_risk(train_dir, artifacts_dir, config_path=None):
    c_train = pd.read_csv(train_dir / 'clients_train.csv')
    r_train = pd.read_csv(train_dir / 'credit_train.csv')

    alpha, iterations, depth, lr = 10, 300, 7, 0.05
    
    if config_path and config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
                alpha = cfg.get('huber_alpha', alpha)
                iterations = cfg.get('catboost_iterations', iterations)
                depth = cfg.get('catboost_depth', depth)
                lr = cfg.get('catboost_lr', lr)
        except Exception as e:
            print(e)

    inflow_oof = np.zeros(len(c_train))
    
    gkf = GroupKFold(n_splits=4)
    for tr, va in gkf.split(c_train, groups=c_train['branch_id']):
        sc_fold = StandardScaler()
        X_tr = sc_fold.fit_transform(c_train.loc[tr, clients_cols])
        X_va = sc_fold.transform(c_train.loc[va, clients_cols])
        y_tr = c_train.loc[tr, 'inflow_next_30d'] / 1000
        
        m_fold = HuberRegressor(alpha=alpha)
        m_fold.fit(X_tr, y_tr)
        inflow_oof[va] = m_fold.predict(X_va) * 1000

    train = add_risk_features(r_train, inflow_oof, c_train['client_id'].values)

    sc_full = StandardScaler()
    X_full = sc_full.fit_transform(c_train[clients_cols])
    inflow_model = HuberRegressor(alpha=alpha)
    inflow_model.fit(X_full, c_train['inflow_next_30d'] / 1000)

    risk_model = CatBoostClassifier(
        iterations=iterations, learning_rate=lr, max_depth=depth,
        auto_class_weights='Balanced', verbose=0, random_state=67
    )
    risk_model.fit(train[risk_cols], train['overdue_next_60d'])

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'inflow_scaler': sc_full, 
        'inflow_model': inflow_model, 
        'risk_model': risk_model,
        'clients_cols': clients_cols, 
        'risk_cols': risk_cols, 
        'target_scale': 1000
    }, artifacts_dir / 'risk_model.joblib')
    
    manifest = {
        "task": "risk",
        "model_file": "risk_model.joblib",
        "feature_columns": risk_cols,
        "effective_config": {"alpha": alpha, "iterations": iterations, "depth": depth, "lr": lr}
    }
    (artifacts_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    print("Risk модель")


def predict_risk(test_dir, artifacts_dir, out_path):
    bundle = joblib.load(artifacts_dir / 'risk_model.joblib')
    c_test = pd.read_csv(test_dir / 'clients_test.csv')
    r_test = pd.read_csv(test_dir / 'credit_test.csv')
    
    pred_in = bundle['inflow_model'].predict(bundle['inflow_scaler'].transform(c_test[bundle['clients_cols']])) * bundle['target_scale']
    test = add_risk_features(r_test, pred_in, c_test['client_id'].values)
    proba = bundle['risk_model'].predict_proba(test[bundle['risk_cols']])[:, 1]
    
    out = pd.DataFrame({'application_id': r_test['application_id'], 'probability_overdue': proba})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("Risk предсказания")