import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import joblib

PROJECT_DIR = os.path.abspath('.')
DATA_DIR   = os.path.join(PROJECT_DIR, 'Data')
OUT_DIR    = os.path.join(PROJECT_DIR, 'outputs')
FIG_DIR    = os.path.join(OUT_DIR, 'figures')
MODEL_DIR  = os.path.join(OUT_DIR, 'models')
for p in (DATA_DIR, OUT_DIR, FIG_DIR, MODEL_DIR):
    os.makedirs(p, exist_ok=True)
DATASET_PATH = os.path.join(DATA_DIR, 'Dataset.csv')

# column rename mapping
RENAME_DICT = {
    'MagpieData mean Column': 'Column Average',
    'MagpieData range Column': 'Column Span',
    'MagpieData mean GSbandgap': 'Bandgap Average',
    'MagpieData mean Electronegativity': 'Electronegativity Average',
    'MagpieData mean CovalentRadius': 'Covalent Radius Average',
    'MagpieData mean MeltingT': 'Melting Point Average',
    'MagpieData avg_dev MeltingT': 'Melting Point Variability',
    'MagpieData mean NpUnfilled': 'Unfilled p-Orbital Count',
    'MagpieData avg_dev Electronegativity': 'Electronegativity Spread',
    'MagpieData avg_dev NpValence': 'Valence p-Orbital Fluctuation',
    'MagpieData mean GSvolume_pa': 'Atomic Volume Average',
    'MagpieData mean NUnfilled': 'Unfilled Orbital Count',
    'MagpieData avg_dev Column': 'Column Variation',
    'MagpieData range GSvolume_pa': 'Atomic Volume Range',
    'MagpieData avg_dev GSvolume_pa': 'Atomic Volume Fluctuation',
    'MagpieData avg_dev CovalentRadius': 'Covalent Radius Jiggle',
    'MagpieData avg_dev GSbandgap': 'Bandgap Ripple',
    'MagpieData avg_dev NUnfilled': 'Unfilled Orbital Wobble',
    'MagpieData avg_dev MendeleevNumber': 'Mendeleev Number Swirl',
    'MagpieData mean MendeleevNumber': 'Mendeleev Number Average'
}


# Load & clean

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
df.rename(columns=RENAME_DICT, inplace=True)

TARGET = 'Work function (avg. if finite dipole)'
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df.dropna(subset=[TARGET], inplace=True)

X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Models: Random Forest & XGBoost

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=0),
    {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'max_features': ['sqrt', 'log2']
    },
    cv=5, scoring='r2', n_jobs=-1, verbose=2
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print('Best RF parameters:', rf_grid.best_params_)
print('RF R²:', r2_score(y_test, y_pred_rf))
print('RF MAE:', mean_absolute_error(y_test, y_pred_rf))
print('RF RMSE:', root_mean_squared_error(y_test, y_pred_rf))

xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42),
    {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    },
    cv=5, scoring='r2', n_jobs=-1, verbose=2
)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

print('Best XGB parameters:', xgb_grid.best_params_)
print('XGB R²:', r2_score(y_test, y_pred_xgb))
print('XGB MAE:', mean_absolute_error(y_test, y_pred_xgb))
print('XGB RMSE:', root_mean_squared_error(y_test, y_pred_xgb))


# Feature importance (permutation)

perm_rf = permutation_importance(best_rf, X_test, y_test, n_repeats=10, random_state=42)
perm_rf_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm_rf.importances_mean}).sort_values('Importance', ascending=False)
perm_xgb = permutation_importance(best_xgb, X_test, y_test, n_repeats=10, random_state=42)
perm_xgb_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': perm_xgb.importances_mean}).sort_values('Importance', ascending=False)


# Plot helpers

def safe_name(s: str) -> str:
    return s.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

sns.set(style='whitegrid')

# Permutation importance barplots
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=perm_rf_df.head(20), hue='Feature', palette='mako', legend=False)
plt.title('Permutation Importance (Top 20 Features) - Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_permutation_importance.png'), dpi=600)
plt.show()

plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=perm_xgb_df.head(20), hue='Feature', palette='mako', legend=False)
plt.title('Permutation Importance (Top 20 Features) - XGBoost')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xgb_permutation_importance.png'), dpi=600)
plt.show()

# Predicted vs. Actual
plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred_rf, s=60, color='#8f00ff', edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='#a83279', linestyle='--', linewidth=2.5, label='Perfect Prediction')
plt.xlabel('Actual Work Function (eV)')
plt.ylabel('Predicted Work Function (eV)')
plt.title('Random Forest: Predicted vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_predicted_vs_actual.png'), dpi=600)
plt.show()

plt.figure(figsize=(7, 7))
sns.scatterplot(x=y_test, y=y_pred_xgb, s=60, color='#00bcd4', edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='#00695c', linestyle='--', linewidth=2.5, label='Perfect Prediction')
plt.xlabel('Actual Work Function (eV)')
plt.ylabel('Predicted Work Function (eV)')
plt.title('XGBoost: Predicted vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xgb_predicted_vs_actual.png'), dpi=600)
plt.show()

# Top-3 features vs target (from permutation ranking)
for feature in perm_rf_df['Feature'].head(3):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[feature], y=df[TARGET], alpha=0.6, color='#ff00ff', edgecolor='k')
    sns.regplot(x=df[feature], y=df[TARGET], scatter=False, color='black', ci=None, line_kws={'lw': 2})
    plt.xlabel(feature)
    plt.ylabel('Work Function (eV)')
    plt.title(f'{feature} vs Work Function (RF)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"rf_{safe_name(feature)}_vs_workfunction.png"), dpi=600)
    plt.show()

for feature in perm_xgb_df['Feature'].head(3):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[feature], y=df[TARGET], alpha=0.6, color='#00acc1', edgecolor='k')
    sns.regplot(x=df[feature], y=df[TARGET], scatter=False, color='black', ci=None, line_kws={'lw': 2})
    plt.xlabel(feature)
    plt.ylabel('Work Function (eV)')
    plt.title(f'{feature} vs Work Function (XGB)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"xgb_{safe_name(feature)}_vs_workfunction.png"), dpi=600)
    plt.show()

# Correlation heatmaps among top-10 features
corr_rf = df[perm_rf_df['Feature'].head(10)].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_rf, annot=True, cmap='rocket', fmt='.2f', square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Between Top 10 Features (RF)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_top_feature_correlation_heatmap.png'), dpi=600)
plt.show()

corr_xgb = df[perm_xgb_df['Feature'].head(10)].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr_xgb, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Between Top 10 Features (XGB)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xgb_top_feature_correlation_heatmap.png'), dpi=600)
plt.show()


# SHAP analyses

explainer_rf = shap.TreeExplainer(best_rf)
shap_values_rf = explainer_rf.shap_values(X_test)
shap.summary_plot(shap_values_rf, X_test, feature_names=X_test.columns, show=False)
plt.title('SHAP Summary Plot (Random Forest)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'rf_shap_summary.png'), dpi=600)
plt.show()

shap_imp_rf = np.abs(shap_values_rf).mean(axis=0)
shap_df_rf = pd.DataFrame({'Feature': X_test.columns, 'Importance': shap_imp_rf}).sort_values('Importance', ascending=False)
for feature in shap_df_rf['Feature'].head(3):
    shap.dependence_plot(feature, shap_values_rf, X_test, show=False)
    plt.title(f'SHAP Dependence Plot: {feature} (RF)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"rf_shap_dependence_{safe_name(feature)}.png"), dpi=600)
    plt.show()

explainer_xgb = shap.Explainer(best_xgb, X_test)
shap_values_xgb = explainer_xgb(X_test)
shap.summary_plot(shap_values_xgb, X_test, show=False)
plt.title('SHAP Summary Plot (XGBoost)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'xgb_shap_summary.png'), dpi=600)
plt.show()

shap_imp_xgb = np.abs(shap_values_xgb.values).mean(axis=0)
shap_df_xgb = pd.DataFrame({'Feature': X_test.columns, 'Importance': shap_imp_xgb}).sort_values('Importance', ascending=False)
for feature in shap_df_xgb['Feature'].head(3):
    shap.dependence_plot(feature, shap_values_xgb.values, X_test, show=False)
    plt.title(f'SHAP Dependence Plot: {feature} (XGB)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"xgb_shap_dependence_{safe_name(feature)}.png"), dpi=600)
    plt.show()


# Save models

joblib.dump(best_rf, os.path.join(MODEL_DIR, 'best_random_forest_model.pkl'))
joblib.dump(best_xgb, os.path.join(MODEL_DIR, 'best_xgboost_model.pkl'))

print('\nSaved figures to:', FIG_DIR)
print('Saved models  to:', MODEL_DIR)