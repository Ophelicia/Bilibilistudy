"""
Part 5: Machine Learning Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from utils_analysis import (
    load_data, save_figure, save_table, save_report,
    calculate_success_index, print_section_header, logger
)


class MachineLearningAnalysis:

    def __init__(self, data_path='bilibili_videos_data.csv'):
        self.df = load_data(data_path)
        if 'success_index' not in self.df.columns:
            self.df['success_index'] = calculate_success_index(self.df)
        self.report = {'model_comparison': [], 'feature_importance': {}, 'shap_analysis': {}}

    def analyze_all(self):
        print_section_header("Machine Learning Analysis")
        X_train, X_test, y_train, y_test, names = self.prepare_data()
        models = self.train_models(X_train, X_test, y_train, y_test)
        best_name, best_model = self.select_best(models)
        self.feature_importance(best_model, names, best_name)
        self.shap_analysis(best_model, X_test, names, best_name)
        self.prediction_visualization(models, X_test, y_test, names)
        save_report(self.report, 'ml_analysis_report.json')
        logger.info("✅ Machine learning analysis completed!")

    def prepare_data(self):
        print_section_header("1. Feature Engineering")
        df = self.df.copy()
        continuous = ['duration', 'pub_hour', 'like_rate', 'coin_rate',
                      'favorite_rate', 'share_rate', 'danmaku_rate', 'reply_rate']
        for col in continuous:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'tname' in df.columns:
            top = df['tname'].value_counts().head(10).index.tolist()
            df['region_encoded'] = df['tname'].apply(lambda x: top.index(x) if x in top else -1)
        else:
            df['region_encoded'] = 0

        dur_map = {'短视频': 0, '中等': 1, '长视频': 2, '超长视频': 3}
        df['duration_category_encoded'] = df.get(
            'duration_category', pd.Series(['中等'] * len(df))
        ).map(dur_map).fillna(1)
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = 0
        if 'pub_dayofweek' not in df.columns:
            df['pub_dayofweek'] = 0

        cat_features = ['region_encoded', 'duration_category_encoded', 'is_weekend', 'pub_dayofweek']
        all_features = continuous + cat_features
        for f in all_features:
            if f not in df.columns:
                df[f] = 0

        X = np.nan_to_num(df[all_features].values, nan=0.0)
        y = np.nan_to_num(df['success_index'].values, nan=0.0)
        valid = y > 0
        X, y = X[valid], y[valid]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        n = len(continuous)
        X_train[:, :n] = scaler.fit_transform(X_train[:, :n])
        X_test[:, :n] = scaler.transform(X_test[:, :n])

        print(f"✅ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")
        return X_train, X_test, y_train, y_test, all_features

    def train_models(self, X_train, X_test, y_train, y_test):
        print_section_header("2. Model Training & Evaluation")
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
        }
        results = []
        trained = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            trained[name] = model
            y_tr = model.predict(X_train)
            y_te = model.predict(X_test)
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
            r = {
                'model': name,
                'train_r2': round(r2_score(y_train, y_tr), 4),
                'test_r2': round(r2_score(y_test, y_te), 4),
                'train_rmse': round(np.sqrt(mean_squared_error(y_train, y_tr)), 4),
                'test_rmse': round(np.sqrt(mean_squared_error(y_test, y_te)), 4),
                'train_mae': round(mean_absolute_error(y_train, y_tr), 4),
                'test_mae': round(mean_absolute_error(y_test, y_te), 4),
                'cv_r2_mean': round(cv.mean(), 4),
                'cv_r2_std': round(cv.std(), 4)
            }
            results.append(r)
            print(f"  {name}: Train R²={r['train_r2']}, Test R²={r['test_r2']}, "
                  f"CV={r['cv_r2_mean']}±{r['cv_r2_std']}")

        self.report['model_comparison'] = results
        save_table(pd.DataFrame(results), 'model_comparison.csv')

        # Visualization: Model comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        x_pos = np.arange(len(results))
        w = 0.35
        model_names = [r['model'] for r in results]

        axes[0].bar(x_pos - w / 2, [r['train_r2'] for r in results], w, label='Train', color='#3498db')
        axes[0].bar(x_pos + w / 2, [r['test_r2'] for r in results], w, label='Test', color='#e74c3c')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(model_names, rotation=15)
        axes[0].set_ylabel('R² Score', fontsize=12)
        axes[0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].bar(x_pos - w / 2, [r['train_rmse'] for r in results], w, label='Train', color='#3498db')
        axes[1].bar(x_pos + w / 2, [r['test_rmse'] for r in results], w, label='Test', color='#e74c3c')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(model_names, rotation=15)
        axes[1].set_ylabel('RMSE', fontsize=12)
        axes[1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        axes[2].bar(x_pos, [r['cv_r2_mean'] for r in results], color='#27ae60', alpha=0.8)
        axes[2].errorbar(x_pos, [r['cv_r2_mean'] for r in results],
                         yerr=[r['cv_r2_std'] for r in results], fmt='none', ecolor='black', capsize=5)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(model_names, rotation=15)
        axes[2].set_ylabel('R² Score', fontsize=12)
        axes[2].set_title('Cross-Validation R² (5-Fold)', fontsize=14, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '20_model_comparison.png')
        return trained

    def select_best(self, models):
        best = max(self.report['model_comparison'], key=lambda x: x['test_r2'])
        print(f"✅ Best model: {best['model']} (Test R²={best['test_r2']})")
        return best['model'], models[best['model']]

    def feature_importance(self, model, names, model_name):
        print_section_header("4. Feature Importance")
        if not hasattr(model, 'feature_importances_'):
            return
        imp = model.feature_importances_
        imp_df = pd.DataFrame({'feature': names, 'importance': imp}).sort_values('importance', ascending=False)
        save_table(imp_df, 'feature_importance.csv')
        self.report['feature_importance'] = {'model': model_name, 'features': imp_df.to_dict('records')}

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp_df)))
        ax.barh(imp_df['feature'], imp_df['importance'], color=colors, alpha=0.85)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Feature Importance ({model_name})', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (feat, val) in enumerate(zip(imp_df['feature'], imp_df['importance'])):
            ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        save_figure(fig, '21_feature_importance.png')

    def shap_analysis(self, model, X_test, names, model_name):
        print_section_header("5. SHAP Analysis")
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            sample_size = min(500, X_test.shape[0])
            X_sample = X_test[:sample_size]
            shap_values = explainer.shap_values(X_sample)

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=names, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance', fontweight='bold', fontsize=14, pad=20)
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.tight_layout()
            save_figure(fig, '22_shap_summary_bar.png')

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=names, show=False)
            plt.title('SHAP Value Distribution', fontweight='bold', fontsize=14, pad=20)
            plt.xlabel('SHAP Value (Impact on Output)', fontsize=12)
            plt.tight_layout()
            save_figure(fig, '23_shap_summary_scatter.png')

            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_df = pd.DataFrame({
                'feature': names, 'mean_abs_shap': mean_shap
            }).sort_values('mean_abs_shap', ascending=False)
            save_table(shap_df, 'shap_importance.csv')
            self.report['shap_analysis'] = {
                'model': model_name, 'sample_size': sample_size,
                'features': shap_df.to_dict('records')
            }
        except ImportError:
            logger.warning("SHAP not installed, skipping SHAP analysis")
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")

    def prediction_visualization(self, models, X_test, y_test, names):
        print_section_header("6. Prediction Visualization")
        best_info = max(self.report['model_comparison'], key=lambda x: x['test_r2'])
        best_model = models[best_info['model']]
        y_pred = best_model.predict(X_test)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(y_test, y_pred, alpha=0.5, s=25, color='steelblue', edgecolors='black', linewidth=0.3)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Perfect Fit')
        r2 = r2_score(y_test, y_pred)
        ax1.set_xlabel('Actual Success Index', fontsize=12)
        ax1.set_ylabel('Predicted Success Index', fontsize=12)
        ax1.set_title(f'Actual vs Predicted ({best_info["model"]}, R²={r2:.4f})',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        errors = y_test - y_pred
        ax2.hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        ax2.set_xlabel('Prediction Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Error Distribution (MAE={mae:.2f}, RMSE={rmse:.2f})',
                      fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '24_ml_prediction_analysis.png')


def main():
    MachineLearningAnalysis().analyze_all()

if __name__ == "__main__":
    main()
