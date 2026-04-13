"""
Part 4: Regression Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

from utils_analysis import (
    load_data, save_figure, save_table, save_report,
    calculate_success_index, print_section_header, logger
)


class RegressionAnalysis:

    def __init__(self, data_path='bilibili_videos_data.csv'):
        self.df = load_data(data_path)
        if 'success_index' not in self.df.columns:
            self.df['success_index'] = calculate_success_index(self.df)
        self.report = {'model_summary': {}, 'coefficients': [], 'diagnostics': {}, 'predictions': {}}

    def analyze_all(self):
        print_section_header("Multiple Regression Analysis")
        try:
            X, y, names = self.prepare_data()
            model, results = self.build_model(X, y, names)
            self.diagnose_model(results)
            self.interpret_coefficients(results, names)
            self.prediction_analysis(results, X, y)
            save_report(self.report, 'regression_analysis_report.json')
            logger.info("✅ Regression analysis completed!")
        except Exception as e:
            logger.error(f"Regression analysis failed: {e}")
            import traceback
            traceback.print_exc()

    def prepare_data(self):
        print_section_header("1. Data Preparation")
        df = self.df.copy()
        for col in ['duration', 'pub_hour', 'success_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['duration', 'pub_hour', 'success_index'])

        continuous = ['duration', 'pub_hour']
        if 'duration_category' not in df.columns:
            df['duration_category'] = pd.cut(
                df['duration'], bins=[0, 180, 600, 1800, np.inf],
                labels=['短视频', '中等', '长视频', '超长视频']
            )
        dur_dum = pd.get_dummies(df['duration_category'], prefix='dur', drop_first=True, dtype=float)
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df.get('pub_dayofweek', pd.Series([0] * len(df))).isin([5, 6]).astype(float)
        weekend = pd.DataFrame({'weekend': df['is_weekend'].astype(float)})

        X_df = pd.concat([
            df[continuous].reset_index(drop=True),
            dur_dum.reset_index(drop=True),
            weekend.reset_index(drop=True)
        ], axis=1)
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        X_df = X_df.fillna(0.0)

        scaler = StandardScaler()
        X_df[continuous] = scaler.fit_transform(X_df[continuous])
        X = X_df.values.astype(np.float64)
        y = df['success_index'].values.astype(np.float64)
        names = X_df.columns.tolist()
        print(f"✅ Prepared: {X.shape[0]} rows × {X.shape[1]} features")
        return X, y, names

    def build_model(self, X, y, names):
        print_section_header("2. Building OLS Model")
        X_c = sm.add_constant(X.astype(np.float64))
        model = sm.OLS(y.astype(np.float64), X_c)
        results = model.fit()
        print(results.summary())
        self.report['model_summary'] = {
            'r_squared': float(results.rsquared),
            'adj_r_squared': float(results.rsquared_adj),
            'f_statistic': float(results.fvalue),
            'f_pvalue': float(results.f_pvalue),
            'n_observations': int(results.nobs)
        }
        return model, results

    def diagnose_model(self, results):
        print_section_header("3. Model Diagnostics")
        residuals = results.resid
        fitted = results.fittedvalues

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].scatter(fitted, residuals, alpha=0.5, s=20, color='steelblue')
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Fitted Values', fontsize=13, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        sm.qqplot(residuals, line='45', ax=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot', fontsize=13, fontweight='bold')

        std_resid = np.sqrt(np.abs(residuals / residuals.std()))
        axes[1, 0].scatter(fitted, std_resid, alpha=0.5, s=20, color='orange')
        axes[1, 0].set_xlabel('Fitted Values', fontsize=11)
        axes[1, 0].set_ylabel('Sqrt(|Standardized Residuals|)', fontsize=11)
        axes[1, 0].set_title('Scale-Location Plot', fontsize=13, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].hist(residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residual Value', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '17_regression_diagnostics.png')

        try:
            from statsmodels.stats.stattools import durbin_watson
            dw = durbin_watson(residuals)
            self.report['diagnostics']['durbin_watson'] = float(dw)
            print(f"Durbin-Watson: {dw:.4f}")
        except Exception:
            pass
        try:
            jb_stat, jb_p = scipy_stats.jarque_bera(residuals)
            self.report['diagnostics']['jarque_bera'] = {
                'statistic': float(jb_stat), 'pvalue': float(jb_p)
            }
            print(f"Jarque-Bera: stat={jb_stat:.4f}, p={jb_p:.6f}")
        except Exception:
            pass

    def interpret_coefficients(self, results, names):
        print_section_header("4. Coefficient Interpretation")
        params = np.array(results.params)
        bse = np.array(results.bse)
        tvalues = np.array(results.tvalues)
        pvalues = np.array(results.pvalues)

        coef_df = pd.DataFrame({
            '特征': ['Intercept'] + names,
            '系数': params, '标准误': bse, 't值': tvalues, 'p值': pvalues
        })
        coef_df['显著性'] = coef_df['p值'].apply(
            lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        )
        save_table(coef_df, 'regression_coefficients.csv')
        self.report['coefficients'] = coef_df.to_dict('records')

        coef_sorted = coef_df.iloc[1:].copy()
        coef_sorted['abs'] = coef_sorted['系数'].abs()
        coef_sorted = coef_sorted.sort_values('abs', ascending=False)

        sig = coef_sorted[coef_sorted['p值'] < 0.05]
        if len(sig) > 0:
            fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.5)))
            colors = ['#e74c3c' if c < 0 else '#27ae60' for c in sig['系数']]
            bars = ax.barh(range(len(sig)), sig['系数'], color=colors, alpha=0.8)
            ax.set_yticks(range(len(sig)))
            ax.set_yticklabels(sig['特征'])
            ax.axvline(0, color='black', linewidth=1)
            ax.set_xlabel('Coefficient Value', fontsize=12)
            ax.set_title('Significant Regression Coefficients (p < 0.05)', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            for i, (bar, val, s) in enumerate(zip(bars, sig['系数'], sig['显著性'])):
                offset = 0.02 if val >= 0 else -0.02
                ha = 'left' if val >= 0 else 'right'
                ax.text(val + offset, i, f'{val:.4f}{s}', va='center', ha=ha, fontsize=9)
            plt.tight_layout()
            save_figure(fig, '18_regression_coefficients.png')

    def prediction_analysis(self, results, X, y):
        print_section_header("5. Prediction Analysis")
        y_pred = results.predict(sm.add_constant(X))
        errors = y - y_pred
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        self.report['predictions'] = {'mae': float(mae), 'rmse': float(rmse)}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.scatter(y, y_pred, alpha=0.5, s=20, color='steelblue')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Fit')
        ax1.set_xlabel('Actual Value', fontsize=12)
        ax1.set_ylabel('Predicted Value', fontsize=12)
        ax1.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Error', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Error Distribution (MAE={mae:.2f}, RMSE={rmse:.2f})',
                      fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '19_prediction_analysis.png')


def main():
    RegressionAnalysis().analyze_all()

if __name__ == "__main__":
    main()
