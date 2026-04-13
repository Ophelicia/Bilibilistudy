"""
Part 3: Univariate Analysis of Success Factors (English Labels)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import f_oneway, kruskal

from utils_analysis import (
    load_data, save_figure, save_table, save_report,
    calculate_success_index, print_section_header, logger
)


class UnivariateAnalysis:

    def __init__(self, data_path='bilibili_videos_data.csv'):
        self.df = load_data(data_path, translate=True)
        self.df['success_index'] = calculate_success_index(self.df)
        self.report = {
            'success_index_stats': {}, 'categorical_analysis': [],
            'continuous_analysis': [], 'significant_factors': []
        }

    def analyze_all(self):
        print_section_header("Univariate Analysis")
        self.analyze_success_distribution()
        self.analyze_categorical_factors()
        self.analyze_continuous_factors()
        self.summarize_significant_factors()
        save_report(self.report, 'univariate_analysis_report.json')
        logger.info("✅ Univariate analysis completed!")

    def analyze_success_distribution(self):
        print_section_header("Success Index Distribution")
        s = self.df['success_index']
        self.report['success_index_stats'] = {
            'mean': float(s.mean()), 'median': float(s.median()),
            'std': float(s.std()), 'min': float(s.min()), 'max': float(s.max()),
            'q25': float(s.quantile(0.25)), 'q75': float(s.quantile(0.75))
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.hist(s, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(s.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {s.mean():.2f}')
        ax1.axvline(s.median(), color='green', linestyle='--', linewidth=2,
                    label=f'Median: {s.median():.2f}')
        ax1.set_xlabel('Success Index', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Success Index', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        ax2.boxplot(s, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Success Index', fontsize=12)
        ax2.set_title('Success Index Boxplot', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig, '13_success_index_distribution.png')

    def analyze_categorical_factors(self):
        print_section_header("Categorical Variable Analysis")

        variables = [
            ('tname', 'Category'),
            ('duration_category', 'Duration Type'),
            ('pub_hour', 'Publishing Hour'),
            ('pub_dayofweek', 'Day of Week'),
            ('is_weekend', 'Weekend'),
        ]

        results = []
        for var, name_en in variables:
            if var not in self.df.columns:
                continue
            groups = [g['success_index'].values for _, g in self.df.groupby(var)]
            if len(groups) < 2:
                continue
            levene_stat, levene_p = stats.levene(*groups)
            if levene_p > 0.05:
                f_stat, p_val = f_oneway(*groups)
                test_name = 'ANOVA'
            else:
                f_stat, p_val = kruskal(*groups)
                test_name = 'Kruskal-Wallis'

            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            results.append({
                'Variable': name_en, 'Test': test_name,
                'Statistic': round(f_stat, 4), 'P-value': round(p_val, 6),
                'Significance': sig, 'N_Groups': len(groups)
            })
            print(f"  {name_en}: {test_name} stat={f_stat:.4f}, p={p_val:.6f} ({sig})")

        self.report['categorical_analysis'] = results
        save_table(pd.DataFrame(results), 'categorical_factors_analysis.csv')

        # Category vs Success Index (Top 15)
        if 'tname' in self.df.columns:
            top_regions = self.df['tname'].value_counts().head(15).index
            filtered = self.df[self.df['tname'].isin(top_regions)]
            region_mean = filtered.groupby('tname')['success_index'].mean().sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(12, 7))
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(region_mean)))
            ax.barh(range(len(region_mean)), region_mean.values, color=colors)
            ax.set_yticks(range(len(region_mean)))
            ax.set_yticklabels(region_mean.index, fontsize=10)
            ax.set_xlabel('Average Success Index', fontsize=12)
            ax.set_title('Average Success Index by Category (Top 15)',
                          fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            for i, v in enumerate(region_mean.values):
                ax.text(v + 0.2, i, f'{v:.2f}', va='center', fontsize=9)
            plt.tight_layout()
            save_figure(fig, '14_category_success_index.png')

        # Duration vs Success
        if 'duration_category' in self.df.columns:
            dur_order = ['Short (<3min)', 'Medium (3-10min)', 'Long (10-30min)', 'Extra Long (>30min)']
            dur_data = []
            for d in dur_order:
                subset = self.df[self.df['duration_category'] == d]
                if len(subset) > 0:
                    dur_data.append((d, subset['success_index'].mean()))
            if dur_data:
                dur_labels, dur_means = zip(*dur_data)
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(dur_labels, dur_means,
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(dur_data)])
                ax.set_ylabel('Average Success Index', fontsize=12)
                ax.set_xlabel('Duration Category', fontsize=12)
                ax.set_title('Success Index by Video Duration', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                for bar, m in zip(bars, dur_means):
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                            f'{m:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                plt.xticks(rotation=15, ha='right')
                plt.tight_layout()
                save_figure(fig, '15_duration_success_index.png')

    def analyze_continuous_factors(self):
        print_section_header("Continuous Variable Analysis")

        variables = [
            ('duration', 'Duration'),
            ('like_rate', 'Like Rate'),
            ('coin_rate', 'Coin Rate'),
            ('favorite_rate', 'Favorite Rate'),
            ('engagement_rate', 'Engagement Rate'),
            ('completion_rate_proxy', 'Completion Proxy'),
        ]

        results = []
        for var, name_en in variables:
            if var not in self.df.columns:
                continue
            clean = self.df[[var, 'success_index']].dropna()
            if len(clean) < 3:
                continue
            pr, pp = stats.pearsonr(clean[var], clean['success_index'])
            sr, sp = stats.spearmanr(clean[var], clean['success_index'])
            sig = '***' if min(pp, sp) < 0.001 else '**' if min(pp, sp) < 0.01 else '*' if min(pp, sp) < 0.05 else 'ns'
            results.append({
                'Variable': name_en, 'Pearson_r': round(pr, 4), 'Pearson_p': round(pp, 6),
                'Spearman_r': round(sr, 4), 'Spearman_p': round(sp, 6),
                'Significance': sig, 'Abs_Correlation': round(abs(pr), 4)
            })
            print(f"  {name_en}: Pearson r={pr:.4f}, p={pp:.6f} ({sig})")

        self.report['continuous_analysis'] = results
        save_table(pd.DataFrame(results).sort_values('Abs_Correlation', ascending=False),
                   'continuous_factors_analysis.csv')

        # Scatter plots
        plot_vars = [(v, n) for v, n in variables if v in self.df.columns]
        n_plots = len(plot_vars)
        if n_plots == 0:
            return
        ncols = min(3, n_plots)
        nrows = (n_plots + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (var, name_en) in enumerate(plot_vars):
            ax = axes[idx]
            clean = self.df[[var, 'success_index']].dropna()
            ax.scatter(clean[var], clean['success_index'], alpha=0.3, s=20, color='steelblue')
            z = np.polyfit(clean[var], clean['success_index'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(clean[var].min(), clean[var].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.8)
            r_val = [r for r in results if r['Variable'] == name_en]
            r_text = f"r={r_val[0]['Pearson_r']:.3f}" if r_val else ""
            ax.set_xlabel(name_en, fontsize=11)
            ax.set_ylabel('Success Index', fontsize=11)
            ax.set_title(f'{name_en} vs Success Index ({r_text})',
                          fontsize=12, fontweight='bold')
            ax.grid(alpha=0.3)

        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])
        plt.tight_layout()
        save_figure(fig, '16_continuous_factors_scatter.png')

    def summarize_significant_factors(self):
        sig = []
        for r in self.report['categorical_analysis']:
            if r['P-value'] < 0.05:
                sig.append({
                    'Factor': r['Variable'], 'Type': 'Categorical',
                    'P-value': r['P-value'], 'Significance': r['Significance']
                })
        for r in self.report['continuous_analysis']:
            if r['Pearson_p'] < 0.05:
                sig.append({
                    'Factor': r['Variable'], 'Type': 'Continuous',
                    'P-value': r['Pearson_p'], 'Correlation': r['Pearson_r'],
                    'Significance': r['Significance']
                })
        self.report['significant_factors'] = sig
        if sig:
            save_table(pd.DataFrame(sig).sort_values('P-value'), 'significant_factors_summary.csv')
        print(f"\n✅ Significant factors (p<0.05): {len(sig)}")


def main():
    UnivariateAnalysis().analyze_all()

if __name__ == "__main__":
    main()
