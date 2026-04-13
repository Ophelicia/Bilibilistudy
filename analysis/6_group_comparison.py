"""
Part 6: Group Comparison Analysis (English Labels)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.preprocessing import StandardScaler

from utils_analysis import (
    load_data, save_figure, save_table, save_report,
    calculate_success_index, print_section_header, logger,
    translate_dataframe
)


class GroupComparison:

    def __init__(self, data_path='bilibili_videos_data.csv'):
        self.df = load_data(data_path, translate=True)
        try:
            clustered = pd.read_csv('results/tables/videos_with_clusters.csv')
            if 'cluster_name' in clustered.columns and 'cluster' in clustered.columns:
                self.df = self.df.merge(
                    clustered[['bvid', 'cluster', 'cluster_name']], on='bvid', how='left'
                )
        except FileNotFoundError:
            self.df['cluster'] = 0
            self.df['cluster_name'] = 'All Videos'
        if 'success_index' not in self.df.columns:
            self.df['success_index'] = calculate_success_index(self.df)
        self.report = {'group_statistics': [], 'group_comparisons': []}

    def analyze_all(self):
        print_section_header("Group Comparison Analysis")
        if self.df['cluster'].nunique() <= 1:
            print("⚠️ No cluster labels detected, skipping")
            return
        self.compare_basic()
        self.compare_region()
        self.compare_time()
        self.compare_duration()
        self.create_heatmap()
        save_report(self.report, 'group_comparison_report.json')
        logger.info("✅ Group comparison analysis completed!")

    def _get_cluster_labels(self):
        names = sorted(self.df['cluster_name'].unique())
        short = [f'C{i}' for i in range(len(names))]
        return names, short

    def compare_basic(self):
        print_section_header("1. Basic Group Statistics")
        names, short = self._get_cluster_labels()
        stats_list = []

        for i, name in enumerate(names):
            cd = self.df[self.df['cluster_name'] == name]
            stats_list.append({
                'Group': f'C{i}: {name}',
                'Count': len(cd),
                'Pct': f"{len(cd) / len(self.df) * 100:.2f}%",
                'Avg_Views': int(cd['view'].mean()),
                'Avg_Like_Rate': f"{cd['like_rate'].mean() * 100:.2f}%",
                'Avg_Coin_Rate': f"{cd['coin_rate'].mean() * 100:.2f}%",
                'Avg_Fav_Rate': f"{cd['favorite_rate'].mean() * 100:.2f}%",
                'Avg_Engagement': f"{cd['engagement_rate'].mean() * 100:.2f}%",
                'Avg_Success': round(cd['success_index'].mean(), 2)
            })
        self.report['group_statistics'] = stats_list
        save_table(pd.DataFrame(stats_list), 'group_basic_statistics.csv')

        # Boxplot
        fig, ax = plt.subplots(figsize=(12, 6))
        data = [self.df[self.df['cluster_name'] == n]['success_index'].values for n in names]
        box_labels = [f'C{i}: {n}' for i, n in enumerate(names)]
        bp = ax.boxplot(data, labels=box_labels, patch_artist=True, widths=0.6)
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']
        for patch, c in zip(bp['boxes'], colors[:len(names)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        means = [np.mean(d) for d in data]
        ax.scatter(range(1, len(means) + 1), means, color='black', marker='D',
                   s=60, zorder=5, label='Mean')
        ax.set_ylabel('Success Index', fontsize=12)
        ax.set_title('Success Index Distribution by User Group', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        save_figure(fig, '25_group_success_comparison.png')

        # Multi-metric bar
        fig, ax = plt.subplots(figsize=(14, 6))
        metrics = ['like_rate', 'coin_rate', 'favorite_rate', 'engagement_rate']
        metric_labels = ['Like Rate', 'Coin Rate', 'Favorite Rate', 'Engagement Rate']
        x = np.arange(len(metric_labels))
        width = 0.8 / len(names)

        for i, name in enumerate(names):
            cd = self.df[self.df['cluster_name'] == name]
            vals = [cd[m].mean() * 100 for m in metrics]
            bars = ax.bar(x + i * width - width * len(names) / 2, vals, width,
                          label=f'C{i}: {name}', color=colors[i % len(colors)], alpha=0.8)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel('Rate (%)', fontsize=12)
        ax.set_title('Interaction Rates Comparison by Cluster', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_figure(fig, '26_group_interaction_comparison.png')

    def compare_region(self):
        print_section_header("2. Category Performance by Group")
        names, short = self._get_cluster_labels()

        pivot = self.df.pivot_table(
            values='success_index', index='tname',
            columns='cluster_name', aggfunc='mean'
        )
        # Rename columns to short labels
        col_map = {n: f'C{i}: {n}' for i, n in enumerate(names)}
        pivot = pivot.rename(columns=col_map)
        save_table(pivot.round(2), 'group_region_performance.csv')

        fig, ax = plt.subplots(figsize=(14, 8))
        top_regions = self.df['tname'].value_counts().head(15).index
        pivot_top = pivot.loc[pivot.index.isin(top_regions)]
        sns.heatmap(pivot_top, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, linecolor='white')
        ax.set_xlabel('User Group', fontsize=12)
        ax.set_ylabel('Video Category', fontsize=12)
        ax.set_title('Average Success Index by Category and Group',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, '27_group_region_heatmap.png')

    def compare_time(self):
        print_section_header("3. Publishing Time by Group")
        if 'pub_hour' not in self.df.columns:
            return
        names, short = self._get_cluster_labels()
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        for i, name in enumerate(names):
            cd = self.df[self.df['cluster_name'] == name]
            hour_success = cd.groupby('pub_hour')['success_index'].mean()
            ax1.plot(hour_success.index, hour_success.values, marker='o', linewidth=2,
                     label=f'C{i}: {name}', color=colors[i % len(colors)], markersize=5)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Success Index', fontsize=12)
        ax1.set_title('Success Index by Publishing Hour', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24))
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        for i, name in enumerate(names):
            cd = self.df[self.df['cluster_name'] == name]
            hour_count = cd['pub_hour'].value_counts().sort_index()
            ax2.plot(hour_count.index, hour_count.values, marker='s', linewidth=2,
                     label=f'C{i}: {name}', color=colors[i % len(colors)], markersize=5)
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Number of Videos', fontsize=12)
        ax2.set_title('Publishing Volume by Hour', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(0, 24))
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '28_group_publishing_time.png')

    def compare_duration(self):
        print_section_header("4. Duration Preference by Group")
        if 'duration_category' not in self.df.columns:
            return
        names, short = self._get_cluster_labels()
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

        pivot = self.df.pivot_table(
            values='success_index', index='duration_category',
            columns='cluster_name', aggfunc='mean'
        )
        dur_order = ['Short (<3min)', 'Medium (3-10min)', 'Long (10-30min)', 'Extra Long (>30min)']
        pivot = pivot.reindex([o for o in dur_order if o in pivot.index])

        col_map = {n: f'C{i}: {n}' for i, n in enumerate(names)}
        pivot = pivot.rename(columns=col_map)
        save_table(pivot.round(2), 'group_duration_preference.csv')

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(pivot))
        width = 0.8 / len(names)

        for i, col in enumerate(pivot.columns):
            vals = pivot[col].values
            bars = ax.bar(x + i * width - width * len(names) / 2, vals, width,
                          label=col, color=colors[i % len(colors)], alpha=0.8)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                            f'{v:.1f}', ha='center', va='bottom', fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=15, ha='right')
        ax.set_xlabel('Video Duration Category', fontsize=12)
        ax.set_ylabel('Average Success Index', fontsize=12)
        ax.set_title('Success Index by Duration Category and Group',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_figure(fig, '29_group_duration_preference.png')

    def create_heatmap(self):
        print_section_header("5. Comprehensive Group Comparison")
        names, short = self._get_cluster_labels()
        data = []

        for i, name in enumerate(names):
            cd = self.df[self.df['cluster_name'] == name]
            data.append({
                'Group': f'C{i}: {name}',
                'Avg Views': cd['view'].mean(),
                'Like Rate (%)': cd['like_rate'].mean() * 100,
                'Coin Rate (%)': cd['coin_rate'].mean() * 100,
                'Favorite Rate (%)': cd['favorite_rate'].mean() * 100,
                'Danmaku Rate (%)': cd['danmaku_rate'].mean() * 100,
                'Comment Rate (%)': cd['reply_rate'].mean() * 100,
                'Engagement (%)': cd['engagement_rate'].mean() * 100,
                'Success Index': cd['success_index'].mean()
            })

        comp_df = pd.DataFrame(data).set_index('Group')

        # Standardized heatmap
        scaler = StandardScaler()
        comp_norm = pd.DataFrame(
            scaler.fit_transform(comp_df),
            index=comp_df.index, columns=comp_df.columns
        )

        fig, ax = plt.subplots(figsize=(14, max(6, len(comp_norm) * 1.2)))
        sns.heatmap(comp_norm.T, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, linewidths=0.8, linecolor='white', ax=ax,
                    cbar_kws={'label': 'Standardized Score'})
        ax.set_xlabel('User Group', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        ax.set_title('Comprehensive Group Comparison (Standardized)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, '30_group_comprehensive_heatmap.png')
        save_table(comp_df.round(2), 'group_comprehensive_comparison.csv')

        # Radar comparison
        self._plot_group_radar(comp_df)

    def _plot_group_radar(self, comp_df):
        categories = [c for c in comp_df.columns if c != 'Avg Views']
        n_cats = len(categories)
        angles = [n / float(n_cats) * 2 * pi for n in range(n_cats)]
        angles += angles[:1]

        norm_df = comp_df[categories].copy()
        for col in norm_df.columns:
            col_min, col_max = norm_df[col].min(), norm_df[col].max()
            if col_max > col_min:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.5

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

        for i, (group, row) in enumerate(norm_df.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2,
                    label=group, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.12, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        short_labels = [c.replace(' (%)', '').replace('Rate', 'R.') for c in categories]
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_title('Group Feature Comparison (Radar Chart)',
                      fontsize=14, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax.grid(True)
        plt.tight_layout()
        save_figure(fig, '31_group_radar_comparison.png')


def main():
    GroupComparison().analyze_all()

if __name__ == "__main__":
    main()
