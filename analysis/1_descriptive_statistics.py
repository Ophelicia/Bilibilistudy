"""
Part 1: Descriptive Statistics Analysis (English Labels)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils_analysis import (
    load_data, save_figure, save_table, save_report,
    get_basic_stats, format_large_number, print_section_header, logger,
    DURATION_CN_TO_EN
)


class DescriptiveStatistics:

    def __init__(self, data_path='bilibili_videos_data.csv'):
        self.df = load_data(data_path, translate=True)
        self.report = {
            'data_info': {}, 'basic_stats': {},
            'distribution_analysis': {}, 'correlation_analysis': {}
        }

    def analyze_all(self):
        print_section_header("Descriptive Statistics Analysis")
        self.analyze_data_info()
        self.analyze_basic_features()
        self.analyze_interaction_stats()
        self.analyze_time_features()
        self.analyze_correlation()
        save_report(self.report, 'descriptive_statistics_report.json')
        logger.info("✅ Descriptive statistics analysis completed!")

    def analyze_data_info(self):
        print_section_header("1. Dataset Overview")
        info = {
            'total_videos': len(self.df),
            'total_columns': len(self.df.columns),
            'unique_regions': self.df['tname'].nunique(),
            'date_range': {
                'start': self.df['pubdate_formatted'].min(),
                'end': self.df['pubdate_formatted'].max()
            },
            'missing_values': self.df.isnull().sum().to_dict()
        }
        self.report['data_info'] = info
        print(f"Total videos: {info['total_videos']}, Categories: {info['unique_regions']}")

    def analyze_basic_features(self):
        print_section_header("2. Basic Feature Analysis")

        # Region distribution (already translated)
        region_dist = self.df['tname'].value_counts()
        region_pct = (region_dist / len(self.df) * 100).round(2)
        region_summary = pd.DataFrame({
            'Category': region_dist.index,
            'Count': region_dist.values,
            'Percentage(%)': region_pct.values
        })
        save_table(region_summary, 'region_distribution.csv')
        self.report['basic_stats']['region_distribution'] = region_summary.to_dict('records')

        # Duration distribution (already translated)
        duration_dist = self.df['duration_category'].value_counts()
        self.report['basic_stats']['duration_distribution'] = duration_dist.to_dict()

        # === Figure 01: Region distribution ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        top15 = region_dist.head(15)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top15)))

        ax1.pie(top15.values, labels=top15.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 9})
        ax1.set_title('Video Category Distribution (Top 15)', fontsize=14, fontweight='bold')

        ax2.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
        ax2.set_xlabel('Number of Videos', fontsize=12)
        ax2.set_title('Number of Videos per Category (Top 15)', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        for i, v in enumerate(top15.values[::-1]):
            ax2.text(v + 0.5, i, str(v), va='center', fontsize=9)

        plt.tight_layout()
        save_figure(fig, '01_region_distribution.png')

        # === Figure 02: Duration distribution ===
        fig, ax = plt.subplots(figsize=(10, 6))
        dur_order = ['Short (<3min)', 'Medium (3-10min)', 'Long (10-30min)', 'Extra Long (>30min)']
        dur_counts = [duration_dist.get(cat, 0) for cat in dur_order]

        bars = ax.bar(dur_order, dur_counts,
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('Number of Videos', fontsize=12)
        ax.set_xlabel('Duration Category', fontsize=12)
        ax.set_title('Video Duration Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h, f'{int(h)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.tight_layout()
        save_figure(fig, '02_duration_distribution.png')

    def analyze_interaction_stats(self):
        print_section_header("3. Interaction Metrics Statistics")

        cols = ['view', 'like', 'coin', 'favorite', 'share', 'danmaku', 'reply']
        names = ['Views', 'Likes', 'Coins', 'Favorites', 'Shares', 'Danmaku', 'Comments']
        stats_summary = []
        for col, name in zip(cols, names):
            if col in self.df.columns:
                s = get_basic_stats(self.df[col])
                stats_summary.append({
                    'Metric': name,
                    'Min': format_large_number(s['min']),
                    'Mean': format_large_number(s['mean']),
                    'Median': format_large_number(s['median']),
                    'Max': format_large_number(s['max']),
                    'Std': format_large_number(s['std'])
                })
        save_table(pd.DataFrame(stats_summary), 'interaction_statistics.csv')
        self.report['distribution_analysis']['interaction_stats'] = stats_summary

        # === Figure 04: Boxplot ===
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        for idx, (col, name) in enumerate(zip(cols, names)):
            if col in self.df.columns and idx < len(axes):
                data_log = np.log10(self.df[col] + 1)
                bp = axes[idx].boxplot(data_log, vert=True, patch_artist=True,
                                       boxprops=dict(facecolor='lightblue', alpha=0.7))
                axes[idx].set_ylabel('log10(value+1)', fontsize=10)
                axes[idx].set_title(name, fontsize=12, fontweight='bold')
                axes[idx].grid(axis='y', alpha=0.3)
                mean_val = self.df[col].mean()
                axes[idx].text(0.5, 0.95, f'Mean: {format_large_number(mean_val)}',
                               transform=axes[idx].transAxes, ha='center', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
        if len(cols) < len(axes):
            fig.delaxes(axes[-1])
        plt.suptitle('Interaction Metrics Distribution (Log Scale)',
                      fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        save_figure(fig, '04_interaction_boxplot.png')

        # Interaction rates
        rate_cols = ['like_rate', 'coin_rate', 'favorite_rate', 'share_rate',
                     'danmaku_rate', 'reply_rate', 'engagement_rate']
        rate_names = ['Like Rate', 'Coin Rate', 'Favorite Rate', 'Share Rate',
                      'Danmaku Rate', 'Comment Rate', 'Engagement Rate']
        rate_summary = []
        for col, name in zip(rate_cols, rate_names):
            if col in self.df.columns:
                s = get_basic_stats(self.df[col])
                rate_summary.append({
                    'Metric': name,
                    'Mean': f"{s['mean'] * 100:.2f}%",
                    'Median': f"{s['median'] * 100:.2f}%",
                    'Max': f"{s['max'] * 100:.2f}%"
                })
        save_table(pd.DataFrame(rate_summary), 'interaction_rates.csv')
        self.report['distribution_analysis']['interaction_rates'] = rate_summary

        # === Figure 05: Interaction rates bar ===
        fig, ax = plt.subplots(figsize=(12, 6))
        means = [self.df[col].mean() * 100 for col in rate_cols if col in self.df.columns]
        available_names = [n for c, n in zip(rate_cols, rate_names) if c in self.df.columns]
        bars = ax.bar(range(len(available_names)), means, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(available_names)))
        ax.set_xticklabels(available_names, rotation=30, ha='right')
        ax.set_ylabel('Average Rate (%)', fontsize=12)
        ax.set_title('Average Interaction Rates', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{m:.2f}%', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        save_figure(fig, '05_interaction_rates.png')

    def analyze_time_features(self):
        print_section_header("4. Temporal Feature Analysis")
        if 'pub_hour' not in self.df.columns:
            return

        hour_dist = self.df['pub_hour'].value_counts().sort_index()
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_dist = self.df['pub_dayofweek'].value_counts().sort_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Hourly
        bars1 = ax1.bar(hour_dist.index, hour_dist.values, color='coral', alpha=0.7)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Videos', fontsize=12)
        ax1.set_title('Publishing Time Distribution (24-Hour)', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24))
        ax1.grid(alpha=0.3)
        peak_hour = hour_dist.idxmax()
        ax1.axvline(peak_hour, color='red', linestyle='--', alpha=0.7,
                     label=f'Peak: {peak_hour}:00')
        ax1.legend(fontsize=10)

        # Day of week
        day_x = [day_labels[i] for i in day_dist.index]
        bars2 = ax2.bar(day_x, day_dist.values, color='skyblue', alpha=0.7)
        for i, bar in enumerate(bars2):
            if day_dist.index[i] in [5, 6]:
                bar.set_color('#f39c12')
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Number of Videos', fontsize=12)
        ax2.set_title('Day of Week Distribution (Orange = Weekend)', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '06_time_distribution.png')

        weekend_count = self.df['is_weekend'].sum()
        self.report['distribution_analysis']['time_features'] = {
            'hour_distribution': hour_dist.to_dict(),
            'day_distribution': {day_labels[i]: int(v) for i, v in zip(day_dist.index, day_dist.values)},
            'weekend_ratio': f"{weekend_count / len(self.df) * 100:.2f}%"
        }

    def analyze_correlation(self):
        print_section_header("5. Correlation Analysis")

        numeric_cols = ['view', 'like', 'coin', 'favorite', 'share', 'danmaku', 'reply',
                        'duration', 'like_rate', 'coin_rate', 'favorite_rate',
                        'engagement_rate', 'completion_rate_proxy']
        available = [c for c in numeric_cols if c in self.df.columns]
        corr_matrix = self.df[available].corr()

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_figure(fig, '07_correlation_heatmap.png')

        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': round(val, 3)
                    })
        self.report['correlation_analysis'] = {
            'high_correlations': high_corr,
            'view_correlations': corr_matrix['view'].drop('view').sort_values(
                ascending=False).to_dict() if 'view' in corr_matrix else {}
        }


def main():
    DescriptiveStatistics().analyze_all()


if __name__ == "__main__":
    main()
