"""
Part 7: Visualization Report Summary
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils_analysis import (
    load_data, save_report, print_section_header,
    calculate_success_index, logger, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR
)


class VisualizationReport:

    def __init__(self):
        self.reports = self._load_reports()

    def _load_reports(self):
        reports = {}
        files = [
            'descriptive_statistics_report.json',
            'clustering_analysis_report.json',
            'univariate_analysis_report.json',
            'regression_analysis_report.json',
            'ml_analysis_report.json',
            'group_comparison_report.json'
        ]
        for f in files:
            path = REPORTS_DIR / f
            if path.exists():
                with open(path, 'r', encoding='utf-8') as fh:
                    reports[f.replace('_report.json', '')] = json.load(fh)
        return reports

    def generate_all(self):
        print_section_header("Generating Final Report")
        self.generate_text_summary()
        self.generate_dashboard()
        self.generate_recommendations()
        logger.info("✅ All reports generated!")

    def generate_text_summary(self):
        print_section_header("1. Text Summary")
        lines = []
        lines.append("=" * 80)
        lines.append("Bilibili Video Topic Success Key Factor Analysis - Summary")
        lines.append("=" * 80)
        lines.append("")

        # Dataset overview
        desc = self.reports.get('descriptive_statistics', {})
        info = desc.get('data_info', {})
        lines.append("[Section 1: Dataset Overview]")
        lines.append(f"  Total videos: {info.get('total_videos', 'N/A')}")
        lines.append(f"  Categories: {info.get('unique_regions', 'N/A')}")
        lines.append("")

        # Clustering results
        cluster = self.reports.get('clustering_analysis', {})
        optimal_k = cluster.get('optimal_k_analysis', {}).get('optimal_k', 'N/A')
        sil = cluster.get('optimal_k_analysis', {}).get('silhouette_score', 'N/A')
        lines.append("[Section 2: Clustering Results]")
        lines.append(f"  Optimal K: {optimal_k}")
        lines.append(f"  Silhouette Score: {sil}")
        for p in cluster.get('cluster_profiles', []):
            lines.append(f"\n  Cluster {p['cluster_id']}:")
            lines.append(f"    Size: {p['size']} videos ({p['percentage']})")
            lines.append(f"    Avg Engagement Rate: {p['avg_engagement_rate']}")
            lines.append(f"    Top Category: {p['top_region']}")
        lines.append("")

        # Significant factors
        uni = self.reports.get('univariate_analysis', {})
        sig_factors = uni.get('significant_factors', [])
        lines.append("[Section 3: Significant Factors]")
        if sig_factors:
            lines.append("  Factors significantly affecting success (p < 0.05):")
            for f in sig_factors[:10]:
                corr_str = f", r={f.get('correlation', '')}" if 'correlation' in f else ""
                lines.append(f"    - {f.get('factor', 'N/A')}: p={f.get('p_value', 'N/A')}{corr_str}")
        lines.append("")

        # Regression results
        reg = self.reports.get('regression_analysis', {})
        ms = reg.get('model_summary', {})
        lines.append("[Section 4: Regression Analysis]")
        lines.append(f"  R-squared: {ms.get('r_squared', 'N/A')}")
        lines.append(f"  Adjusted R-squared: {ms.get('adj_r_squared', 'N/A')}")
        lines.append(f"  F-statistic: {ms.get('f_statistic', 'N/A')}")
        coefs = reg.get('coefficients', [])
        sig_coefs = [c for c in coefs if c.get('显著性', '') in ['*', '**', '***']]
        if sig_coefs:
            lines.append("  Significant coefficients:")
            for c in sig_coefs[:5]:
                lines.append(f"    - {c.get('特征', 'N/A')}: "
                             f"β={c.get('系数', 0):.4f}{c.get('显著性', '')}")
        lines.append("")

        # ML results
        ml = self.reports.get('ml_analysis', {})
        mc = ml.get('model_comparison', [])
        if mc:
            best = max(mc, key=lambda x: x.get('test_r2', 0))
            lines.append("[Section 5: Machine Learning Results]")
            lines.append(f"  Best Model: {best.get('model', 'N/A')}")
            lines.append(f"  Test R²: {best.get('test_r2', 'N/A')}")
            lines.append(f"  Test RMSE: {best.get('test_rmse', 'N/A')}")
            lines.append(f"  CV R² Mean: {best.get('cv_r2_mean', 'N/A')}")
            fi = ml.get('feature_importance', {}).get('features', [])
            if fi:
                lines.append("  Top 5 Important Features:")
                for feat in fi[:5]:
                    lines.append(f"    - {feat.get('feature', 'N/A')}: "
                                 f"{feat.get('importance', 0):.4f}")

        summary = "\n".join(lines)
        path = REPORTS_DIR / 'analysis_summary.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(summary)
        print(f"\n✅ Saved: {path}")

    def generate_dashboard(self):
        print_section_header("2. Interactive Dashboard")
        try:
            df = load_data('bilibili_videos_data.csv')
            if 'success_index' not in df.columns:
                df['success_index'] = calculate_success_index(df)

            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Category Distribution', 'Success Index Distribution',
                    'Feature Importance', 'Publishing Hour Distribution',
                    'Cluster Size', 'Model R² Comparison'
                ),
                specs=[
                    [{'type': 'pie'}, {'type': 'histogram'}],
                    [{'type': 'bar'}, {'type': 'bar'}],
                    [{'type': 'bar'}, {'type': 'bar'}]
                ]
            )

            # Category pie
            region = df['tname'].value_counts().head(10)
            fig.add_trace(
                go.Pie(labels=region.index.tolist(), values=region.values.tolist(),
                       textinfo='label+percent'),
                row=1, col=1
            )

            # Success index histogram
            fig.add_trace(
                go.Histogram(x=df['success_index'], nbinsx=50, name='Success Index',
                             marker_color='steelblue'),
                row=1, col=2
            )

            # Feature importance
            imp_file = RESULTS_DIR / 'tables' / 'feature_importance.csv'
            if imp_file.exists():
                imp = pd.read_csv(imp_file).head(10)
                fig.add_trace(
                    go.Bar(x=imp['importance'], y=imp['feature'],
                           orientation='h', marker_color='#27ae60'),
                    row=2, col=1
                )

            # Publishing hour
            if 'pub_hour' in df.columns:
                h = df['pub_hour'].value_counts().sort_index()
                fig.add_trace(
                    go.Bar(x=h.index.tolist(), y=h.values.tolist(),
                           marker_color='coral'),
                    row=2, col=2
                )

            # Cluster size
            cl_file = RESULTS_DIR / 'tables' / 'cluster_profiles.csv'
            if cl_file.exists():
                cl = pd.read_csv(cl_file)
                fig.add_trace(
                    go.Bar(x=[f"Cluster {i}" for i in range(len(cl))],
                           y=cl['size'].tolist(), marker_color='#9b59b6'),
                    row=3, col=1
                )

            # Model comparison
            m_file = RESULTS_DIR / 'tables' / 'model_comparison.csv'
            if m_file.exists():
                m = pd.read_csv(m_file)
                fig.add_trace(
                    go.Bar(x=m['model'].tolist(), y=m['test_r2'].tolist(),
                           marker_color='#f39c12'),
                    row=3, col=2
                )

            fig.update_layout(
                height=1200, showlegend=False,
                title_text="Bilibili Video Analysis Dashboard",
                title_font_size=20
            )
            fig.write_html(str(FIGURES_DIR / 'interactive_dashboard.html'))
            logger.info("✅ Interactive dashboard saved")
        except Exception as e:
            logger.warning(f"Dashboard generation failed: {e}")

    def generate_recommendations(self):
        print_section_header("3. Practical Recommendations")
        lines = []
        lines.append("=" * 80)
        lines.append("Practical Recommendations Based on Data Analysis")
        lines.append("=" * 80)
        lines.append("")

        cluster = self.reports.get('clustering_analysis', {})
        profiles = cluster.get('cluster_profiles', [])
        if profiles:
            lines.append("[Content Strategy by User Group]")
            lines.append("")
            for p in profiles:
                engagement = float(p['avg_engagement_rate'].rstrip('%'))
                lines.append(f"  Cluster {p['cluster_id']}:")
                lines.append(f"    Engagement: {p['avg_engagement_rate']}, "
                             f"Top Category: {p['top_region']}")
                if engagement > 15:
                    lines.append("    Strategy: Focus on in-depth content, "
                                 "prioritize quality, suitable for long-form videos")
                elif engagement > 8:
                    lines.append("    Strategy: Balance depth and entertainment, "
                                 "suitable for medium-length videos")
                else:
                    lines.append("    Strategy: Light entertainment content, "
                                 "fast-paced, suitable for short videos")
                lines.append("")

        lines.append("[Optimal Publishing Time]")
        lines.append("  - Choose publishing time based on target audience behavior")
        lines.append("  - Weekday evenings and weekend afternoons generally perform better")
        lines.append("")

        lines.append("[Video Duration Recommendations]")
        lines.append("  - Short (<3min): Quick consumption content, strong opening hook")
        lines.append("  - Medium (3-10min): Tutorials, reviews, explanations")
        lines.append("  - Long (10-30min): Deep analysis, storytelling content")
        lines.append("  - Extra Long (>30min): Requires exceptionally strong content")
        lines.append("")

        lines.append("[Category Selection]")
        lines.append("  - Match category to content precisely")
        lines.append("  - Popular categories are highly competitive; consider niche entry points")
        lines.append("")

        lines.append("[Improving Engagement Rate]")
        lines.append("  - Optimize title and thumbnail for higher click-through rate")
        lines.append("  - Include call-to-action for likes, coins, and favorites")
        lines.append("  - Reply to comments to build community")
        lines.append("  - Set danmaku interaction points at key moments")
        lines.append("")

        # Add key findings from ML
        ml = self.reports.get('ml_analysis', {})
        fi = ml.get('feature_importance', {}).get('features', [])
        if fi:
            lines.append("[Key Success Factors (ML-based)]")
            for i, feat in enumerate(fi[:5], 1):
                lines.append(f"  {i}. {feat.get('feature', 'N/A')}: "
                             f"importance = {feat.get('importance', 0):.4f}")

        text = "\n".join(lines)
        path = REPORTS_DIR / 'practical_recommendations.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(text)
        print(f"\n✅ Saved: {path}")


def main():
    VisualizationReport().generate_all()

if __name__ == "__main__":
    main()
