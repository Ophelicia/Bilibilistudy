"""
Part 2: Clustering Analysis (English Labels)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px

from utils_analysis import (
    load_data, save_figure, save_table, save_report,
    print_section_header, logger, FIGURES_DIR
)


class ClusteringAnalysis:

    def __init__(self, data_path='bilibili_videos_data.csv'):
        self.df = load_data(data_path, translate=True)
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.optimal_k = None
        self.report = {
            'clustering_config': {}, 'optimal_k_analysis': {},
            'cluster_profiles': {}, 'cluster_statistics': {}
        }

    def analyze_all(self):
        print_section_header("Clustering Analysis")
        X, features = self.prepare_features()
        self.optimal_k = self.find_optimal_k(X, features)
        self.perform_kmeans(X, features, self.optimal_k)
        self.analyze_clusters()
        self.visualize_clusters(X, features)
        self.hierarchical_clustering(X)
        self.save_results()
        save_report(self.report, 'clustering_analysis_report.json')
        logger.info("✅ Clustering analysis completed!")

    def prepare_features(self):
        feature_cols = ['like_rate', 'coin_rate', 'favorite_rate', 'share_rate',
                        'danmaku_rate', 'reply_rate', 'completion_rate_proxy']
        available = [c for c in feature_cols if c in self.df.columns]
        X = self.scaler.fit_transform(self.df[available].fillna(0))
        self.report['clustering_config'] = {
            'n_samples': X.shape[0], 'n_features': X.shape[1], 'features': available
        }
        return X, available

    def find_optimal_k(self, X, features, k_range=range(2, 11)):
        print_section_header("Finding Optimal K")
        inertias, sil_scores, ch_scores, db_scores = [], [], [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, labels))
            ch_scores.append(calinski_harabasz_score(X, labels))
            db_scores.append(davies_bouldin_score(X, labels))

        best_k = list(k_range)[np.argmax(sil_scores)]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
        axes[0, 0].set_ylabel('Inertia (SSE)', fontsize=11)
        axes[0, 0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].plot(k_range, sil_scores, 'go-', linewidth=2, markersize=8)
        axes[0, 1].axvline(best_k, color='red', linestyle='--', label=f'Optimal K={best_k}')
        axes[0, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
        axes[0, 1].set_ylabel('Silhouette Score', fontsize=11)
        axes[0, 1].set_title('Silhouette Score (Higher is Better)', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].plot(k_range, ch_scores, 'mo-', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Number of Clusters (K)', fontsize=11)
        axes[1, 0].set_ylabel('CH Index', fontsize=11)
        axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].plot(k_range, db_scores, 'co-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('Number of Clusters (K)', fontsize=11)
        axes[1, 1].set_ylabel('DB Index', fontsize=11)
        axes[1, 1].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_figure(fig, '08_optimal_k_selection.png')

        eval_df = pd.DataFrame({
            'K': list(k_range), 'Silhouette': sil_scores,
            'CH_Index': ch_scores, 'DB_Index': db_scores
        })
        save_table(eval_df, 'clustering_evaluation_metrics.csv')
        self.report['optimal_k_analysis'] = {
            'optimal_k': int(best_k),
            'silhouette_score': float(sil_scores[best_k - 2]),
            'ch_score': float(ch_scores[best_k - 2]),
            'db_score': float(db_scores[best_k - 2])
        }
        print(f"✅ Optimal K = {best_k}")
        return best_k

    def perform_kmeans(self, X, features, k):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        self.cluster_labels = km.fit_predict(X)
        self.df['cluster'] = self.cluster_labels
        self.cluster_centers = pd.DataFrame(
            self.scaler.inverse_transform(km.cluster_centers_), columns=features
        )
        sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        self.report['cluster_statistics']['cluster_sizes'] = sizes.to_dict()

    def analyze_clusters(self):
        print_section_header("Cluster Profile Analysis")
        profiles = []
        cluster_name_map = {}

        for cid in sorted(self.df['cluster'].unique()):
            cd = self.df[self.df['cluster'] == cid]
            engagement = cd['engagement_rate'].mean()

            if engagement > 0.15:
                name = "High Engagement"
            elif engagement > 0.08:
                name = "Moderate Engagement"
            else:
                name = "Light Engagement"

            cluster_name_map[cid] = name

            p = {
                'cluster_id': int(cid),
                'cluster_name': name,
                'size': len(cd),
                'percentage': f"{len(cd) / len(self.df) * 100:.2f}%",
                'avg_like_rate': f"{cd['like_rate'].mean() * 100:.2f}%",
                'avg_coin_rate': f"{cd['coin_rate'].mean() * 100:.2f}%",
                'avg_favorite_rate': f"{cd['favorite_rate'].mean() * 100:.2f}%",
                'avg_engagement_rate': f"{cd['engagement_rate'].mean() * 100:.2f}%",
                'avg_view': int(cd['view'].mean()),
                'median_view': int(cd['view'].median()),
                'top_region': cd['tname'].mode()[0] if len(cd['tname'].mode()) > 0 else 'N/A',
                'avg_duration': int(cd['duration'].mean()),
                'peak_hour': int(cd['pub_hour'].mode()[0]) if 'pub_hour' in cd.columns and len(cd['pub_hour'].mode()) > 0 else 0
            }
            profiles.append(p)
            self.df.loc[self.df['cluster'] == cid, 'cluster_name'] = name

            print(f"  Cluster {cid}: {name} | Size: {p['size']} ({p['percentage']}) | "
                  f"Engagement: {p['avg_engagement_rate']} | Top: {p['top_region']}")

        save_table(pd.DataFrame(profiles), 'cluster_profiles.csv')
        self.report['cluster_profiles'] = profiles

    def visualize_clusters(self, X, features):
        print_section_header("Cluster Visualization")

        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.cluster_labels,
                             cmap='tab10', alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
        ax.set_title('Clustering Results (PCA Visualization)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        cluster_names = self.df.groupby('cluster')['cluster_name'].first().to_dict()
        handles, labels = scatter.legend_elements()
        legend_labels = [f"C{i}: {cluster_names.get(i, '')}" for i in sorted(cluster_names.keys())]
        ax.legend(handles, legend_labels, loc='best', fontsize=10)
        plt.tight_layout()
        save_figure(fig, '09_clustering_pca.png')

        # Radar chart
        n_clusters = len(self.cluster_centers)
        fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 6),
                                 subplot_kw=dict(projection='polar'))
        if n_clusters == 1:
            axes = [axes]

        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        feature_labels = [f.replace('_rate', '').replace('_proxy', '').replace('completion', 'compl')
                          for f in features]

        for idx, cid in enumerate(sorted(self.df['cluster'].unique())):
            ax = axes[idx]
            values = self.cluster_centers.iloc[cid].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=9)
            name = cluster_names.get(cid, f'Cluster {cid}')
            ax.set_title(f'C{cid}: {name}', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        save_figure(fig, '10_cluster_radar_charts.png')

        # 3D interactive
        try:
            pca_3d = PCA(n_components=3, random_state=42)
            X_3d = pca_3d.fit_transform(X)
            plot_df = pd.DataFrame({
                'PC1': X_3d[:, 0], 'PC2': X_3d[:, 1], 'PC3': X_3d[:, 2],
                'Cluster': self.df['cluster_name'].values,
                'Title': self.df['title'].values,
                'Views': self.df['view'].values,
                'Category': self.df['tname'].values
            })
            fig_3d = px.scatter_3d(
                plot_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                hover_data=['Title', 'Views', 'Category'],
                title='3D Clustering Visualization (PCA)'
            )
            fig_3d.write_html(str(FIGURES_DIR / '11_clustering_3d_interactive.html'))
            logger.info("✅ 3D interactive chart saved")
        except Exception as e:
            logger.warning(f"3D chart failed: {e}")

    def hierarchical_clustering(self, X):
        print_section_header("Hierarchical Clustering")
        linkage_matrix = linkage(X, method='ward')
        fig, ax = plt.subplots(figsize=(16, 8))
        dendrogram(linkage_matrix, ax=ax, truncate_mode='lastp', p=30,
                   leaf_font_size=10)
        ax.set_xlabel('Sample Index / Cluster Size', fontsize=12)
        ax.set_ylabel('Distance (Ward)', fontsize=12)
        ax.set_title('Hierarchical Clustering Dendrogram (Ward Method)',
                      fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_figure(fig, '12_hierarchical_dendrogram.png')

    def save_results(self):
        output_cols = ['bvid', 'title', 'tname', 'view', 'like', 'coin', 'favorite',
                       'engagement_rate', 'cluster', 'cluster_name']
        available = [c for c in output_cols if c in self.df.columns]
        save_table(self.df[available], 'videos_with_clusters.csv')


def main():
    ClusteringAnalysis().analyze_all()

if __name__ == "__main__":
    main()
