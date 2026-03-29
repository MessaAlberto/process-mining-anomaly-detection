import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_COMP_DIR, IF_THRESHOLD_COMP_DIR, IQR_THRESHOLDS, LEVELS

def generate_if_threshold_plots():
    """
    Generate line plots showing how the performance of the best-performing Isolation Forest model changes across different IQR thresholds for anomaly detection.
    """
    result_csv_path = MODEL_COMP_DIR / 'avg_metrics.csv'
    if not result_csv_path.exists():
        print(f"Error: {result_csv_path} not found.")
        return

    df = pd.read_csv(result_csv_path)
    df = df[df['model'] == 'IsolationForest'].copy()

    level_order = LEVELS
    feat_order = ['minimal', 'position_based', 'time_based_circular', 'time_based_linear']

    df = df[df['level'].isin(level_order) & df['feature_set'].isin(feat_order)].copy()

    df['level'] = pd.Categorical(df['level'], categories=level_order, ordered=True)
    df['feature_set'] = pd.Categorical(df['feature_set'], categories=feat_order, ordered=True)

    sns.set_theme(style="whitegrid")

    metrics_config = {
        'f1_score_anomalous': 'F1-Score (Anomalies)',
        'precision_anomalous': 'Precision (Anomalies)',
        'recall_anomalous': 'Recall (Anomalies)',
        'f1_score_normal': 'F1-Score (Normal Data)',
        'precision_normal': 'Precision (Normal Data)',
        'recall_normal': 'Recall (Normal Data)'
    }

    for metric, title_name in metrics_config.items():
        if metric not in df.columns:
            continue

        g = sns.relplot(
            data=df,
            x='threshold_value',
            y=metric,
            hue='feature_set',
            col='level',
            row='dataset',
            kind='line',
            marker='o',
            markersize=7,
            palette='viridis',
            height=4,
            aspect=1.3,
            linewidth=2.5,
            facet_kws={'sharex': False, 'sharey': False}
        )

        g.set_titles("{row_name} | {col_name}", fontweight='bold', pad=4)

        for ax in g.axes.flat:
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.1))

            ax.set_xticks(IQR_THRESHOLDS)
            ax.set_xlabel("Threshold (IQR Multiplier)", fontsize=10)

            ax.tick_params(axis='x', rotation=45, labelsize=9, labelbottom=True)
            ax.tick_params(axis='y', labelsize=9, labelleft=True)

            ax.set_ylabel(title_name, fontsize=10)

        plt.subplots_adjust(top=0.92, hspace=0.6, wspace=0.3)
        g.fig.suptitle(f"Isolation Forest IQR Analysis: {title_name}", fontsize=18, fontweight='bold')

        out_name = f'if_iqr_trend_{metric}.png'
        g.savefig(IF_THRESHOLD_COMP_DIR / out_name, bbox_inches='tight')
        print(f"Plot saved: {out_name}")
        plt.close()

if __name__ == "__main__":
    generate_if_threshold_plots()