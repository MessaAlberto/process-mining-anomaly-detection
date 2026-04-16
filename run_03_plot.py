import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_COMP_DIR, TEST_FEATURE_SETS, LEVELS

def plot_gmm_results():
    result_csv_path = MODEL_COMP_DIR / 'gmm_unsupervised.csv'
    if not result_csv_path.exists():
        print(f"File {result_csv_path} not found.")
        return

    df = pd.read_csv(result_csv_path)

    level_order = LEVELS
    feat_order = list(TEST_FEATURE_SETS.keys())
    model_order = ['IsolationForest', 'LOF', 'DBSCAN']

    df = df[df['level'].isin(level_order) & df['feature_set'].isin(feat_order)]
    df['level'] = pd.Categorical(df['level'], categories=level_order, ordered=True)
    df['feature_set'] = pd.Categorical(df['feature_set'], categories=feat_order, ordered=True)
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)

    sns.set_theme(style="whitegrid")

    metrics_config = {
        'f1_score_anomalous_mean': 'F1-Score (Anomalies)',
        'precision_anomalous_mean': 'Precision (Anomalies)',
        'recall_anomalous_mean': 'Recall (Anomalies)',
        'f1_score_normal_mean': 'F1-Score (Normal)',
        'precision_normal_mean': 'Precision (Normal)',
        'recall_normal_mean': 'Recall (Normal)',
        'predicted_anomaly_rate_mean': 'Predicted Anomaly Rate (%)'
    }

    for metric, title_name in metrics_config.items():
        if metric not in df.columns:
            continue

        g = sns.catplot(
            data=df, x='model', y=metric, hue='feature_set', col='level', row='dataset',
            kind='bar', palette='viridis', height=5, aspect=1.2, sharey=False,
            order=[m for m in model_order if m in df['model'].unique()], 
            hue_order=[f for f in feat_order if f in df['feature_set'].unique()]
        )

        g.set_axis_labels("Algorithm", title_name)
        g.set_titles("{row_name} | {col_name}", fontweight='bold', pad=2)

        for ax in g.axes.flat:
            if 'rate' not in metric:
                ax.set_ylim(0, 1.05)
                ax.set_yticks(np.arange(0, 1.1, 0.1))
            else:
                ax.set_ylim(0, 100)
            
            ax.tick_params(labelbottom=True)
            
            for p in ax.patches:
                height = p.get_height()
                if pd.notnull(height) and height > 0:
                    ax.annotate(
                        f"{height:.2f}",
                        (p.get_x() + p.get_width()/2, height),
                        ha='center', va='bottom', fontsize=8,
                        xytext=(0, 4), textcoords='offset points', fontweight='bold'
                    )

        plt.subplots_adjust(top=0.9, hspace=0.4)
        g.fig.suptitle(f"GMM Unsupervised Evaluation: {title_name}", fontsize=16, fontweight='bold', y=0.98)
        
        out_name = MODEL_COMP_DIR / f'gmm_unsupervised_{metric}.png'
        g.savefig(out_name, bbox_inches='tight')
        print(f"Saved plot: {out_name}")
        plt.close()

if __name__ == "__main__":
    plot_gmm_results()