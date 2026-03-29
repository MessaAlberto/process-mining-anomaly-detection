import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_COMP_DIR, TEST_FEATURE_SETS, LEVELS

def plot_best_models():
    """
    Generate comparative bar plots for the best-performing models across datasets, levels, and feature sets.
    """
    result_csv_path = MODEL_COMP_DIR / 'avg_metrics.csv'
    if not result_csv_path.exists():
        print(f"Error: {result_csv_path} not found. Run the evaluation phase first.")
        return

    df = pd.read_csv(result_csv_path)
    idx_best = df.groupby(['dataset', 'model', 'level', 'feature_set'], dropna=False)['f1_score_anomalous'].idxmax()
    df_plot = df.loc[idx_best].copy()

    level_order = LEVELS
    feat_order = list(TEST_FEATURE_SETS.keys())
    model_order = ['IsolationForest', 'LOF', 'DBSCAN']

    df_plot = df_plot[df_plot['level'].isin(level_order) & df_plot['feature_set'].isin(feat_order)]
    
    df_plot['level'] = pd.Categorical(df_plot['level'], categories=level_order, ordered=True)
    df_plot['feature_set'] = pd.Categorical(df_plot['feature_set'], categories=feat_order, ordered=True)
    df_plot['model'] = pd.Categorical(df_plot['model'], categories=model_order, ordered=True)

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
        if metric not in df_plot.columns:
            print(f"Skipping metric {metric}: not found in data.")
            continue

        g = sns.catplot(
            data=df_plot,
            x='model',
            y=metric,
            hue='feature_set',
            col='level',
            row='dataset',
            kind='bar',
            palette='viridis',
            height=5,
            aspect=1.2,
            sharey=False,
            order=model_order,
            hue_order=feat_order
        )

        g.set_axis_labels("Algorithm", title_name)
        g.set_titles("{row_name} | {col_name}", fontweight='bold', pad=2)

        for ax in g.axes.flat:
            ax.set_ylim(0, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.1))
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("Algorithm")

            title_parts = ax.get_title().split('|')
            if len(title_parts) < 2:
                continue

            curr_ds = title_parts[0].strip()
            curr_lvl = title_parts[1].strip()

            subset = df_plot[(df_plot['dataset'] == curr_ds) & (df_plot['level'] == curr_lvl)]
            
            subset = subset.sort_values(['feature_set', 'model'])

            for p, (_, row) in zip(ax.patches, subset.iterrows()):
                height = p.get_height()
                if pd.notnull(height) and height > 0:
                    th_val = str(row['threshold_value'])
                    
                    if th_val != "N/A" and th_val != "nan":
                        label = f"{th_val}"
                        
                        ax.annotate(
                            label,
                            (p.get_x() + p.get_width()/2, height),
                            ha='center',
                            va='bottom',
                            fontsize=8,
                            xytext=(0, 4),
                            textcoords='offset points',
                            fontweight='bold'
                        )

        plt.subplots_adjust(top=0.9, hspace=0.4)
        g.fig.suptitle(f"Best Models Comparison: {title_name}", fontsize=16, fontweight='bold', y=0.98)

        out_name = f'{metric}_comparison.png'
        g.savefig(MODEL_COMP_DIR / out_name, bbox_inches='tight')
        print(f"Saved plot: {out_name}")
        plt.close()

if __name__ == "__main__":
    plot_best_models()