import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import RESULTS_DIR, TEST_FEATURE_SETS

result_csv_path = RESULTS_DIR / 'model_evaluation.csv'
if not result_csv_path.exists():
    print(f"Error: {result_csv_path} not found.")
    exit(1)

df = pd.read_csv(result_csv_path)

idx_best = df.groupby(['dataset', 'model', 'level', 'feature_set'], observed=True)['f1_score_anomalous'].idxmax()
df_plot = df.loc[idx_best].copy()

level_order = ['L3_Global', 'L2_Activity', 'L1_Micro']
feat_order = list(TEST_FEATURE_SETS.keys())
df_plot['level'] = pd.Categorical(df_plot['level'], categories=level_order, ordered=True)
df_plot['feature_set'] = pd.Categorical(df_plot['feature_set'], categories=feat_order, ordered=True)

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
        sharey=False
    )

    g.set_axis_labels("Algorithm", title_name)
    g.set_titles("{row_name} | {col_name}", fontweight='bold', pad=2)

    for ax in g.axes.flat:
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("Algorithm")

        title_parts = ax.get_title().split('|')
        curr_ds = title_parts[0].strip()
        curr_lvl = title_parts[1].strip()

        subset = df_plot[(df_plot['dataset'] == curr_ds) & (df_plot['level'] == curr_lvl)]

        for p in ax.patches:
            height = p.get_height()
            
            if pd.notnull(height) and height > 0:
                row = subset[np.isclose(subset[metric], height, atol=1e-5)].iloc[0]
                th_val = row['threshold']

                label = ""
                if pd.notna(th_val) and str(th_val).lower() != 'nan' and str(th_val).upper() != 'N/A':
                    label += f"(th:{th_val})"

                ax.annotate(label,
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=8,
                            xytext=(0, 4), textcoords='offset points',
                            fontweight='bold')

    plt.subplots_adjust(top=0.9, hspace=0.4)

    out_name = f'{metric}_comparison.png'
    g.savefig(RESULTS_DIR / out_name)
    print(f"Plot saved: {out_name}")
    plt.close()