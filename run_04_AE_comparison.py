import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_COMP_DIR, AE_COMP_DIR, ANOMALY_RATE

def generate_comparison():
    """
    Generate a comparison between the average best-performing Isolation Forest model from the evaluation phase and the results reported by AE.
    """
    result_path = MODEL_COMP_DIR / 'avg_metrics.csv'
    if not result_path.exists():
        print(f"File not found: {result_path}")
        return

    df_user = pd.read_csv(result_path)
    df_user['ds_clean'] = df_user['dataset'].str.replace('.csv', '', regex=False).str.lower()
    
    df_if = df_user[
        (df_user['model'] == 'IsolationForest') &
        (df_user['level'] != 'L1_Micro')
    ].copy()

    if df_if.empty:
        print("No data found for IsolationForest without L1_Micro!")
        return

    # Select the best-performing configuration for each dataset based on F1-Score for anomalies
    idx_best = df_if.groupby('ds_clean')['f1_score_anomalous'].idxmax()
    user_best = df_if.loc[idx_best].copy()

    user_best['average_precision'] = (user_best['precision_normal'] * (1 - ANOMALY_RATE)) + (user_best['precision_anomalous'] * ANOMALY_RATE)
    user_best['average_recall'] = (user_best['recall_normal'] * (1 - ANOMALY_RATE)) + (user_best['recall_anomalous'] * ANOMALY_RATE)
    user_best['average_f1'] = (user_best['f1_score_normal'] * (1 - ANOMALY_RATE)) + (user_best['f1_score_anomalous'] * ANOMALY_RATE)

    # Retrieve AE results from paper (Table 5)
    AE_result = {
        'bpi_2012': {
            'normal_precision': 0.92, 'normal_recall': 0.61, 'normal_f1': 0.73,
            'anomalous_precision': 0.11, 'anomalous_recall': 0.47, 'anomalous_f1': 0.18,
            'average_precision': 0.839, 'average_recall': 0.596, 'average_f1': 0.675
        },
        'bpi_2013': {
            'normal_precision': 0.93, 'normal_recall': 0.65, 'normal_f1': 0.77,
            'anomalous_precision': 0.16, 'anomalous_recall': 0.58, 'anomalous_f1': 0.25,
            'average_precision': 0.853, 'average_recall': 0.643, 'average_f1': 0.718

        },
        'small_log': {
            'normal_precision': 0.97, 'normal_recall': 0.99, 'normal_f1': 0.98,
            'anomalous_precision': 0.92, 'anomalous_recall': 0.69, 'anomalous_f1': 0.78,
            'average_precision': 0.965, 'average_recall': 0.963, 'average_f1': 0.962
        },
        'large_log': {
            'normal_precision': 0.97, 'normal_recall': 0.99, 'normal_f1': 0.98,
            'anomalous_precision': 0.82, 'anomalous_recall': 0.71, 'anomalous_f1': 0.76,
            'average_precision': 0.955, 'average_recall': 0.962, 'average_f1': 0.958
        }
    }

    rows = []
    plot_data = []
    common_ds = ['bpi_2012', 'bpi_2013', 'small_log', 'large_log']

    metrics_map = [
        ('Normal', 'Precision', 'normal_precision', 'precision_normal'),
        ('Normal', 'Recall', 'normal_recall', 'recall_normal'),
        ('Normal', 'F1-Score', 'normal_f1', 'f1_score_normal'),
        ('Anomalous', 'Precision', 'anomalous_precision', 'precision_anomalous'),
        ('Anomalous', 'Recall', 'anomalous_recall', 'recall_anomalous'),
        ('Anomalous', 'F1-Score', 'anomalous_f1', 'f1_score_anomalous'),
        ('Average', 'Precision', 'average_precision', 'average_precision'),
        ('Average', 'Recall', 'average_recall', 'average_recall'),
        ('Average', 'F1-Score', 'average_f1', 'average_f1')
    ]

    for ds in common_ds:
        matched_ds = user_best[user_best['ds_clean'].str.contains(ds)]
        if matched_ds.empty:
            continue

        user_row = matched_ds.iloc[0]
        com_row = AE_result[ds]
        
        win_lvl = user_row['level']
        win_feat = user_row['feature_set']
        win_th = user_row['threshold_value']
        
        ds_label = f"{ds}\n({win_lvl},\n{win_feat},\nth:{win_th})"

        for cls, metric_name, com_key, user_key in metrics_map:
            val_com = com_row[com_key]
            val_user = user_row[user_key]
            imp = ((val_user - val_com) / val_com) * 100 if val_com > 0 else 0

            rows.append({
                'Dataset': ds,
                'Class': cls,
                'Metric': metric_name,
                'AE': f"{val_com:.3f}",
                'My Framework (IF)': f"{val_user:.3f}",
                'Improvement (%)': f"{imp:+.1f}%",
                'Winning Level': win_lvl,
                'Winning Feature': win_feat,
                'Threshold': win_th
            })

            plot_data.extend([
                {'Dataset': ds_label, 'Class': cls, 'Metric': metric_name, 'Author': 'AE', 'Score': float(val_com)},
                {'Dataset': ds_label, 'Class': cls, 'Metric': metric_name, 'Author': 'My Framework (IF)', 'Score': float(val_user)}
            ])

    df_md = pd.DataFrame(rows)
    df_md.to_csv(AE_COMP_DIR / 'ae_vs_my_framework.csv', index=False)
    print(f"Saved comparison data: {AE_COMP_DIR / 'ae_vs_my_framework.csv'}")

    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        df_plot['Class'] = pd.Categorical(df_plot['Class'], categories=['Normal', 'Anomalous', 'Average'], ordered=True)
        df_plot['Metric'] = pd.Categorical(df_plot['Metric'], categories=['Precision', 'Recall', 'F1-Score'], ordered=True)

        sns.set_theme(style="whitegrid")

        g = sns.catplot(
            data=df_plot,
            x='Dataset',
            y='Score',
            hue='Author',
            col='Class',
            row='Metric',
            kind='bar',
            palette='magma',
            height=4.5,
            aspect=1.4,
            sharey=False
        )

        g.set_axis_labels("", "Score")
        g.set_titles("{col_name} Class | {row_name}", fontweight='bold')

        for ax in g.axes.flat:
            ax.set_ylim(0, 1.15)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.tick_params(labelbottom=True, labelsize=9)

            for p in ax.patches:
                height = p.get_height()
                if pd.notnull(height) and height > 0:
                    ax.annotate(f"{height:.2f}",
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9,
                                xytext=(0, 4), textcoords='offset points',
                                fontweight='bold')

        plt.subplots_adjust(top=0.90, hspace=0.6, bottom=0.15)
        g.fig.suptitle("Performance Comparison: My Framework vs AE", fontsize=18, fontweight='bold')

        plot_path = AE_COMP_DIR / 'ae_vs_my_framework_all_metrics.png'
        g.savefig(plot_path)
        plt.close()
        print(f"Saved comparison plot: {plot_path}")

if __name__ == "__main__":
    generate_comparison()