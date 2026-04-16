import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_COMP_DIR, AE_COMP_DIR, ANOMALY_RATE


def compute_average_metrics(df):
    df['average_precision'] = (df['precision_normal_mean'] * (1 - ANOMALY_RATE)) + \
                              (df['precision_anomalous_mean'] * ANOMALY_RATE)
    df['average_recall'] = (df['recall_normal_mean'] * (1 - ANOMALY_RATE)) + \
                           (df['recall_anomalous_mean'] * ANOMALY_RATE)
    df['average_f1'] = (df['f1_score_normal_mean'] * (1 - ANOMALY_RATE)) + \
                       (df['f1_score_anomalous_mean'] * ANOMALY_RATE)
    return df


def extract_model(df, cfg):
    m, l, f = cfg
    return df[(df['model'] == m) & (df['level'] == l) & (df['feature_set'] == f)].copy()


def format_label(name, cfg):
    if cfg is None:
        return name
    m, l, f = cfg
    return f"{name} ({m} | {l} | {f})"


def generate_comparison():
    gmm_path = MODEL_COMP_DIR / 'gmm_unsupervised.csv'

    if not gmm_path.exists():
        print("Missing gmm_unsupervised.csv")
        return

    df = pd.read_csv(gmm_path)
    df['ds_clean'] = df['dataset'].str.replace('.csv', '', regex=False).str.lower()

    df = df[(df['model'] == 'IsolationForest') & (df['level'] != 'L1_Micro')].copy()
    df = compute_average_metrics(df)

    # SELECT MODELS based on different criteria
    # Oracle: Best F1 for anomalous class per dataset (supervised)
    idx_oracle = df.groupby('ds_clean')['f1_score_anomalous_mean'].idxmax()
    oracle = df.loc[idx_oracle].copy()

    # Model based on distance between GMM components (unsupervised)
    idx_unsup = df.groupby('ds_clean')['separation_score'].idxmax()
    local_unsup = df.loc[idx_unsup].copy()

    # Models based on performance across all datasets (global selection)
    cfg_l3_pc = ('IsolationForest', 'L3_Global', 'pure_cyclical')
    fixed_l3_pc_model = extract_model(df, cfg_l3_pc)

    cfg_l3_dt_cyclical = ('IsolationForest', 'L3_Global', 'delta_cyclical')
    fixed_l3_dt_cyclical_model = extract_model(df, cfg_l3_dt_cyclical)

    label_oracle = "2. Oracle (Local F1)"
    label_unsup = "3. Unsupervised (Local GMM Separation)"
    label_fixed_pc = format_label("5. Fixed Config Cyclical", cfg_l3_pc)
    label_fixed_dt_cyclical = format_label("6. Fixed Config DT+Cyclical", cfg_l3_dt_cyclical)

    approaches = {
        '1. Comuzzi': None,
        label_oracle: oracle,
        label_unsup: local_unsup,
        label_fixed_pc: fixed_l3_pc_model,
        label_fixed_dt_cyclical: fixed_l3_dt_cyclical_model
    }

    common_ds = ['bpi_2012', 'bpi_2013', 'small_log', 'large_log']

    dataset_labels = {}
    for ds in common_ds:
        match_unsup = local_unsup[local_unsup['ds_clean'] == ds]
        match_oracle = oracle[oracle['ds_clean'] == ds]

        label_parts = [ds]
        if not match_unsup.empty:
            lvl_u = match_unsup.iloc[0]['level'].split('_')[0]
            feat_u = match_unsup.iloc[0]['feature_set']
            label_parts.append(f"[GMM: {lvl_u} | {feat_u}]")
            
        if not match_oracle.empty:
            lvl_o = match_oracle.iloc[0]['level'].split('_')[0]
            feat_o = match_oracle.iloc[0]['feature_set']
            label_parts.append(f"[Oracle: {lvl_o} | {feat_o}]")
            
        dataset_labels[ds] = "\n".join(label_parts)

    Comuzzi_result = {
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

    metrics_map = [
        ('Normal', 'Precision', 'normal_precision', 'precision_normal_mean'),
        ('Normal', 'Recall', 'normal_recall', 'recall_normal_mean'),
        ('Normal', 'F1-Score', 'normal_f1', 'f1_score_normal_mean'),
        ('Anomalous', 'Precision', 'anomalous_precision', 'precision_anomalous_mean'),
        ('Anomalous', 'Recall', 'anomalous_recall', 'recall_anomalous_mean'),
        ('Anomalous', 'F1-Score', 'anomalous_f1', 'f1_score_anomalous_mean'),
        ('Average', 'Precision', 'average_precision', 'average_precision'),
        ('Average', 'Recall', 'average_recall', 'average_recall'),
        ('Average', 'F1-Score', 'average_f1', 'average_f1')
    ]

    rows = []
    plot_data = []

    for ds in common_ds:
        com = Comuzzi_result[ds]
        display_ds = dataset_labels[ds]

        for cls, metric_name, com_key, user_key in metrics_map:
            val_com = com[com_key]

            plot_data.append({
                'Dataset': display_ds,
                'Class': cls,
                'Metric': metric_name,
                'Approach': '1. Comuzzi',
                'Score': val_com
            })

            row_dict = {
                'Dataset': ds,
                'Class': cls,
                'Metric': metric_name,
                'Comuzzi': f"{val_com:.3f}"
            }

            for name, df_model in approaches.items():
                if name == '1. Comuzzi':
                    continue

                match = df_model[df_model['ds_clean'] == ds]
                if match.empty:
                    continue

                val_model = match.iloc[0][user_key]

                row_dict[name] = f"{val_model:.3f}"

                plot_data.append({
                    'Dataset': display_ds,
                    'Class': cls,
                    'Metric': metric_name,
                    'Approach': name,
                    'Score': val_model
                })

            rows.append(row_dict)

    AE_COMP_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AE_COMP_DIR / 'full_comparison_global_models.csv', index=False)

    df_plot = pd.DataFrame(plot_data)
    df_plot['Class'] = pd.Categorical(df_plot['Class'], categories=['Normal', 'Anomalous', 'Average'], ordered=True)
    df_plot['Metric'] = pd.Categorical(df_plot['Metric'], categories=['Precision', 'Recall', 'F1-Score'], ordered=True)

    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=df_plot,
        x='Dataset',
        y='Score',
        hue='Approach',
        col='Class',
        row='Metric',
        kind='bar',
        palette='viridis',
        height=5.0,
        aspect=1.6,
        sharey=False,
        sharex=False
    )

    g.set_axis_labels("", "Score")
    g.set_titles("{col_name} Class | {row_name}", fontweight='bold')

    for ax in g.axes.flat:
        ax.set_ylim(0, 1.15)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

        ax.tick_params(labelbottom=True, labelsize=9)
        ax.set_xlabel("")

        for p in ax.patches:
            height = p.get_height()
            if pd.notnull(height) and height > 0:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=8,
                            xytext=(0, 4), textcoords='offset points')

    plt.subplots_adjust(top=0.92, hspace=0.5, bottom=0.15)
    g.fig.suptitle("Comuzzi vs Global & Local Unsupervised Selection", fontsize=18)

    plot_path = AE_COMP_DIR / 'comparison_global_models.png'
    g.savefig(plot_path)
    plt.close()

    print(f"\nSaved results in {AE_COMP_DIR}")


if __name__ == "__main__":
    generate_comparison()
