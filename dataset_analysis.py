import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_DIR, DATASET_SCHEMAS, DATASETS, MIN_SAMPLES

def analyze_datasets():
    results = []

    for file_name in DATASETS:
        input_path = RAW_DIR / file_name
        if not input_path.exists():
            print(f"File {file_name} not found, skipping.")
            continue

        print(f"Analyzing {file_name}...")
        schema = DATASET_SCHEMAS[file_name]
        case_col = schema['case']
        act_col = schema['act']
        time_col = schema['time']
        res_col = schema['res']

        df = pd.read_csv(input_path)
        df[time_col] = pd.to_datetime(df[time_col], utc=True, format='mixed')

        df = df.sort_values([case_col, time_col]).reset_index(drop=True)
        df['delta_t'] = df.groupby(case_col)[time_col].diff().dt.total_seconds().fillna(0)

        total_events = len(df)
        num_activities = df[act_col].nunique()
        num_resources = df[res_col].nunique() if res_col and res_col in df.columns else 0

        l3_variance = df['delta_t'].var()
        l3_std_hours = np.sqrt(l3_variance) / 3600.0 if pd.notna(l3_variance) and l3_variance >= 0 else 0
        l3_mean_hours = df['delta_t'].mean() / 3600.0
        l3_min_hours = df['delta_t'].min() / 3600.0
        l3_max_hours = df['delta_t'].max() / 3600.0

        results.append({
            'Dataset': file_name, 
            'Level': 'L3_Global',
            'Distinct_Activities': num_activities,
            'Distinct_Resources': num_resources,
            'Num_Models': 1,
            'Total_Events': total_events,
            'Avg_Events_per_Model': total_events,
            'Median_Events_per_Model': total_events,
            'Models_Dropped_Under_30': 0 if total_events >= MIN_SAMPLES else 1,
            'Percent_Events_Dropped': 0.0 if total_events >= MIN_SAMPLES else 100.0,
            'Avg_DeltaT_Variance': l3_variance,
            'Avg_DeltaT_Std_Hours': l3_std_hours,
            'Avg_DeltaT_Mean_Hours': l3_mean_hours,
            'Avg_DeltaT_Min_Hours': l3_min_hours,
            'Avg_DeltaT_Max_Hours': l3_max_hours
        })

        l2_groups = df.groupby(act_col)
        l2_stats = process_groups(l2_groups, total_events, file_name, 'L2_Activity', num_activities, num_resources)
        results.append(l2_stats)

        if res_col and res_col in df.columns:
            l1_groups = df.groupby([act_col, res_col])
            l1_stats = process_groups(l1_groups, total_events, file_name, 'L1_Micro', num_activities, num_resources)
            results.append(l1_stats)
        else:
            print(f"  -> No resource column for {file_name}, skipping L1.")

    summary_df = pd.DataFrame(results)
    summary_df = summary_df.round(2)

    output_path = RAW_DIR.parent / 'dataset_analysis_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nAnalysis completed. Saved to {output_path}")

    print("\n--- Results Extract ---")
    extract_cols = ['Dataset', 'Level', 'Distinct_Activities', 'Distinct_Resources', 'Total_Events', 'Avg_Events_per_Model', 'Models_Dropped_Under_30', 'Percent_Events_Dropped', 'Avg_DeltaT_Variance', 'Avg_DeltaT_Std_Hours']
    print(summary_df[extract_cols].to_string())

def process_groups(groupby_obj, total_events, dataset_name, level_name, num_acts, num_res):
    group_sizes = groupby_obj.size()
    group_variances = groupby_obj['delta_t'].var().fillna(0)
    group_means = groupby_obj['delta_t'].mean().fillna(0)
    group_mins = groupby_obj['delta_t'].min().fillna(0)
    group_maxs = groupby_obj['delta_t'].max().fillna(0)

    valid_groups = group_sizes[group_sizes >= MIN_SAMPLES]
    dropped_groups = group_sizes[group_sizes < MIN_SAMPLES]
    dropped_events_count = dropped_groups.sum()

    avg_variance = group_variances[valid_groups.index].mean() if len(valid_groups) > 0 else 0
    avg_std_hours = np.sqrt(avg_variance) / 3600.0 if avg_variance >= 0 else 0
    avg_mean_hours = (group_means[valid_groups.index].mean() / 3600.0) if len(valid_groups) > 0 else 0
    avg_min_hours = (group_mins[valid_groups.index].mean() / 3600.0) if len(valid_groups) > 0 else 0
    avg_max_hours = (group_maxs[valid_groups.index].mean() / 3600.0) if len(valid_groups) > 0 else 0

    return {
        'Dataset': dataset_name,
        'Level': level_name,
        'Distinct_Activities': num_acts,
        'Distinct_Resources': num_res,
        'Num_Models': len(group_sizes),
        'Total_Events': total_events,
        'Avg_Events_per_Model': group_sizes.mean(),
        'Median_Events_per_Model': group_sizes.median(),
        'Models_Dropped_Under_30': len(dropped_groups),
        'Percent_Events_Dropped': (dropped_events_count / total_events) * 100 if total_events > 0 else 0,
        'Avg_DeltaT_Variance': avg_variance,
        'Avg_DeltaT_Std_Hours': avg_std_hours,
        'Avg_DeltaT_Mean_Hours': avg_mean_hours,
        'Avg_DeltaT_Min_Hours': avg_min_hours,
        'Avg_DeltaT_Max_Hours': avg_max_hours
    }

if __name__ == "__main__":
    analyze_datasets()