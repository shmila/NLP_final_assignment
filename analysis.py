import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from os.path import join

from metrics import calculate_coherence_score_sliding_window, calculate_coherence_metric


def open_questions_full_analysis(control_valid_answers, patient_valid_answers, tokenizer, embedding_model, k,
                                 sliding_window=True, content_words=False):
    # Determine the directory based on content_words and sliding_window flags
    base_dir = "image_description_answer_results"
    # base_dir = "results_jsons"

    content_dir = "content_words" if content_words else "all_words"
    method_dir = "sliding_window_coherence_metric_calculation" if sliding_window else "vanilla_coherence_metric_calculation "
    save_dir = os.path.join(base_dir, content_dir, method_dir)

    # Create the directories if they do not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate coherence scores based on the method chosen
    if sliding_window:
        control_scores = calculate_coherence_score_sliding_window(control_valid_answers, k, tokenizer, embedding_model)
        patient_scores = calculate_coherence_score_sliding_window(patient_valid_answers, k, tokenizer, embedding_model)
    else:
        control_scores = calculate_coherence_metric(control_valid_answers, k, tokenizer, embedding_model)
        patient_scores = calculate_coherence_metric(patient_valid_answers, k, tokenizer, embedding_model)

    control_df = pd.DataFrame(control_scores)
    patient_df = pd.DataFrame(patient_scores)
    control_df['group'] = 'Control'
    patient_df['group'] = 'Patient'
    combined_df = pd.concat([control_df, patient_df])

    # Save histograms
    plt.figure(figsize=(14, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        sns.histplot(control_df[control_df['question_index'] == i]['score'], kde=True, color='blue',
                     label='Control Group', bins=20)
        sns.histplot(patient_df[patient_df['question_index'] == i]['score'], kde=True, color='red',
                     label='Patient Group', bins=20)
        plt.title(f'Distribution of Scores for Question {i + 1}')
        plt.xlabel('Coherence Score')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.suptitle(f'Score Distribution per Question for k={k}', y=1.02)
    plt.savefig(os.path.join(save_dir, f'score_distribution_per_question_k_{k}.png'))
    plt.close()

    # Save box plots per question
    plt.figure(figsize=(14, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='group', y='score', data=combined_df[combined_df['question_index'] == i])
        plt.title(f'Box Plot of Scores for Question {i + 1}')
        plt.xlabel('Group')
        plt.ylabel('Coherence Score')
    plt.tight_layout()
    plt.suptitle(f'Box Plot per Question for k={k}', y=1.02)
    plt.savefig(os.path.join(save_dir, f'box_plot_per_question_k_{k}.png'))
    plt.close()

    # Save overall box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='group', y='score', data=combined_df)
    plt.title(f'Overall Box Plot of Coherence Scores for k={k}')
    plt.xlabel('Group')
    plt.ylabel('Coherence Score')
    plt.savefig(os.path.join(save_dir, f'overall_box_plot_k_{k}.png'))
    plt.close()

    results = []

    print(f"T-Test Results per Question for k={k}:")
    for i in range(4):
        control_scores_q = control_df[control_df['question_index'] == i]['score'].dropna()
        patient_scores_q = patient_df[patient_df['question_index'] == i]['score'].dropna()
        t_stat, p_value = ttest_ind(control_scores_q, patient_scores_q, alternative='greater', equal_var=True)
        control_mean = control_scores_q.mean()
        patient_mean = patient_scores_q.mean()
        results.append({
            'k': k,
            'question': i + 1,
            'test': 'T-Test',
            't_stat': t_stat,
            'p_value': p_value,
            'control_mean': control_mean,
            'patient_mean': patient_mean
        })
        print(
            f"Question {i + 1}: t-statistic = {t_stat}, p-value = {p_value}, control mean = {control_mean}, patient mean = {patient_mean}")

    control_mean = control_df['score'].mean()
    patient_mean = patient_df['score'].mean()
    t_stat, p_value = ttest_ind(control_df['score'].dropna(), patient_df['score'].dropna(), alternative='greater',
                                equal_var=True)
    results.append({
        'k': k,
        'question': 'Overall',
        'test': 'T-Test',
        't_stat': t_stat,
        'p_value': p_value,
        'control_mean': control_mean,
        'patient_mean': patient_mean
    })
    print(
        f"\nOverall T-Test Results for k={k}: t-statistic = {t_stat}, p-value = {p_value}, control mean = {control_mean}, patient mean = {patient_mean}\n")

    print(f"Welch's Test Results per Question for k={k}:")
    for i in range(4):
        control_scores_q = control_df[control_df['question_index'] == i]['score'].dropna()
        patient_scores_q = patient_df[patient_df['question_index'] == i]['score'].dropna()
        t_stat, p_value = ttest_ind(control_scores_q, patient_scores_q, alternative='greater', equal_var=False)
        results.append({
            'k': k,
            'question': i + 1,
            'test': 'Welch\'s Test',
            't_stat': t_stat,
            'p_value': p_value
        })
        print(f"Question {i + 1}: t-statistic = {t_stat}, p-value = {p_value}")

    t_stat, p_value = ttest_ind(control_df['score'].dropna(), patient_df['score'].dropna(), alternative='greater',
                                equal_var=False)
    results.append({
        'k': k,
        'question': 'Overall',
        'test': 'Welch\'s Test',
        't_stat': t_stat,
        'p_value': p_value
    })
    print(f"\nOverall Welch's Test Results for k={k}: t-statistic = {t_stat}, p-value = {p_value}\n")

    print(f"Mann-Whitney U Test Results per Question for k={k}:")
    for i in range(4):
        control_scores_q = control_df[control_df['question_index'] == i]['score'].dropna()
        patient_scores_q = patient_df[patient_df['question_index'] == i]['score'].dropna()
        if len(control_scores_q) > 0 and len(patient_scores_q) > 0:
            stat, p_value = mannwhitneyu(control_scores_q, patient_scores_q, alternative='greater')
            results.append({
                'k': k,
                'question': i + 1,
                'test': 'Mann-Whitney U Test',
                'u_stat': stat,
                'p_value': p_value
            })
            print(f"Question {i + 1}: U statistic = {stat}, p-value = {p_value}")
        else:
            print(f"Question {i + 1}: Not enough data for Mann-Whitney U test.")

    try:
        stat, p_value = mannwhitneyu(control_df['score'].dropna(), patient_df['score'].dropna(), alternative='greater')
        results.append({
            'k': k,
            'question': 'Overall',
            'test': 'Mann-Whitney U Test',
            'u_stat': stat,
            'p_value': p_value
        })
        print(f"\nOverall Mann-Whitney U Test Results for k={k}: U statistic = {stat}, p-value = {p_value}\n")
    except ValueError as e:
        print(f"Overall: Error in Mann-Whitney U test: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(save_dir, f'statistical_tests_results_k_{k}.csv'), index=False)


def TAT_questions_full_analysis(control_valid_answers, patient_valid_answers, tokenizer, embedding_model, k,
                                sliding_window=True, content_words=False):
    # Determine the directory based on content_words and sliding_window flags
    base_dir = "image_description_answer_results"
    content_dir = "content_words" if content_words else "all_words"
    method_dir = "sliding_window_coherence_metric_calculation" if sliding_window else "vanilla_coherence_metric_calculation"
    save_dir = os.path.join(base_dir, content_dir, method_dir)

    # Create the directories if they do not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Calculate coherence scores based on the method chosen
    if sliding_window:
        print("Processing control answers...")
        control_scores = calculate_coherence_score_sliding_window(control_valid_answers, k, tokenizer, embedding_model)
        print("Processing patient answers...")
        patient_scores = calculate_coherence_score_sliding_window(patient_valid_answers, k, tokenizer, embedding_model)
    else:
        print("Processing control answers...")
        control_scores = calculate_coherence_metric(control_valid_answers, k, tokenizer, embedding_model)
        print("Processing patient answers...")
        patient_scores = calculate_coherence_metric(patient_valid_answers, k, tokenizer, embedding_model)

    # Create DataFrames for control and patient groups
    control_df = pd.DataFrame(control_scores)
    patient_df = pd.DataFrame(patient_scores)

    control_df['group'] = 'Control'
    patient_df['group'] = 'Patient'
    metric_name = "sliding_window" if sliding_window else "vanilla"
    combined_df = pd.concat([control_df, patient_df])

    control_csv_name = f'control_image_description_answers_scores_{metric_name}_k_{k}.csv'
    patient_csv_name = f'patient_image_description_answers_scores_{metric_name}_k_{k}.csv'
    combined_csv_name = f'image_description_answers_scores_{metric_name}_k_{k}.csv'

    # Save DataFrames as CSV files
    control_df.to_csv(join(save_dir, control_csv_name), index=False)
    patient_df.to_csv(join(save_dir, patient_csv_name), index=False)
    combined_df.to_csv(join(save_dir, combined_csv_name), index=False)

    # Ensure no empty sequences are passed to plotting functions
    if control_df.empty or patient_df.empty:
        print(f"Warning: One of the DataFrames is empty. Skipping plots and statistical analysis for k={k}.")
        return

    num_questions = 14

    # Statistical analysis
    control_mean = control_df['score'].mean()
    patient_mean = patient_df['score'].mean()

    print(f"Statistical Analysis Results per Question for k={k}:")

    for i in range(num_questions):
        control_scores_q = control_df[control_df['question_index'] == i]['score'].dropna()
        patient_scores_q = patient_df[patient_df['question_index'] == i]['score'].dropna()

        if control_scores_q.empty or patient_scores_q.empty:
            print(f"Warning: Not enough data for Question {i + 1}. Skipping this question.")
            continue

        t_stat, p_value = ttest_ind(control_scores_q, patient_scores_q, alternative='greater', equal_var=True)
        t_stat_welch, p_value_welch = ttest_ind(control_scores_q, patient_scores_q, alternative='greater',
                                                equal_var=False)

        try:
            u_stat_mann_whitney, p_value_mann_whitney = mannwhitneyu(control_scores_q, patient_scores_q,
                                                                     alternative='greater')
            print(f"Question {i + 1}: U statistic = {u_stat_mann_whitney}, p-value = {p_value_mann_whitney}")
        except ValueError as e:
            print(f"Warning: Mann-Whitney U Test failed for Question {i + 1} due to: {e}")

        print(
            f"Question {i + 1}: t-statistic = {t_stat}, p-value = {p_value}, \n"
            f"control mean = {control_scores_q.mean()},\n"
            f" patient mean = {patient_scores_q.mean()}")

        print(f"Question {i + 1}: Welch's t-statistic = {t_stat_welch}, p-value = {p_value_welch}")

    # Overall statistical analysis
    print(f"Overall Statistical Analysis Results for k={k}:")

    if not control_df['score'].dropna().empty and not patient_df['score'].dropna().empty:
        overall_t_stat, overall_p_value = ttest_ind(control_df['score'].dropna(), patient_df['score'].dropna(),
                                                    alternative='greater',
                                                    equal_var=True)
        overall_t_stat_welch, overall_p_value_welch = ttest_ind(control_df['score'].dropna(),
                                                                patient_df['score'].dropna(),
                                                                alternative='greater',
                                                                equal_var=False)
        overall_u_stat, overall_p_value_mann_whitney = mannwhitneyu(control_df['score'].dropna(),
                                                                    patient_df['score'].dropna(),
                                                                    alternative='greater')

        print(
            f"\nOverall T-Test Results for k={k}: t-statistic = {overall_t_stat}, p-value = {overall_p_value}, "
            f"control mean = {control_mean}, patient mean = {patient_mean}\n")

        print(
            f"\nOverall Welch's Test Results for k={k}: t-statistic = {overall_t_stat_welch}, p-value = {overall_p_value_welch}\n")

        print(
            f"\nOverall Mann-Whitney U Test Results for k={k}: U statistic = {overall_u_stat}, p-value = {overall_p_value_mann_whitney}\n")
    else:
        print(f"Warning: Not enough data for overall statistical analysis for k={k}.")
