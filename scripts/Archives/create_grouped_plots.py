#!/usr/bin/env python3
"""
Create grouped forecast plots similar to the cybersecurity examples.
Shows disorders with related treatments on the same plot.

Usage:
    python scripts/create_grouped_plots.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    """Simple visual smoothing via centered moving average (edge-padded)."""
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or y.size < 3:
        return y
    window = int(window)
    if y.size < window:
        return y
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    y_pad = np.pad(y, (pad_left, pad_right), mode='edge')
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y_pad, kernel, mode='valid')

# Define disorder-treatment groupings (clinical relationships)
DISORDER_TREATMENT_GROUPS = {
    # Personality & Mood Disorders
    "RMD_Antisocial Personality Disorder": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Social Skills Training",
        "PT_Psychoeducation Programs",
        "PT_Trauma-Focused CBT"
    ],
    "RMD_Narcissistic Personality": [
        "PT_Interpersonal Psychotherapy",
        "PT_Cognitive Behavioral Therapy",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Histrionic Personality": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Interpersonal Psychotherapy",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Avoidant Personality": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Exposure Therapy",
        "PT_Social Skills Training",
        "PT_Virtual Reality Exposure Therapy"
    ],
    "RMD_Schizotypal Personality": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Antipsychotic Medications",
        "PT_Social Skills Training"
    ],
    
    # Psychotic Disorders
    "RMD_Schizoaffective": [
        "PT_Antipsychotic Medications",
        "PT_Mood Stabilizers",
        "PT_Antidepressants",
        "PT_Cognitive Behavioral Therapy",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Brief Psychotic": [
        "PT_Antipsychotic Medications",
        "PT_Benzodiazepines",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Delusional": [
        "PT_Antipsychotic Medications",
        "PT_Cognitive Behavioral Therapy",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Hallucinogen-Induced Psychotic": [
        "PT_Antipsychotic Medications",
        "PT_Benzodiazepines",
        "PT_Cognitive Behavioral Therapy"
    ],
    "RMD_Postpartum Psychosis": [
        "PT_Antipsychotic Medications",
        "PT_Mood Stabilizers",
        "PT_Electroencephalography"
    ],
    
    # Delusional Syndromes
    "RMD_Capgras": [
        "PT_Antipsychotic Medications",
        "PT_Cognitive Behavioral Therapy",
        "PT_Brain Stimulation Technologies",
        "PT_Digital Imaging Technologies for Brain Scanning"
    ],
    "RMD_Fregoli Delusion": [
        "PT_Antipsychotic Medications",
        "PT_Cognitive Behavioral Therapy",
        "PT_Brain Stimulation Technologies"
    ],
    "RMD_Othello": [
        "PT_Antipsychotic Medications",
        "PT_Cognitive Behavioral Therapy",
        "PT_Interpersonal Psychotherapy"
    ],
    "RMD_Erotomanic": [
        "PT_Antipsychotic Medications",
        "PT_Cognitive Behavioral Therapy",
        "PT_Psychoeducation Programs"
    ],
    
    # Eating Disorders
    "RMD_Anorexia nervosa": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Dietary Interventions & Supplements",
        "PT_Interpersonal Psychotherapy",
        "PT_Antidepressants",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Rumination": [
        "PT_Behavioral Activation Therapy",
        "PT_Cognitive Behavioral Therapy",
        "PT_Dietary Interventions & Supplements"
    ],
    
    # Childhood Developmental Disorders
    "RMD_Childhood Disintegrative Disorder": [
        "PT_Behavioral Activation Therapy",
        "PT_Social Skills Training",
        "PT_Cognitive Enhancement Programs",
        "PT_Animal-Assisted Interventions"
    ],
    "RMD_Selective Mutism": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Exposure Therapy",
        "PT_Social Skills Training",
        "PT_Virtual Reality Exposure Therapy"
    ],
    
    # Attachment & Behavioral Disorders
    "RMD_Disinhibited Social Engagement": [
        "PT_Behavioral Activation Therapy",
        "PT_Trauma-Focused CBT",
        "PT_Animal-Assisted Interventions",
        "PT_Social Skills Training"
    ],
    "RMD_Reactive Attachment": [
        "PT_Trauma-Focused CBT",
        "PT_Behavioral Activation Therapy",
        "PT_Animal-Assisted Interventions",
        "PT_Interpersonal Psychotherapy"
    ],
    
    # Impulse Control Disorders
    "RMD_Impulse Control": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Mood Stabilizers",
        "PT_Antidepressants",
        "PT_Mindfulness-Based Stress Reduction"
    ],
    "RMD_Kleptomania": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Antidepressants",
        "PT_Mood Stabilizers"
    ],
    "RMD_Pyromania": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Behavioral Activation Therapy",
        "PT_Antidepressants"
    ],
    
    # Sleep Disorders
    "RMD_Circadian Rhythm Sleep-Wake": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Meditation and Relaxation Techniques",
        "PT_Mindfulness-Based Stress Reduction",
        "PT_Biofeedback Devices"
    ],
    "RMD_Kleine-Levin": [
        "PT_Mood Stabilizers",
        "PT_Neurocognitive Training",
        "PT_Electroencephalography"
    ],
    
    # Neurological/Cognitive Disorders
    "RMD_Dementia pugilistica": [
        "PT_Neuroprotective Agents",
        "PT_Cognitive Enhancement Programs",
        "PT_Regenerative Medicine Approaches",
        "PT_Stem Cell Therapy for Neurological Disorders"
    ],
    "RMD_Hyperthymesia": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Neurocognitive Training",
        "PT_Brain-Computer Interface"
    ],
    "RMD_Savant": [
        "PT_Cognitive Enhancement Programs",
        "PT_Neurocognitive Training",
        "PT_Social Skills Training"
    ],
    
    # Mood Disorders
    "RMD_Cyclothymic": [
        "PT_Mood Stabilizers",
        "PT_Cognitive Behavioral Therapy",
        "PT_Interpersonal Psychotherapy",
        "PT_Mindfulness-Based Stress Reduction"
    ],
    
    # Memory & Dissociative Disorders
    "RMD_Dissociative Fugue": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Eye Movement Desensitization and Reprocessing",
        "PT_Trauma-Focused CBT",
        "PT_Psychoeducation Programs"
    ],
    "RMD_Psychogenic Amnesia": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Eye Movement Desensitization and Reprocessing",
        "PT_Trauma-Focused CBT"
    ],
    "RMD_Paramnesia": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Neurocognitive Training",
        "PT_Digital Imaging Technologies for Brain Scanning"
    ],
    "RMD_Reduplicative Paramnesia": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Neurocognitive Training",
        "PT_Antipsychotic Medications"
    ],
    
    # Factitious Disorders
    "RMD_Factitious": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Psychoeducation Programs",
        "PT_Interpersonal Psychotherapy"
    ],
    "RMD_Ganser": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Psychoeducation Programs",
        "PT_Antipsychotic Medications"
    ],
    
    # Rare Syndromes
    "RMD_Diogenes": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Antidepressants",
        "PT_Social Skills Training"
    ],
    "RMD_Ekbom": [
        "PT_Antipsychotic Medications",
        "PT_Cognitive Behavioral Therapy",
        "PT_Antidepressants"
    ],
    "RMD_Olfactory Reference": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Antidepressants",
        "PT_Exposure Therapy"
    ],
    "RMD_Walking Corpse": [
        "PT_Antipsychotic Medications",
        "PT_Antidepressants",
        "PT_Electroencephalography",
        "PT_Brain Stimulation Technologies"
    ],
    "RMD_Stendhal": [
        "PT_Cognitive Behavioral Therapy",
        "PT_Benzodiazepines",
        "PT_Mindfulness-Based Stress Reduction"
    ]
}


def load_forecast_data(filepath):
    """Load forecast data from text file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data = {}
    for line in lines:
        if line.startswith('Node:'):
            data['node'] = line.split(':', 1)[1].strip()
        elif line.startswith('Historical Data:'):
            data['historical'] = ast.literal_eval(line.split(':', 1)[1].strip())
        elif line.startswith('Forecast:'):
            data['forecast'] = ast.literal_eval(line.split(':', 1)[1].strip())
        elif line.startswith('95% Confidence:'):
            data['confidence'] = ast.literal_eval(line.split(':', 1)[1].strip())
    
    return data


def create_grouped_plot(disorder_name, treatment_names, data_dir, output_dir):
    """Create a plot showing disorder with related treatments"""
    
    # Load disorder data
    disorder_file = os.path.join(data_dir, f"{disorder_name}.txt")
    if not os.path.exists(disorder_file):
        print(f"Warning: {disorder_name} data not found")
        return
    
    disorder_data = load_forecast_data(disorder_file)
    
    # Load treatment data
    treatment_data_list = []
    valid_treatments = []
    for treatment in treatment_names:
        treatment_file = os.path.join(data_dir, f"{treatment}.txt")
        if os.path.exists(treatment_file):
            treatment_data_list.append(load_forecast_data(treatment_file))
            valid_treatments.append(treatment)
    
    if not valid_treatments:
        print(f"Warning: No treatment data found for {disorder_name}")
        return
    
    # Create plot (match the example grouped plots styling)
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12.5, 7))
    
    # Time axis setup
    hist_data = np.array(disorder_data['historical'])
    fore_data = np.array(disorder_data['forecast'])
    conf_data = np.array(disorder_data['confidence'])

    # Visual-only smoothing (does not affect saved forecast numbers)
    try:
        smooth_window = int(os.environ.get('BMTGNN_PLOT_SMOOTH_WINDOW', '3'))
    except Exception:
        smooth_window = 3
    if smooth_window < 1:
        smooth_window = 1

    hist_data_s = smooth_series(hist_data, smooth_window)
    fore_data_s = smooth_series(fore_data, smooth_window)
    conf_data_s = np.maximum(smooth_series(conf_data, smooth_window), 0.0)
    
    T_hist = len(hist_data)
    T_fore = len(fore_data)
    
    # Create x-axis (monthly from 2004)
    start_year = 2004
    x_hist = np.arange(T_hist)
    x_fore = np.arange(T_hist, T_hist + T_fore)
    # Include the last historical point so forecast lines connect smoothly
    x_fore_conn = np.arange(T_hist - 1, T_hist + T_fore)
    
    # Style tweaks to match example plots
    try:
        forecast_alpha = float(os.environ.get('BMTGNN_PLOT_FORECAST_ALPHA', '0.18'))
    except Exception:
        forecast_alpha = 0.18
    line_main = 2.5
    line_other = 1.5

    # Color palette similar to examples
    colors = ['#8B0000', '#FF8C00', '#9932CC', '#008B8B', '#228B22', 
              '#DC143C', '#4B0082', '#FF1493', '#00CED1', '#32CD32']
    
    # Plot disorder (main metric in blue with CI)
    disorder_display = disorder_name.replace('RMD_', '').replace('_', ' ')
    ax.plot(
        x_hist,
        hist_data_s,
        '-',
        color='RoyalBlue',
        linewidth=line_main,
        label=disorder_display,
        alpha=0.9,
        zorder=3,
    )
    disorder_fore_with_conn = np.concatenate([[hist_data_s[-1]], fore_data_s])
    ax.plot(
        x_fore_conn,
        disorder_fore_with_conn,
        '-',
        color='RoyalBlue',
        linewidth=line_main,
        alpha=0.9,
        zorder=3,
    )
    # Forecast fill (example-style translucent region)
    ax.fill_between(
        x_fore,
        0,
        fore_data_s,
        color='RoyalBlue',
        alpha=forecast_alpha,
        linewidth=0,
        zorder=1.5,
    )
    
    # Plot related treatments
    for idx, (treatment, tdata) in enumerate(zip(valid_treatments, treatment_data_list)):
        color = colors[idx % len(colors)]
        treatment_display = treatment.replace('PT_', '').replace('_', ' ')
        
        t_hist = np.array(tdata['historical'])
        t_fore = np.array(tdata['forecast'])

        t_hist_s = smooth_series(t_hist, smooth_window)
        t_fore_s = smooth_series(t_fore, smooth_window)
        
        # Plot historical as lines
        ax.plot(
            x_hist,
            t_hist_s,
            '-',
            color=color,
            linewidth=line_other,
            label=treatment_display,
            alpha=0.85,
            zorder=2.5,
        )
        
        # Plot forecast with connection + filled area (forecast region only)
        t_fore_with_conn = np.concatenate([[t_hist_s[-1]], t_fore_s])
        ax.plot(
            x_fore_conn,
            t_fore_with_conn,
            '-',
            color=color,
            linewidth=line_other,
            alpha=0.85,
            zorder=2.5,
        )
        # Forecast shading (example-style)
        ax.fill_between(
            x_fore,
            0,
            t_fore_s,
            color=color,
            alpha=forecast_alpha,
            linewidth=0,
            zorder=1,
        )
    
    # Generate year labels for x-axis
    tick_positions = []
    tick_labels = []
    total_steps = T_hist + T_fore
    
    for step in range(0, total_steps, 12):
        year = start_year + step // 12
        tick_positions.append(step)
        tick_labels.append(str(year))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=11)
    
    # Styling
    ax.set_ylabel('Trend', fontsize=14)
    ax.set_xlabel('')
    ax.set_title(disorder_display, fontsize=22, pad=12)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=10,
        frameon=False,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, T_hist + T_fore)
    
    # Ensure non-negative values
    ax.set_ylim(bottom=0)
    
    # Make room for the outside legend
    fig.subplots_adjust(right=0.78)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    safe_name = disorder_name.replace('/', '_').replace('\\', '_')
    
    plt.savefig(os.path.join(output_dir, f'{safe_name}.png'), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, f'{safe_name}.pdf'), 
                bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f'Created grouped plot for {disorder_name}')


def main():
    """Generate all grouped plots"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'model' / 'Bayesian' / 'forecast' / 'data'
    output_dir = base_dir / 'model' / 'Bayesian' / 'forecast' / 'grouped_plots'
    
    print("=" * 80)
    print("Grouped Forecast Plots Generator")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Generating {len(DISORDER_TREATMENT_GROUPS)} grouped plots...")
    print()
    
    for disorder, treatments in DISORDER_TREATMENT_GROUPS.items():
        create_grouped_plot(disorder, treatments, data_dir, output_dir)
    
    print()
    print("=" * 80)
    print("Grouped plots complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
