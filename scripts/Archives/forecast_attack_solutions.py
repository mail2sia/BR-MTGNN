#!/usr/bin/env python3
"""
Attack-Solution Forecast Script for B-MTGNN

Forecasts future trends for attack nodes and their pertinent solution technologies.
Includes gap analysis, confidence intervals, and Bayesian uncertainty estimation.

Usage:
    python scripts/forecast_attack_solutions.py

Output:
    - Forecast plots for each attack and its solutions
    - Numerical data files with confidence intervals
    - Gap analysis CSV files
"""

import os
import sys
import csv
import numpy as np
import torch
from collections import defaultdict
from matplotlib import pyplot


# Ensure project root is on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def consistent_name(name):
    """Clean and format node names for display"""
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '')
    name = name.replace('Solution_', '').replace('_Mentions', '')
    return name.strip()


def exponential_smoothing(data, alpha=0.1):
    """Apply exponential smoothing to time series data"""
    smoothed = []
    for i in range(data.shape[1]):
        series = data[:, i]
        smooth_series = torch.zeros_like(series)
        smooth_series[0] = series[0]
        for t in range(1, len(series)):
            smooth_series[t] = alpha * series[t] + (1 - alpha) * smooth_series[t-1]
        smoothed.append(smooth_series)
    return smoothed


def getClosestCurveLarger(c, forecast, confidence, attack, solutions, col):
    """
    Returns the closest curve cc to a given curve c in a list of forecasted curves,
    where cc is strictly larger than c
    """
    d = 999999999
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if not col[j] in solutions and not col[j] == attack:  # exclude irrelevant curves
            continue
        if torch.mean(f) <= torch.mean(c):
            continue  # must be larger
        if torch.mean(f) - torch.mean(c) < d:
            d = torch.mean(f) - torch.mean(c)
            cc = f.clone()
            cc_conf = f_conf.clone()
    return cc, cc_conf


def getClosestCurveSmaller(c, forecast, confidence, attack, solutions, col):
    """
    Returns closest curve cc to given curve c in a list of forecasted curves,
    where cc is strictly smaller than c
    """
    d = 999999999
    cc = None
    cc_conf = None
    for j in range(forecast.shape[1]):
        f = forecast[:, j]
        f_conf = confidence[:, j]
        if not col[j] in solutions and not col[j] == attack:  # exclude irrelevant curves
            continue
        if torch.mean(f) >= torch.mean(c):
            continue  # must be smaller
        if torch.abs(torch.mean(f) - torch.mean(c)) < d:
            d = torch.abs(torch.mean(f) - torch.mean(c))
            cc = f.clone()
            cc_conf = f_conf.clone()
    return cc, cc_conf


def zero_negative_curves(data, forecast, attack, solutions, index):
    """Negative values (due to smoothing) are changed to 0"""
    a = data[:, index[attack]]
    f = forecast[:, index[attack]]
    for i in range(a.shape[0]):
        if a[i] < 0:
            a[i] = 0
    for i in range(f.shape[0]):
        if f[i] < 0:
            f[i] = 0

    for s in solutions:
        a = data[:, index[s]]
        f = forecast[:, index[s]]
        for i in range(a.shape[0]):
            if a[i] < 0:
                a[i] = 0
        for i in range(f.shape[0]):
            if f[i] < 0:
                f[i] = 0
    return data, forecast


def plot_forecast(data, forecast, confidence, attack, solutions, index, col, alarming=True):
    """
    Plots forecast of attack and relevant solutions trends.
    If alarming is set to True, plots the solutions trend forecasted to be less than the attack trend.
    """
    data, forecast = zero_negative_curves(data, forecast, attack, solutions, index)

    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
               "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
               "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
               "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
               "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
               "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]

    pyplot.style.use("seaborn-v0_8-dark")
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    # Plot the forecast of attack
    counter = 0
    d = torch.cat((data[:, index[attack]], forecast[0:1, index[attack]]), dim=0)  # connect the past to future in the plot
    f = forecast[:, index[attack]]
    c = confidence[:, index[attack]]
    a = consistent_name(attack)
    ax.plot(range(len(d)), d, '-', color=colours[counter], label=a, linewidth=2)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, '-', color=colours[counter], linewidth=2)
    ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=colours[counter], alpha=0.6)
    f_attack = f.clone()
    counter += 1

    # Remove technologies that we are not worried about in the future
    if alarming:
        for s in list(solutions):
            f = forecast[:, index[s]]
            if torch.mean(f) >= torch.mean(f_attack):  # solution with higher trend in the future
                solutions.remove(s)

    # Plot the forecast of the solutions
    for s in solutions:
        d = torch.cat((data[:, index[s]], forecast[0:1, index[s]]), dim=0)  # connect the past to future in the plot
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        s_name = consistent_name(s)
        ax.plot(range(len(d)), d, '-', color=colours[counter], label=s_name, linewidth=1)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, '-', color=colours[counter], linewidth=1)
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=colours[counter], alpha=0.6)
        if torch.mean(f_attack) > torch.mean(f):
            cc, cc_conf = getClosestCurveLarger(f, forecast, confidence, attack, solutions, col)  # to highlight the gap
            if cc is not None:
                ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), cc - cc_conf, f + c, color=colours[counter], alpha=0.3)
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, attack, solutions, col)
            if cc is not None:
                ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), cc + cc_conf, f - c, color=colours[counter], alpha=0.3)

        counter += 1

    x = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028']
    ax.set_xticks([6, 18, 30, 42, 54, 66, 78, 90, 102, 114, 126, 138, 150, 162, 174], x)  # positions of years on x axis

    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(a, y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)

    # Save and show the forecast
    images_dir = 'model/Bayesian/forecast/plots/'
    os.makedirs(images_dir, exist_ok=True)
    pyplot.savefig(images_dir + a.replace('/', '_') + '.png', bbox_inches="tight")
    pyplot.savefig(images_dir + a.replace('/', '_') + ".pdf", bbox_inches="tight", format='pdf')
    pyplot.show(block=False)
    pyplot.pause(5)
    pyplot.close()


def save_data(data, forecast, confidence, variance, col):
    """Saves the numerical forecast to text file as well as past data of each node"""
    file_dir = 'model/Bayesian/forecast/data/'
    os.makedirs(file_dir, exist_ok=True)
    
    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]
        with open(file_dir + name.replace('/', '_') + '.txt', 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')


def save_gap(forecast, attack, solutions, index):
    """
    Saves the forecasted trend's gap between attack and its relevant solutions to a csv file.
    The gap is for 3 years resulting in 3 values per solution.
    """
    gap_dir = 'model/Bayesian/forecast/gap/'
    os.makedirs(gap_dir, exist_ok=True)
    
    with open(gap_dir + consistent_name(attack).replace('/', '_') + '_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Solution', '2026', '2027', '2028'])
        table = []
        a = forecast[:, index[attack]].tolist()
        a_reduced = [sum(a[i:i + 12]) / 12 for i in range(0, len(a), 12)]  # mean of every 12 months
        for s in solutions:
            row = [consistent_name(s)]
            f = forecast[:, index[s]].tolist()
            f_reduced = [sum(f[i:i + 12]) / 12 for i in range(0, len(f), 12)]  # mean of every 12 months

            gap = [x - y for x, y in zip(a_reduced, f_reduced)]  # calculate the gap
            row.extend(gap)  # 3 years gap
            table.append(row)
        sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
        for row in sorted_table:
            writer.writerow(row)


def create_columns(file_name):
    """Given data file, returns the list of column names and dictionary of the format (column name,column index)"""
    col_name = []
    col_index = {}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name = col_name[1:]

        for i, c in enumerate(col_name):
            col_index[c] = i

        return col_name, col_index


def build_graph(file_name):
    """Builds the attacks and pertinent technologies graph"""
    # Initialise an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    # Read the graph CSV file
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = row[0]
            # Extract the adjacent nodes from the remaining columns
            adjacent_nodes = [node for node in row[1:] if node]  # does not include empty columns

            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with', len(graph), 'attacks...')
    return graph


def main():
    """Main forecasting pipeline"""
    
    # Configuration
    data_file = './data/sm_data_g.csv'
    model_file = 'model/Bayesian/model.pt'
    nodes_file = 'data/node.csv'
    graph_file = 'data/graph_topk_k12.csv'

    # Load the data
    fin = open(data_file)
    rawdat = np.loadtxt(fin, delimiter=',')
    n, m = rawdat.shape

    # Load column names and dictionary of (column name, index)
    col, index = create_columns(nodes_file)

    # Build the graph in the format {attack: list of pertinent technologies}
    graph = build_graph(graph_file)

    # For normalisation
    scale = np.ones(m)
    dat = np.zeros(rawdat.shape)

    # Normalise
    for i in range(m):
        scale[i] = np.max(np.abs(rawdat[:, i]))
        dat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))

    print('data shape:', dat.shape)

    # Preparing last part of the data to be used for the forecast
    P = 10  # look back
    X = torch.from_numpy(dat[-P:, :])  # look back 10 months
    X = torch.unsqueeze(X, dim=0)
    X = torch.unsqueeze(X, dim=1)
    X = X.transpose(2, 3)
    X = X.to(torch.float)

    # Load the model
    model = None
    with open(model_file, 'rb') as f:
        model = torch.load(f)

    # Bayesian estimation
    num_runs = 10

    # Create a list to store the outputs
    outputs = []

    # Use model to predict next time step
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(X)
            y_pred = output[-1, :, :, -1].clone()  # 36x142
        outputs.append(y_pred)

    # Stack the outputs along a new dimension
    outputs = torch.stack(outputs)

    Y = torch.mean(outputs, dim=0)
    variance = torch.var(outputs, dim=0)  # variance
    std_dev = torch.std(outputs, dim=0)  # standard deviation
    
    # Calculate 95% confidence interval
    z = 1.96
    confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

    dat *= scale
    Y *= scale
    variance *= scale
    confidence *= scale

    print('output shape:', Y.shape)

    # Plotting and saving results

    # Save the data to desk
    dat = torch.from_numpy(dat)
    save_data(dat, Y, confidence, variance, col)

    # Combine data
    all_data = torch.cat((dat, Y), dim=0)

    # Scale down full data (global normalisation)
    incident_max = -999999999
    mention_max = -999999999

    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1]):
            if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16, 32):
                continue
            if 'Mention' in col[j]:
                if all_data[i, j] > mention_max:
                    mention_max = all_data[i, j]
            else:
                if all_data[i, j] > incident_max:
                    incident_max = all_data[i, j]

    all_n = torch.zeros(all_data.shape[0], all_data.shape[1])
    confidence_n = torch.zeros(confidence.shape[0], confidence.shape[1])
    u = 0
    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1]):
            if 'Mention' in col[j]:
                all_n[i, j] = all_data[i, j] / mention_max
            else:
                all_n[i, j] = all_data[i, j] / incident_max

            if i >= all_data.shape[0] - 36:
                confidence_n[u, j] = confidence[u, j] * (all_n[i, j] / all_data[i, j] if all_data[i, j] != 0 else 0)
        if i >= all_data.shape[0] - 36:
            u += 1

    # Smoothing
    smoothed_dat = torch.stack(exponential_smoothing(all_n, 0.1))
    smoothed_confidence = torch.stack(exponential_smoothing(confidence_n, 0.1))

    # Plot all forecasted nodes in the graph as groups of plots
    # Each plot consists of a single attack and its pertinent technologies
    for attack, solutions in graph.items():
        plot_forecast(smoothed_dat[:-36, :], smoothed_dat[-36:, :], smoothed_confidence, attack, solutions, index, col)
        save_gap(smoothed_dat[-36:, :], attack, solutions, index)  # save gaps of each attack to file


if __name__ == '__main__':
    main()
