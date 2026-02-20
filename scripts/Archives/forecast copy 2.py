import pickle
import numpy as np
import os
import scipy.sparse as sp
import sys
import torch
import torch.nn as nn
from typing import cast
from scipy.sparse import linalg
from torch.autograd import Variable
import csv
from collections import defaultdict
from matplotlib import pyplot
import random
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.util import unwrap_model_output
from scripts.o_util import build_model_from_checkpoint, filter_state_dict_for_model

pyplot.rcParams['savefig.dpi'] = 1200
pyplot.rcParams['font.family'] = 'sans-serif'
pyplot.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
pyplot.rcParams['font.size'] = 11
pyplot.rcParams['axes.linewidth'] = 0.8
pyplot.rcParams['lines.linewidth'] = 1.6
pyplot.rcParams['xtick.direction'] = 'out'
pyplot.rcParams['ytick.direction'] = 'out'
pyplot.rcParams['xtick.major.size'] = 4
pyplot.rcParams['ytick.major.size'] = 4


def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result


def _median_filter(values, window=5):
    if window <= 1:
        return values
    half = window // 2
    padded = np.pad(values, (half, window - 1 - half), mode='edge')
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        out[i] = np.median(padded[i:i + window])
    return out


def _clip_spikes(values, lower_q=0.01, upper_q=0.99):
    if len(values) == 0:
        return values
    lo = np.quantile(values, lower_q)
    hi = np.quantile(values, upper_q)
    if lo == hi:
        return values
    return np.clip(values, lo, hi)


def _smooth_series(values, window=19, median_window=7):
    if window <= 1:
        return values
    clipped = _clip_spikes(values)
    filtered = _median_filter(clipped, window=median_window)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(filtered, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _smooth_forecast_with_anchor(history, forecast, alpha=0.1, continuity=True):
    if len(forecast) == 0:
        return forecast
    out = np.empty_like(forecast, dtype=float)
    prev = history[-1] if len(history) else forecast[0]
    for i, val in enumerate(forecast):
        prev = alpha * val + (1 - alpha) * prev
        out[i] = prev
    if continuity and len(history):
        offset = history[-1] - out[0]
        out = out + offset
    return out

def consistent_name(name):

    name=name.replace('-ALL','').replace('Mentions-','').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
    
    #special case
    if 'HIDDEN MARKOV MODEL' in name:
        return 'Statistical HMM'

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    if 'IZ' in name:
        name=name.replace('IZ', 'IS')# applicable only in our data (British English)
    if 'IOR' in name:
        name=name.replace('IOR','IOUR')#behaviour (British English)

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result


# saves the numerical forecast to text file as well as past data of each node
def save_data(data, forecast, confidence, variance, col):
    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]
        file_dir = 'model/Bayesian/forecast/data/'
        with open(file_dir + name.replace('/', '_') + '.txt', 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')


# saves the forecasted trend's gap between RMD and its relevant solutions to a csv file
def save_gap(forecast, RMD, solutions, index):
    with open('model/Bayesian/forecast/gap/' + consistent_name(RMD).replace('/', '_') + '_gap.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Solution', '2023', '2024', '2025'])
        table = []
        a = forecast[:, index[RMD]].tolist()
        a_reduced = [sum(a[i:i+12]) / 12 for i in range(0, len(a), 12)]
        for s in solutions:
            row = [consistent_name(s)]
            f = forecast[:, index[s]].tolist()
            f_reduced = [sum(f[i:i+12]) / 12 for i in range(0, len(f), 12)]
            gap = [x - y for x, y in zip(a_reduced, f_reduced)]
            row.extend(gap)
            table.append(row)
        sorted_table = sorted(table, key=lambda row: sum(row[-3:]))
        for row in sorted_table:
            writer.writerow(row)

#returns the closest curve cc to a given curve c in a list of forecasted curves, where cc is strictly larger than c
def getClosestCurveLarger(c,forecast,confidence, RMD, solutions,col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in solutions and not col[j]==RMD: #exclude irrelevant curves
            continue 
        if torch.mean(f) <= torch.mean(c):
            continue #must be larger
        if torch.mean(f)-torch.mean(c)<d:
            d=torch.mean(f)-torch.mean(c)
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf


# returns closest curve cc to given curve c in a list of forecasted curves, where cc is strictly smaller than c
def getClosestCurveSmaller(c,forecast,confidence,RMD, solutions, col):
    d=999999999
    cc=None
    cc_conf=None
    for j in range(forecast.shape[1]):
        f= forecast[:,j]
        f_conf=confidence[:,j]
        if not col[j] in solutions and not col[j]==RMD: #exclude irrelevant curves
            continue 
        if torch.mean(f) >= torch.mean(c):
            continue #must be smaller
        if torch.abs(torch.mean(f)-torch.mean(c))<d:
            d=torch.abs(torch.mean(f)-torch.mean(c))
            cc=f.clone()
            cc_conf=f_conf.clone()
    return cc,cc_conf


# negative values (due to smoothing) are changed to 0
def zero_negative_curves(data, forecast, RMD, solutions, index_map):
    a = data[:, index_map[RMD]]
    f= forecast[:, index_map[RMD]]
    for i in range(a.shape[0]):
        if a[i]<0:
            a[i]=0
    for i in range(f.shape[0]):
        if f[i]<0:
            f[i]=0

    for s in solutions:
        a = data[:, index_map[s]]
        f= forecast[:, index_map[s]]
        for i in range(a.shape[0]):
            if a[i]<0:
                a[i]=0
        for i in range(f.shape[0]):
            if f[i]<0:
                f[i]=0
    return data, forecast


# plots forecast of RMD and relevant solutions trends
def plot_forecast(data, forecast, confidence, RMD, solutions, index, col, alarming=True):
    data, forecast = zero_negative_curves(data, forecast, RMD, solutions, index)

    colours = ["RoyalBlue", "Crimson", "DarkOrange", "MediumPurple", "MediumVioletRed",
          "DodgerBlue", "Indigo", "coral", "hotpink", "DarkMagenta",
          "SteelBlue", "brown", "MediumAquamarine", "SlateBlue", "SeaGreen",
          "MediumSpringGreen", "DarkOliveGreen", "Teal", "OliveDrab", "MediumSeaGreen",
          "DeepSkyBlue", "MediumSlateBlue", "MediumTurquoise", "FireBrick",
          "DarkCyan", "violet", "MediumOrchid", "DarkSalmon", "DarkRed"]

    pyplot.style.use("seaborn-v0_8-darkgrid")
    fig, ax = pyplot.subplots(figsize=(11, 7), constrained_layout=True)

    # Plot the forecast of RMD
    counter = 0
    d = torch.cat((data[:, index[RMD]], forecast[0:1, index[RMD]]), dim=0)
    f = forecast[:, index[RMD]]
    c = confidence[:, index[RMD]]
    a = consistent_name(RMD)
    d_np = _clip_spikes(d.numpy())
    f_np = _clip_spikes(f.numpy())
    c_np = _clip_spikes(c.numpy(), lower_q=0.02, upper_q=0.98)
    d_s = _smooth_series(d_np)
    f_s = _smooth_forecast_with_anchor(d_np, _smooth_series(f_np))
    c_s = _smooth_series(c_np)
    ax.plot(range(len(d_s)), d_s, '-', color=colours[counter], label=a, linewidth=2.0)
    ax.plot(range(len(d_s) - 1, (len(d_s) + len(f_s)) - 1), f_s, '-', color=colours[counter], linewidth=2.0)
    ax.fill_between(
        range(len(d_s) - 1, (len(d_s) + len(f_s)) - 1),
        f_s - c_s,
        f_s + c_s,
        color=colours[counter],
        alpha=0.25,
    )
    f_RMD = f.clone()
    counter += 1

    # remove technologies that we are not worried about in the future
    if alarming:
        for s in list(solutions):
            f = forecast[:, index[s]]
            if torch.mean(f) >= torch.mean(f_RMD):
                solutions.remove(s)

    # Plot the forecast of the solutions
    for s in solutions:
        d = torch.cat((data[:, index[s]], forecast[0:1, index[s]]), dim=0)
        f = forecast[:, index[s]]
        c = confidence[:, index[s]]
        s = consistent_name(s)
        d_np = _clip_spikes(d.numpy())
        f_np = _clip_spikes(f.numpy())
        c_np = _clip_spikes(c.numpy(), lower_q=0.02, upper_q=0.98)
        d_s = _smooth_series(d_np)
        f_s = _smooth_forecast_with_anchor(d_np, _smooth_series(f_np))
        c_s = _smooth_series(c_np)
        ax.plot(range(len(d_s)), d_s, '-', color=colours[counter], label=s, linewidth=1.2)
        ax.plot(range(len(d_s) - 1, (len(d_s) + len(f_s)) - 1), f_s, '-', color=colours[counter], linewidth=1.2)
        ax.fill_between(
            range(len(d_s) - 1, (len(d_s) + len(f_s)) - 1),
            f_s - c_s,
            f_s + c_s,
            color=colours[counter],
            alpha=0.25,
        )
        if torch.mean(f_RMD) > torch.mean(f):
            cc, cc_conf = getClosestCurveLarger(f, forecast, confidence, RMD, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(range(len(d)-1, (len(d)+len(f))-1), cc - cc_conf, f + c, color=colours[counter], alpha=0.3)
        else:
            cc, cc_conf = getClosestCurveSmaller(f, forecast, confidence, RMD, solutions, col)
            if cc is not None and cc_conf is not None:
                ax.fill_between(range(len(d)-1, (len(d)+len(f))-1), cc + cc_conf, f - c,  color=colours[counter], alpha=0.3)

        counter += 1

    history_len = int(data.shape[0])
    forecast_len = int(forecast.shape[0])
    total_len = history_len + forecast_len
    start_year = 2004
    forecast_start_year = 2026
    history_end_year = start_year + max(0, (history_len // 12) - 1)
    forecast_years = max(1, (forecast_len + 11) // 12)
    forecast_end_year = forecast_start_year + forecast_years - 1

    history_years = list(range(start_year, history_end_year + 1, 2))
    future_years = list(range(forecast_start_year, forecast_end_year + 1, 2))
    if forecast_end_year not in future_years:
        future_years.append(forecast_end_year)
    if 2029 not in future_years:
        future_years.append(2029)

    tick_positions = []
    tick_labels = []
    for y in history_years:
        pos = (y - start_year) * 12
        if pos < total_len:
            tick_positions.append(pos)
            tick_labels.append(str(y))
    max_tick_pos = total_len - 1
    for y in future_years:
        pos = history_len + (y - forecast_start_year) * 12
        tick_positions.append(pos)
        tick_labels.append(str(y))
        if pos > max_tick_pos:
            max_tick_pos = pos

    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlim(0, max_tick_pos)

    ax.axvline(history_len - 1, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.text(history_len + 0.5, ax.get_ylim()[1], "Forecast", va="top", ha="left", fontsize=10)

    ax.set_ylabel("Trend", fontsize=12)
    ax.set_xlabel("Year", fontsize=11)
    pyplot.yticks(fontsize=10)
    ax.legend(loc="upper left", prop={'size': 9}, bbox_to_anchor=(1.0, 1.02), frameon=False)
    ax.grid(True, alpha=0.3)
    pyplot.xticks(rotation=60, fontsize=10)
    pyplot.title(a, y=1.03, fontsize=14)

    fig = pyplot.gcf()

    # save and show the forecast
    images_dir = 'model/Bayesian/forecast/plots/'
    pyplot.savefig(images_dir + a.replace('/', '_') + '.png', bbox_inches="tight")
    pyplot.savefig(images_dir + a.replace('/', '_') + ".pdf", bbox_inches="tight", format='pdf')
    try:
        backend = pyplot.get_backend().lower()
    except Exception:
        backend = ''
    if 'agg' not in backend:
        pyplot.show(block=False)
        pyplot.pause(5)
    pyplot.close()


#given data file, returns the list of column names and dictionary of the format (column name,column index)
def create_columns(file_name):

    col_name=[]
    col_index={}

    # Read the CSV file of the dataset
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Read the first row
        col_name = [c for c in next(reader)]
        if 'Date' in col_name[0]:
            col_name= col_name[1:]
        
        for i,c in enumerate(col_name):
            col_index[c]=i
        
        return col_name,col_index


#builds the RMDs and pertinent technologies graph
def _map_node_name(name, col_names):
    if name in col_names:
        return name
    if name.isdigit():
        idx = int(name)
        if 0 <= idx < len(col_names):
            return col_names[idx]
    return name

def build_graph(file_name, col_names, threshold=0.0):
    # Initialise an empty dictionary with default value as an empty list
    graph = defaultdict(list)

    try:
        adj = np.loadtxt(file_name, delimiter=',')
        if adj.ndim == 2 and adj.shape[0] == adj.shape[1]:
            n_nodes = min(adj.shape[0], len(col_names))
            for i in range(n_nodes):
                key_node = col_names[i]
                for j in range(n_nodes):
                    if i != j and adj[i, j] > threshold:
                        graph[key_node].append(col_names[j])
            print('Graph loaded with', len(graph), 'RMDs...')
            return graph
    except Exception:
        pass

    # Read the graph CSV file
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the key node from the first column
            key_node = _map_node_name(row[0], col_names)
            # Extract the adjacent nodes from the remaining columns
            adjacent_nodes = [_map_node_name(node, col_names) for node in row[1:] if node]#does not include empty columns
            
            # Add the adjacent nodes to the graph dictionary
            graph[key_node].extend(adjacent_nodes)
    print('Graph loaded with',len(graph),'RMDs...')
    return graph




def main():
    # This script forecasts the future of the graph, up to 3 years in advance
    data_file = './data/sm_data_g.csv'
    model_file = 'model/Bayesian/model.pt'
    nodes_file = 'data/data.csv'
    graph_file = 'data/graph_square.csv'
    device = 'cpu'

    # read the data
    with open(data_file, 'r') as fin:
        rawdat = np.atleast_2d(np.loadtxt(fin, delimiter=',', skiprows=1))
    n, m = rawdat.shape

    # load column names and dictionary of (column name, index)
    col, index = create_columns(data_file)
    if len(col) != m:
        col, index = create_columns(nodes_file)
        if len(col) != m:
            col = col[:m]
            index = {c: i for i, c in enumerate(col)}

    # build the graph in the format {RMD:list of pertinent technologies}
    graph = build_graph(graph_file, col)

    # for normalisation
    scale = np.ones(m, dtype=rawdat.dtype)
    dat = np.zeros((n, m), dtype=rawdat.dtype)

    # normalise
    for i in range(m):
        scale[i] = np.max(np.abs(rawdat[:, i]))
        dat[:, i] = rawdat[:, i] / np.max(np.abs(rawdat[:, i]))

    print('data shape:', dat.shape)

    # load the model
    model: nn.Module
    in_dim = 1
    P = 10
    checkpoint = torch.load(model_file, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model, inferred_seq_len, in_dim, state_dict = build_model_from_checkpoint(checkpoint, device)
        filtered_state, missing, unexpected, shape_mismatch = filter_state_dict_for_model(state_dict, model)
        if missing or unexpected or shape_mismatch:
            print('Model load warnings: missing', len(missing), 'unexpected', len(unexpected), 'shape_mismatch', len(shape_mismatch))
            if missing:
                print('Missing keys (first 8):', missing[:8])
            if unexpected:
                print('Unexpected keys (first 8):', unexpected[:8])
            if shape_mismatch:
                print('Shape mismatches (first 5):', shape_mismatch[:5])
        model.load_state_dict(filtered_state, strict=False)
        model.eval()
        P = inferred_seq_len
    else:
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model.eval()
        else:
            raise TypeError('Checkpoint does not contain a model or state_dict')

    # preparing last part of the data to be used for the forecast
    if P > n:
        raise ValueError(f'Not enough history for seq_length={P}, got {n} rows')

    if in_dim == 1:
        X = torch.from_numpy(dat[-P:, :])
        X = torch.unsqueeze(X, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
    else:
        levels = dat
        pct = np.zeros_like(levels)
        denom = np.abs(levels[:-1]) + 1e-6
        pct[1:] = (levels[1:] - levels[:-1]) / denom
        levels_tail = levels[-P:, :]
        pct_tail = pct[-P:, :]
        X_np = np.stack([levels_tail.T, pct_tail.T], axis=0)
        X = torch.from_numpy(X_np).unsqueeze(0)

    X = X.to(torch.float)

    # Bayesian estimation
    num_runs = 50
    outputs = []

    # Use model to predict next time step
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(X)
            output_tensor = unwrap_model_output(output)
            y_pred = output_tensor[-1, :, :, -1].clone()  # 36x142
        outputs.append(y_pred)

    outputs = torch.stack(outputs)

    Y = torch.mean(outputs, dim=0)
    variance = torch.var(outputs, dim=0)  # variance
    std_dev = torch.std(outputs, dim=0)  # standard deviation
    # Calculate 95% confidence interval
    z = 1.96
    confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

    dat *= scale
    scale_tensor = torch.from_numpy(scale).to(Y)
    Y = Y * scale_tensor
    variance = variance * scale_tensor
    confidence = confidence * scale_tensor

    print('output shape:', Y.shape)

    # save the data to desk
    dat_tensor = torch.from_numpy(dat).float()
    save_data(dat_tensor, Y, confidence, variance, col)

    # combine data
    all = torch.cat((dat_tensor, Y), dim=0)

    # scale down full data (global normalisation)
    incident_max = -999999999
    mention_max = -999999999

    for i in range(all.shape[0]):
        for j in range(all.shape[1]):
            if 'WAR' in col[j] or 'Holiday' in col[j] or j in range(16, 32):
                continue
            if 'Mention' in col[j]:
                if all[i, j] > mention_max:
                    mention_max = all[i, j]
            else:
                if all[i, j] > incident_max:
                    incident_max = all[i, j]

    all_n = torch.zeros(all.shape[0], all.shape[1])
    confidence_n = torch.zeros(confidence.shape[0], confidence.shape[1])
    u = 0
    for i in range(all.shape[0]):
        for j in range(all.shape[1]):
            if 'Mention' in col[j]:
                all_n[i, j] = all[i, j] / mention_max
            else:
                all_n[i, j] = all[i, j] / incident_max

            if i >= all.shape[0] - 36:
                confidence_n[u, j] = confidence[u, j] * (all_n[i, j] / all[i, j])
        if i >= all.shape[0] - 36:
            u += 1

    # smoothing (used for gap reporting only)
    smoothed_dat = torch.stack(exponential_smoothing(all_n, 0.1))
    smoothed_confidence = torch.stack(exponential_smoothing(confidence_n, 0.1))

    # Plot relative to robust history scale to match benchmark-style normalization.
    hist_vals = torch.abs(dat_tensor)
    hist_scale = torch.quantile(hist_vals, 0.95, dim=0)
    hist_scale = torch.where(hist_scale == 0, torch.ones_like(hist_scale), hist_scale)
    plot_all = all / hist_scale
    plot_confidence = confidence / hist_scale

    # plot all forecasted nodes in the graph as groups of plots
    has_rmd_prefix = any(name.startswith('RMD_') for name in col)
    for RMD, solutions in graph.items():
        if has_rmd_prefix and not RMD.startswith('RMD_'):
            continue
        plot_forecast(plot_all[:-36, ], plot_all[-36:, ], plot_confidence, RMD, solutions, index, col)
        save_gap(smoothed_dat[-36:, ], RMD, solutions, index)


if __name__ == '__main__':
    main()
    


