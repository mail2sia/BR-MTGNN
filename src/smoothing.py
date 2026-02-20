from matplotlib import pyplot as plt
import numpy as np
import csv

def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
  
def plot_exponential_smoothing(series, alphas, RMD):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing - "+RMD)
    plt.grid(True);

def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]
    level, trend = series[0], series[1] - series[0]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result

def plot_double_exponential_smoothing(series, alphas, betas, RMD):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing - "+RMD)
    plt.grid(True)
    plt.show()

# The below script performs double exponential smoothing for the data
# Lower alpha/beta => smoother curves (moderate smoothing)
alpha = float(0.08)
beta = float(0.15)
file_name = 'data/data.csv'


def _load_csv_matrix(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError('No data rows found in CSV')

    # Drop first column if it is a date field
    data_header = header[1:] if header and header[0].lower() == 'date' else header
    data_rows = [r[1:] if header and header[0].lower() == 'date' else r for r in rows]

    data = np.asarray(data_rows, dtype=float)
    if data.ndim != 2:
        raise ValueError('Expected 2D numeric matrix')
    return data_header, data


def _smooth_matrix(raw: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    smoothed_cols = []
    for series in raw.transpose():
        dbl = np.asarray(double_exponential_smoothing(series, alpha, beta), dtype=float)
        dbl = dbl[:-1]
        if np.any(dbl < 0.0):
            single = np.asarray(exponential_smoothing(series, alpha), dtype=float)
            smoothed_cols.append(single)
        else:
            smoothed_cols.append(dbl)
    smoothed = np.asarray(smoothed_cols, dtype=float).T
    return smoothed


def _write_csv_no_header(path: str, data: np.ndarray) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data.tolist())


def _write_csv_with_header(path: str, header: list[str], data: np.ndarray) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data.tolist())


if __name__ == '__main__':
    col_names, rawdat = _load_csv_matrix(file_name)
    print(rawdat)
    print(rawdat.shape)

    smoothed = _smooth_matrix(rawdat, alpha, beta)

    _write_csv_no_header('data/sm_data.csv', smoothed)
    _write_csv_with_header('data/sm_data_g.csv', col_names, smoothed)

