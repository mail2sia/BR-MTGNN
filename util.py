from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def set_random_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def pick_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = int(device_name.split(":")[-1]) if ":" in device_name else 0
            if idx < torch.cuda.device_count():
                return torch.device(device_name)
            fallback = torch.device("cuda:0")
            print(f"Warning: {device_name} not available (only {torch.cuda.device_count()} GPU(s)); using {fallback}")
            return fallback
        except (ValueError, RuntimeError):
            return torch.device("cuda:0")
    return torch.device("cpu")


def _try_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _sniff_first_line(path: str | Path) -> Tuple[str, bool]:
    with open(path, "r", encoding="utf-8-sig") as f:
        line = f.readline().strip()
    comma = line.count(",")
    tab = line.count("\t")
    sep = "\t" if tab > comma else ","
    toks = [t.strip() for t in line.split(sep)]
    has_header = not all(_try_float(t) for t in toks if t != "")
    return sep, has_header


def is_date_column(col: str) -> bool:
    c = str(col).strip().lower()
    return c in {"date", "month", "month-year", "time", "timestamp", "ds"} or c.startswith("date")


def normalize_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def strip_signal_suffix(name: str) -> str:
    text = str(name).strip()
    text = re.sub(r"([_\-\s]+)?NoM$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([_\-\s]+)?NoP$", "", text, flags=re.IGNORECASE)
    return text.strip(" _-")


def is_nom_column(col: str) -> bool:
    c = str(col).strip()
    return bool(re.search(r"_NoM$", c, flags=re.IGNORECASE))


def is_nop_column(col: str) -> bool:
    c = str(col).strip()
    return bool(re.search(r"_NoP$", c, flags=re.IGNORECASE))


def is_global_column(col: str, global_regex: str = "") -> bool:
    c = str(col).strip()
    low = c.lower()
    prefixes = ("global_", "global", "g_", "exo_", "cov_", "context_", "war_")
    if low.startswith(prefixes):
        return True
    if global_regex:
        return bool(re.search(global_regex, c, flags=re.IGNORECASE))
    return False


def _parse_monthly_dates(series: pd.Series) -> pd.DatetimeIndex:
    as_str = series.astype(str).str.strip()
    parsed = pd.to_datetime(as_str, format="%b-%y", errors="coerce")
    if parsed.notna().mean() < 0.8:
        parsed = pd.to_datetime(as_str, format="%Y-%m", errors="coerce")
    if parsed.notna().mean() < 0.8:
        parsed = pd.to_datetime(as_str, format="%Y/%m/%d", errors="coerce")
    if parsed.notna().mean() < 0.8:
        parsed = pd.to_datetime(as_str, errors="coerce")
    if parsed.notna().sum() == 0:
        raise ValueError("Date column exists but could not be parsed as monthly datetime.")
    return pd.DatetimeIndex(parsed.ffill().bfill())


def read_table_flexible(path: str | Path, nodes_csv: str | Path | None = None) -> Tuple[pd.DataFrame, Optional[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    sep, has_header = _sniff_first_line(path)
    if has_header:
        df = pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=sep, header=None, engine="python", encoding="utf-8-sig")

    date_col = None
    for col in df.columns:
        if is_date_column(str(col)):
            date_col = str(col)
            break

    if not has_header:
        if nodes_csv is None or not Path(nodes_csv).exists():
            raise ValueError(
                "Input data has no header. Provide --nodes_csv with matching column names so NoM/NoP/Global columns can be detected."
            )
        node_path = Path(nodes_csv)
        nsep, nheader = _sniff_first_line(node_path)
        node_df = (
            pd.read_csv(node_path, sep=nsep, engine="python", encoding="utf-8-sig")
            if nheader
            else pd.read_csv(node_path, sep=nsep, header=None, engine="python", encoding="utf-8-sig")
        )
        node_cols = list(node_df.columns) if nheader else list(node_df.iloc[0].astype(str))
        if node_cols and is_date_column(node_cols[0]) and len(node_cols) == df.shape[1] + 1:
            df.columns = node_cols[1:]
            if len(node_df) == len(df):
                df.insert(0, node_cols[0], node_df.iloc[:, 0].values)
                date_col = node_cols[0]
        elif len(node_cols) == df.shape[1]:
            df.columns = node_cols
        else:
            raise ValueError(
                "nodes_csv columns do not match data_csv width. Cannot infer column names for headerless data."
            )

    if date_col is None and nodes_csv and Path(nodes_csv).exists():
        nsep, nheader = _sniff_first_line(nodes_csv)
        node_df = (
            pd.read_csv(nodes_csv, sep=nsep, engine="python", encoding="utf-8-sig")
            if nheader
            else pd.read_csv(nodes_csv, sep=nsep, header=None, engine="python", encoding="utf-8-sig")
        )
        if len(node_df) == len(df):
            first = str(node_df.columns[0])
            if is_date_column(first):
                # Use pd.concat instead of df.insert to avoid DataFrame fragmentation
                df = pd.concat([node_df.iloc[:, [0]], df], axis=1)
                date_col = first

    for col in df.columns:
        if str(col) == str(date_col):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0.0)
    return df, date_col


@dataclass
class FeatureSchema:
    date_col: Optional[str]
    entity_names: List[str]
    entity_types: List[str]
    nom_cols: List[Optional[str]]
    nop_cols: List[Optional[str]]
    global_cols: List[str]
    input_channels: List[str]
    target_channels: List[str]
    node_count: int
    rmd_nom_indices: List[int]
    pt_nom_indices: List[int]
    nop_indices: List[int]


@dataclass
class TransformConfig:
    log1p: bool
    clip_nonnegative: bool
    nom_scale: List[float]
    nop_scale: List[float]
    global_scale: List[float]


@dataclass
class SplitConfig:
    train_ratio: float
    valid_ratio: float
    seq_in_len: int
    seq_out_len: int
    train_samples: int
    valid_samples: int
    test_samples: int
    train_end_date: str = ""
    val_end_date: str = ""


class RMDPTData:
    def __init__(
        self,
        data_csv: str,
        graph_csv: str,
        nodes_csv: str | None,
        seq_in_len: int,
        seq_out_len: int,
        train_ratio: float,
        valid_ratio: float,
        batch_size: int,
        device: torch.device,
        global_regex: str = "",
        log1p: bool = True,
        clip_nonnegative: bool = True,
        use_bai: bool = False,
        bai_nom_weight: float = 0.7,
        bai_nop_weight: float = 0.3,
        use_tdb: bool = False,
        train_end_date: str = "",
        val_end_date: str = "",
    ):
        self.data_csv = data_csv
        self.graph_csv = graph_csv
        self.nodes_csv = nodes_csv
        self.seq_in_len = int(seq_in_len)
        self.seq_out_len = int(seq_out_len)
        self.train_ratio = float(train_ratio)
        self.valid_ratio = float(valid_ratio)
        self.batch_size = int(batch_size)
        self.device = device
        self.global_regex = global_regex
        self.log1p = bool(log1p)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.use_bai = bool(use_bai)
        self.bai_nom_weight = float(bai_nom_weight)
        self.bai_nop_weight = float(bai_nop_weight)
        self.use_tdb = bool(use_tdb)
        # Date-based split cutoffs (YYYY-MM format). When set, these override
        # ratio-based splitting so that no window's target period crosses a
        # boundary — eliminating target-period leakage between splits.
        self.train_end_date = train_end_date  # last month of training targets
        self.val_end_date = val_end_date      # last month of validation targets

        self.raw_df, self.date_col = read_table_flexible(data_csv, nodes_csv)
        self.dates = self._make_dates()
        self.numeric_cols = [c for c in self.raw_df.columns if c != self.date_col]
        if not self.numeric_cols:
            raise ValueError(f"No numeric columns found in {data_csv}")

        self.schema = self._infer_schema()
        self.nom_raw, self.nop_raw, self.global_raw, self.nom_mask_node, self.nop_mask_node = self._extract_raw_values()
        self.transform = self._fit_transform_config()
        self.nom_scaled, self.nop_scaled, self.global_scaled = self._transform_values()
        self.X, self.Y, self.M_nom, self.M_nop, self.M_pt = self._make_windows()
        self.train, self.valid, self.test = self._split_windows()
        self.adj, self.graph_edges, self.graph_links = build_adjacency_and_edges(
            graph_csv=graph_csv,
            entity_names=self.schema.entity_names,
            entity_types=self.schema.entity_types,
            nom_cols=self.schema.nom_cols,
            nop_cols=self.schema.nop_cols,
        )
        self.adj = self.adj.to(device)

    def _make_dates(self) -> pd.DatetimeIndex:
        if self.date_col is not None:
            try:
                return _parse_monthly_dates(self.raw_df[self.date_col])
            except Exception as exc:
                raise ValueError(f"Failed to parse date column '{self.date_col}': {exc}") from exc
        return pd.date_range(start="2004-01-01", periods=len(self.raw_df), freq="MS")

    def _infer_schema(self) -> FeatureSchema:
        global_cols = [c for c in self.numeric_cols if is_global_column(c, self.global_regex)]
        candidate = [c for c in self.numeric_cols if c not in global_cols and (is_nom_column(c) or is_nop_column(c))]
        if not candidate:
            raise ValueError(
                "No forecast columns found. Expected columns ending with _NoM and/or _NoP in data/sm_data.csv."
            )

        entity_order: List[str] = []
        nom_by_base: Dict[str, str] = {}
        nop_by_base: Dict[str, str] = {}

        for col in candidate:
            base = strip_signal_suffix(col)
            key = normalize_name(base)
            if key not in {normalize_name(x) for x in entity_order}:
                entity_order.append(base)
            if is_nom_column(col):
                nom_by_base[key] = col
            if is_nop_column(col):
                nop_by_base[key] = col

        if not entity_order:
            raise ValueError("Could not infer any entities from NoM/NoP columns.")

        entity_names: List[str] = []
        entity_types: List[str] = []
        nom_cols: List[Optional[str]] = []
        nop_cols: List[Optional[str]] = []
        rmd_nom_indices: List[int] = []
        pt_nom_indices: List[int] = []
        nop_indices: List[int] = []

        for base in entity_order:
            key = normalize_name(base)
            nom_col = nom_by_base.get(key)
            nop_col = nop_by_base.get(key)
            ent_type = infer_entity_type(base, nom_col, nop_col)
            idx = len(entity_names)
            entity_names.append(base)
            entity_types.append(ent_type)
            nom_cols.append(nom_col)
            nop_cols.append(nop_col)
            if nop_col is not None:
                nop_indices.append(idx)
            if nom_col is not None and ent_type == "RMD":
                rmd_nom_indices.append(idx)
            if nom_col is not None and ent_type == "PT":
                pt_nom_indices.append(idx)

        if not rmd_nom_indices:
            raise ValueError("No RMD *_NoM columns found. At least one RMD NoM target is required.")
        if not pt_nom_indices:
            raise ValueError("No PT *_NoM columns found. At least one PT NoM target is required.")

        input_channels = ["NoM_history", "NoP_history"]
        if self.use_tdb:
            input_channels.append("TDB_history")
        if self.use_bai:
            input_channels.append("BAI_history")
        input_channels += [f"Global::{c}" for c in global_cols]

        return FeatureSchema(
            date_col=self.date_col,
            entity_names=entity_names,
            entity_types=entity_types,
            nom_cols=nom_cols,
            nop_cols=nop_cols,
            global_cols=global_cols,
            input_channels=input_channels,
            target_channels=["TDB"] if self.use_tdb else ["NoM", "NoP"],
            node_count=len(entity_names),
            rmd_nom_indices=rmd_nom_indices,
            pt_nom_indices=pt_nom_indices,
            nop_indices=nop_indices,
        )

    def _extract_raw_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = len(self.raw_df)
        n = self.schema.node_count
        nom = np.zeros((t, n), dtype=np.float32)
        nop = np.zeros((t, n), dtype=np.float32)
        nom_mask = np.zeros((n,), dtype=np.float32)
        nop_mask = np.zeros((n,), dtype=np.float32)

        for i, col in enumerate(self.schema.nom_cols):
            if col is not None:
                nom[:, i] = self.raw_df[col].to_numpy(dtype=np.float32)
                nom_mask[i] = 1.0
        for i, col in enumerate(self.schema.nop_cols):
            if col is not None:
                nop[:, i] = self.raw_df[col].to_numpy(dtype=np.float32)
                nop_mask[i] = 1.0

        if self.clip_nonnegative:
            nom = np.clip(nom, 0.0, None)
            nop = np.clip(nop, 0.0, None)

        if self.schema.global_cols:
            g = self.raw_df[self.schema.global_cols].to_numpy(dtype=np.float32)
            if self.clip_nonnegative:
                g = np.clip(g, 0.0, None)
        else:
            g = np.zeros((t, 0), dtype=np.float32)
        return nom, nop, g, nom_mask, nop_mask

    def _fit_transform_config(self) -> TransformConfig:
        n = len(self.nom_raw)
        min_need = self.seq_in_len + self.seq_out_len + 1
        if self.train_end_date:
            # Scaler fitted only on rows whose date <= train_end_date
            cutoff = pd.Timestamp(self.train_end_date)
            train_end = int(np.searchsorted(self.dates, cutoff, side="right"))
            train_end = max(min_need, min(train_end, n))
        else:
            train_end = max(min_need, int(n * self.train_ratio))
            train_end = min(train_end, n)

        nom = np.log1p(self.nom_raw[:train_end]) if self.log1p else self.nom_raw[:train_end]
        nop = np.log1p(self.nop_raw[:train_end]) if self.log1p else self.nop_raw[:train_end]
        glob = np.log1p(self.global_raw[:train_end]) if self.log1p else self.global_raw[:train_end]

        # Clamp to 1.0 minimum: a near-zero training max (e.g. max_raw=0.01 →
        # log1p≈0.01) would otherwise let a moderate validation value (e.g. 50)
        # inflate the scaled target by 100x, making validation loss explode.
        nom_scale = np.maximum(np.max(np.abs(nom), axis=0), 1.0)
        nop_scale = np.maximum(np.max(np.abs(nop), axis=0), 1.0)
        if glob.shape[1] > 0:
            global_scale = np.maximum(np.max(np.abs(glob), axis=0), 1.0)
        else:
            global_scale = np.zeros((0,), dtype=np.float32)

        return TransformConfig(
            log1p=self.log1p,
            clip_nonnegative=self.clip_nonnegative,
            nom_scale=nom_scale.astype(float).tolist(),
            nop_scale=nop_scale.astype(float).tolist(),
            global_scale=global_scale.astype(float).tolist(),
        )

    def _transform_values(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nom = np.log1p(self.nom_raw) if self.log1p else self.nom_raw.copy()
        nop = np.log1p(self.nop_raw) if self.log1p else self.nop_raw.copy()
        glob = np.log1p(self.global_raw) if self.log1p else self.global_raw.copy()

        nom = nom / np.asarray(self.transform.nom_scale, dtype=np.float32)[None, :]
        nop = nop / np.asarray(self.transform.nop_scale, dtype=np.float32)[None, :]
        if glob.shape[1] > 0:
            glob = glob / np.asarray(self.transform.global_scale, dtype=np.float32)[None, :]
        return nom.astype(np.float32), nop.astype(np.float32), glob.astype(np.float32)

    def _make_windows(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t, n = self.nom_scaled.shape
        samples = t - self.seq_in_len - self.seq_out_len + 1
        if samples <= 0:
            raise ValueError(
                f"Not enough rows ({t}) for seq_in_len={self.seq_in_len} and seq_out_len={self.seq_out_len}."
            )

        c_in = len(self.schema.input_channels)
        c_out = 1 if self.use_tdb else 2
        X = np.zeros((samples, c_in, n, self.seq_in_len), dtype=np.float32)
        Y = np.zeros((samples, self.seq_out_len, c_out, n), dtype=np.float32)

        m_nom = np.zeros((samples, self.seq_out_len, n), dtype=np.float32)
        m_nop = np.zeros((samples, self.seq_out_len, n), dtype=np.float32)
        m_pt = np.zeros((samples, self.seq_out_len, n), dtype=np.float32)

        nom_mask = self.nom_mask_node[None, None, :]
        nop_mask = self.nop_mask_node[None, None, :]
        pt_mask = np.zeros((1, 1, n), dtype=np.float32)
        if self.schema.pt_nom_indices:
            pt_mask[:, :, self.schema.pt_nom_indices] = 1.0

        for s in range(samples):
            a = s
            b = a + self.seq_in_len
            c = b + self.seq_out_len

            X[s, 0] = self.nom_scaled[a:b].T
            X[s, 1] = self.nop_scaled[a:b].T
            ch = 2
            if self.use_tdb:
                X[s, ch] = (self.nom_scaled[a:b] + self.nop_scaled[a:b]).T
                ch += 1
            if self.use_bai:
                bai = self.bai_nom_weight * self.nom_scaled[a:b] + self.bai_nop_weight * self.nop_scaled[a:b]
                X[s, ch] = bai.T
                ch += 1
            for g in range(self.global_scaled.shape[1]):
                X[s, ch] = np.repeat(self.global_scaled[a:b, g][None, :], n, axis=0)
                ch += 1

            if self.use_tdb:
                Y[s, :, 0, :] = self.nom_scaled[b:c] + self.nop_scaled[b:c]
            else:
                Y[s, :, 0, :] = self.nom_scaled[b:c]
                Y[s, :, 1, :] = self.nop_scaled[b:c]

            m_nom[s] = np.repeat(nom_mask, self.seq_out_len, axis=1)
            m_nop[s] = np.repeat(nop_mask, self.seq_out_len, axis=1)
            m_pt[s] = np.repeat(pt_mask, self.seq_out_len, axis=1)

        return (
            torch.from_numpy(X),
            torch.from_numpy(Y),
            torch.from_numpy(m_nom),
            torch.from_numpy(m_nop),
            torch.from_numpy(m_pt),
        )

    def _split_windows(self):
        total = int(self.X.shape[0])

        if self.train_end_date and self.val_end_date:
            # Date-based split: window index s has its last target at
            # dates[s + seq_in_len + seq_out_len - 1].
            # A window belongs to train if its last target <= train_end_date,
            # to val if <= val_end_date, otherwise to test.
            # This eliminates overlap between split target periods.
            train_cutoff = pd.Timestamp(self.train_end_date)
            val_cutoff = pd.Timestamp(self.val_end_date)
            last_target_idx = np.array([
                s + self.seq_in_len + self.seq_out_len - 1 for s in range(total)
            ])
            last_target_dates = self.dates[last_target_idx]
            train_mask = last_target_dates <= train_cutoff
            val_mask = (last_target_dates > train_cutoff) & (last_target_dates <= val_cutoff)
            test_mask = last_target_dates > val_cutoff

            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) == 0:
                raise ValueError(f"Date-based split: no training windows before {self.train_end_date}.")
            if len(val_idx) == 0:
                raise ValueError(f"Date-based split: no validation windows between {self.train_end_date} and {self.val_end_date}.")
            if len(test_idx) == 0:
                raise ValueError(f"Date-based split: no test windows after {self.val_end_date}.")

            # Windows are contiguous and ordered — use slice form for clarity
            train_n = int(train_idx[-1]) + 1
            val_n = len(val_idx)
            test_n = len(test_idx)

            self.split = SplitConfig(
                train_ratio=self.train_ratio,
                valid_ratio=self.valid_ratio,
                seq_in_len=self.seq_in_len,
                seq_out_len=self.seq_out_len,
                train_samples=train_n,
                valid_samples=val_n,
                test_samples=test_n,
                train_end_date=self.train_end_date,
                val_end_date=self.val_end_date,
            )
        else:
            train_n = max(1, int(total * self.train_ratio))
            valid_n = max(1, int(total * self.valid_ratio))
            if train_n + valid_n >= total:
                valid_n = max(1, total - train_n - 1)
            test_n = total - train_n - valid_n
            if test_n <= 0:
                raise ValueError("Chronological split produced no test samples. Reduce train_ratio or valid_ratio.")
            val_n = valid_n

            self.split = SplitConfig(
                train_ratio=self.train_ratio,
                valid_ratio=self.valid_ratio,
                seq_in_len=self.seq_in_len,
                seq_out_len=self.seq_out_len,
                train_samples=train_n,
                valid_samples=val_n,
                test_samples=test_n,
            )

        train = TensorDataset(self.X[:train_n], self.Y[:train_n], self.M_nom[:train_n], self.M_nop[:train_n], self.M_pt[:train_n])
        valid = TensorDataset(
            self.X[train_n:train_n + val_n],
            self.Y[train_n:train_n + val_n],
            self.M_nom[train_n:train_n + val_n],
            self.M_nop[train_n:train_n + val_n],
            self.M_pt[train_n:train_n + val_n],
        )
        test = TensorDataset(
            self.X[train_n + val_n:],
            self.Y[train_n + val_n:],
            self.M_nom[train_n + val_n:],
            self.M_nop[train_n + val_n:],
            self.M_pt[train_n + val_n:],
        )
        return train, valid, test

    def loader(self, which: str, shuffle: bool = False) -> DataLoader:
        ds = {"train": self.train, "valid": self.valid, "test": self.test}[which]
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def inverse_targets_tensor(self, scaled: torch.Tensor) -> torch.Tensor:
        # scaled: [..., horizon, channels, nodes] or [batch, horizon, channels, nodes]
        if self.use_tdb:
            # Single TDB channel: scaled value is nom_scaled + nop_scaled.
            # We return it as-is in the combined scaled space; no per-channel inversion.
            # Multiply by nom_scale as a representative unit (both were log1p-normalized).
            nom_scale = torch.tensor(self.transform.nom_scale, dtype=scaled.dtype, device=scaled.device)
            tdb = scaled[..., 0, :] * nom_scale
            if self.transform.log1p:
                tdb = torch.expm1(tdb)
            if self.transform.clip_nonnegative:
                tdb = torch.clamp(tdb, min=0.0)
            return tdb.unsqueeze(-2)   # [..., 1, nodes]

        nom_scale = torch.tensor(self.transform.nom_scale, dtype=scaled.dtype, device=scaled.device)
        nop_scale = torch.tensor(self.transform.nop_scale, dtype=scaled.dtype, device=scaled.device)
        nom = scaled[..., 0, :] * nom_scale
        nop = scaled[..., 1, :] * nop_scale
        if self.transform.log1p:
            nom = torch.expm1(nom)
            nop = torch.expm1(nop)
        if self.transform.clip_nonnegative:
            nom = torch.clamp(nom, min=0.0)
            nop = torch.clamp(nop, min=0.0)
        return torch.stack([nom, nop], dim=-2)

    def inverse_targets_array(self, scaled: np.ndarray) -> np.ndarray:
        if self.use_tdb:
            nom_scale = np.asarray(self.transform.nom_scale, dtype=np.float32)
            tdb = scaled[..., 0, :] * nom_scale
            if self.transform.log1p:
                tdb = np.expm1(tdb)
            if self.transform.clip_nonnegative:
                tdb = np.clip(tdb, 0.0, None)
            return np.expand_dims(tdb, axis=-2)   # [..., 1, nodes]

        nom_scale = np.asarray(self.transform.nom_scale, dtype=np.float32)
        nop_scale = np.asarray(self.transform.nop_scale, dtype=np.float32)
        nom = scaled[..., 0, :] * nom_scale
        nop = scaled[..., 1, :] * nop_scale
        if self.transform.log1p:
            nom = np.expm1(nom)
            nop = np.expm1(nop)
        if self.transform.clip_nonnegative:
            nom = np.clip(nom, 0.0, None)
            nop = np.clip(nop, 0.0, None)
        return np.stack([nom, nop], axis=-2)

    def last_input_window(self) -> torch.Tensor:
        n = self.schema.node_count
        c = len(self.schema.input_channels)
        x = np.zeros((1, c, n, self.seq_in_len), dtype=np.float32)
        a = len(self.nom_scaled) - self.seq_in_len
        x[0, 0] = self.nom_scaled[a:].T
        x[0, 1] = self.nop_scaled[a:].T
        ch = 2
        if self.use_tdb:
            x[0, ch] = (self.nom_scaled[a:] + self.nop_scaled[a:]).T
            ch += 1
        if self.use_bai:
            bai = self.bai_nom_weight * self.nom_scaled[a:] + self.bai_nop_weight * self.nop_scaled[a:]
            x[0, ch] = bai.T
            ch += 1
        for g in range(self.global_scaled.shape[1]):
            x[0, ch] = np.repeat(self.global_scaled[a:, g][None, :], n, axis=0)
            ch += 1
        return torch.from_numpy(x).to(self.device)

    def future_dates(self) -> pd.DatetimeIndex:
        last = pd.Timestamp(self.dates[-1])
        first_future = last + pd.offsets.MonthBegin(1)
        return pd.date_range(first_future, periods=self.seq_out_len, freq="MS")

    def metadata(self) -> Dict:
        graph_map = {
            self.schema.entity_names[i]: [self.schema.entity_names[j] for j in js]
            for i, js in self.graph_links.items()
        }
        return {
            "data_csv": self.data_csv,
            "graph_csv": self.graph_csv,
            "nodes_csv": self.nodes_csv,
            "schema": asdict(self.schema),
            "transform": asdict(self.transform),
            "split": asdict(self.split),
            "entity_names": self.schema.entity_names,
            "entity_types": self.schema.entity_types,
            "channel_names": self.schema.target_channels,
            "global_variable_names": self.schema.global_cols,
            "forecast_horizon": self.seq_out_len,
            "input_length": self.seq_in_len,
            "dates_start": str(pd.Timestamp(self.dates[0]).date()),
            "dates_end": str(pd.Timestamp(self.dates[-1]).date()),
            "graph_mapping": graph_map,
            "bai_config": {
                "use_bai": self.use_bai,
                "bai_nom_weight": self.bai_nom_weight,
                "bai_nop_weight": self.bai_nop_weight,
            },
            "tdb_config": {
                "use_tdb": self.use_tdb,
                "formula": "NoM_scaled + NoP_scaled",
            },
        }


def infer_entity_type(base: str, nom_col: Optional[str], nop_col: Optional[str]) -> str:
    candidates = [base, nom_col or "", nop_col or ""]
    combined = " ".join(candidates).lower().strip()
    if combined.startswith("rmd_") or " rmd_" in combined:
        return "RMD"
    if combined.startswith("pt_") or " pt_" in combined:
        return "PT"
    if re.search(r"\b(therapy|biomarker|device|stimulation|treatment|medicine|drug|digital|ai)\b", combined):
        return "PT"
    return "RMD"


def _entity_lookup(entity_names: List[str], nom_cols: List[Optional[str]], nop_cols: List[Optional[str]]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for i, name in enumerate(entity_names):
        aliases = {name, strip_signal_suffix(name)}
        if nom_cols[i] is not None:
            aliases.add(nom_cols[i])
            aliases.add(strip_signal_suffix(nom_cols[i]))
        if nop_cols[i] is not None:
            aliases.add(nop_cols[i])
            aliases.add(strip_signal_suffix(nop_cols[i]))
        for a in aliases:
            lookup[normalize_name(a)] = i
    return lookup


def build_adjacency_and_edges(
    graph_csv: str,
    entity_names: List[str],
    entity_types: List[str],
    nom_cols: List[Optional[str]],
    nop_cols: List[Optional[str]],
) -> Tuple[torch.Tensor, List[Tuple[str, str, int, int]], Dict[int, List[int]]]:
    n = len(entity_names)
    adj = np.zeros((n, n), dtype=np.float32)
    edges: List[Tuple[str, str, int, int]] = []
    links: Dict[int, List[int]] = {}

    if not graph_csv or not Path(graph_csv).exists():
        # Identity-only prior if graph is missing.
        np.fill_diagonal(adj, 1.0)
        return torch.from_numpy(_row_normalize(adj)), edges, links

    lookup = _entity_lookup(entity_names, nom_cols, nop_cols)

    with open(graph_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            tokens = [str(x).strip() for x in row if str(x).strip()]
            if len(tokens) < 2:
                continue
            src = lookup.get(normalize_name(tokens[0]))
            if src is None:
                continue
            for dst_name in tokens[1:]:
                dst = lookup.get(normalize_name(dst_name))
                if dst is None or dst == src:
                    continue
                adj[src, dst] = 1.0
                adj[dst, src] = 1.0

    # Build oriented RMD->PT links used by gap analysis.
    for i in range(n):
        for j in range(n):
            if i == j or adj[i, j] <= 0:
                continue
            ti = entity_types[i]
            tj = entity_types[j]
            if ti == "RMD" and tj == "PT":
                links.setdefault(i, [])
                if j not in links[i]:
                    links[i].append(j)
                edges.append((entity_names[i], entity_names[j], i, j))

    np.fill_diagonal(adj, 1.0)
    adj = _row_normalize(adj)
    return torch.from_numpy(adj.astype(np.float32)), edges, links


def _row_normalize(a: np.ndarray) -> np.ndarray:
    d = a.sum(axis=1, keepdims=True)
    d[d <= 0] = 1.0
    return a / d


def reshape_model_output(pred: torch.Tensor, horizon: int, out_channels: int, node_count: int) -> torch.Tensor:
    if pred.dim() != 3:
        raise ValueError(f"Expected model output with 3 dims [B, H*C, N], got shape {tuple(pred.shape)}")
    b, hc, n = pred.shape
    if n != node_count:
        raise ValueError(f"Model output node count mismatch: got {n}, expected {node_count}")
    expect = horizon * out_channels
    if hc != expect:
        raise ValueError(f"Model output time/channel mismatch: got {hc}, expected {expect} (= {horizon}*{out_channels})")
    return pred.view(b, out_channels, horizon, n).permute(0, 2, 1, 3).contiguous()


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    den = torch.sum(mask).clamp_min(1.0)
    return torch.sum(x * mask) / den


def multi_target_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask_nom: torch.Tensor,
    mask_nop: torch.Tensor,
    mask_pt: torch.Tensor,
    loss_name: str,
    w_nom: float,
    w_nop: float,
    w_pt: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if loss_name == "mae":
        base = torch.abs(pred - true)
    elif loss_name == "mse":
        base = (pred - true) ** 2
    else:
        base = torch.nn.functional.smooth_l1_loss(pred, true, reduction="none", beta=0.10)

    tdb_mode = (pred.shape[2] == 1)  # single combined channel

    if tdb_mode:
        tdb_err = base[:, :, 0, :]
        # Use nom mask as combined mask (all nodes that have NoM have the combined target)
        loss_tdb = masked_mean(tdb_err, mask_nom)
        total = loss_tdb
        stats = {
            "loss_TDB": float(loss_tdb.detach().item()),
            "loss_total": float(total.detach().item()),
        }
        return total, stats

    nom_err = base[:, :, 0, :]
    nop_err = base[:, :, 1, :]

    loss_nom = masked_mean(nom_err, mask_nom)
    loss_nop = masked_mean(nop_err, mask_nop)
    loss_pt = masked_mean(nom_err, mask_pt)

    total = w_nom * loss_nom + w_nop * loss_nop + w_pt * loss_pt
    stats = {
        "loss_NoM": float(loss_nom.detach().item()),
        "loss_NoP": float(loss_nop.detach().item()),
        "loss_PT": float(loss_pt.detach().item()),
        "loss_total": float(total.detach().item()),
    }
    return total, stats


def predict_mc(
    model: torch.nn.Module,
    X: torch.Tensor,
    data: RMDPTData,
    mc_runs: int,
    device: torch.device,
    horizon: int,
    out_channels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.train()
    scaled_samples = []
    with torch.no_grad():
        for _ in range(int(mc_runs)):
            out = model(X.to(device))
            out = reshape_model_output(out, horizon=horizon, out_channels=out_channels, node_count=data.schema.node_count)
            scaled_samples.append(out.cpu().numpy())

    arr_scaled = np.stack(scaled_samples, axis=0)
    arr_raw = data.inverse_targets_array(arr_scaled)
    mean_raw = arr_raw.mean(axis=0)
    std_raw = arr_raw.std(axis=0)
    mean_scaled = arr_scaled.mean(axis=0)
    std_scaled = arr_scaled.std(axis=0)
    return mean_raw, std_raw, arr_raw, mean_scaled, std_scaled


def predict_loader_mc(
    model: torch.nn.Module,
    loader: DataLoader,
    data: RMDPTData,
    mc_runs: int,
    device: torch.device,
    horizon: int,
    out_channels: int,
) -> Dict[str, np.ndarray]:
    means_raw, stds_raw, trues_raw = [], [], []
    means_scaled, trues_scaled = [], []
    masks_nom, masks_nop, masks_pt = [], [], []

    for X, Y, M_nom, M_nop, M_pt in loader:
        mean_raw, std_raw, _, mean_scaled, _ = predict_mc(
            model=model,
            X=X,
            data=data,
            mc_runs=mc_runs,
            device=device,
            horizon=horizon,
            out_channels=out_channels,
        )
        y_raw = data.inverse_targets_tensor(Y.to(device)).cpu().numpy()

        means_raw.append(mean_raw)
        stds_raw.append(std_raw)
        trues_raw.append(y_raw)
        means_scaled.append(mean_scaled)
        trues_scaled.append(Y.numpy())
        masks_nom.append(M_nom.numpy())
        masks_nop.append(M_nop.numpy())
        masks_pt.append(M_pt.numpy())

    return {
        "mean_raw": np.concatenate(means_raw, axis=0),
        "std_raw": np.concatenate(stds_raw, axis=0),
        "true_raw": np.concatenate(trues_raw, axis=0),
        "mean_scaled": np.concatenate(means_scaled, axis=0),
        "true_scaled": np.concatenate(trues_scaled, axis=0),
        "mask_nom": np.concatenate(masks_nom, axis=0),
        "mask_nop": np.concatenate(masks_nop, axis=0),
        "mask_pt": np.concatenate(masks_pt, axis=0),
    }


def predict_loader_point(
    model: torch.nn.Module,
    loader: DataLoader,
    data: "RMDPTData",
    device: torch.device,
    horizon: int,
    out_channels: int,
) -> Dict[str, np.ndarray]:
    """Deterministic eval() prediction — no dropout noise, for clean point metrics."""
    model.eval()
    means_raw, trues_raw = [], []
    masks_nom, masks_nop, masks_pt = [], [], []

    with torch.no_grad():
        for X, Y, M_nom, M_nop, M_pt in loader:
            raw = model(X.to(device))
            pred = reshape_model_output(raw, horizon=horizon, out_channels=out_channels,
                                        node_count=data.schema.node_count)
            mean_raw = data.inverse_targets_tensor(pred).cpu().numpy()
            y_raw = data.inverse_targets_tensor(Y.to(device)).cpu().numpy()
            means_raw.append(mean_raw)
            trues_raw.append(y_raw)
            masks_nom.append(M_nom.numpy())
            masks_nop.append(M_nop.numpy())
            masks_pt.append(M_pt.numpy())

    model.train()
    return {
        "mean_raw": np.concatenate(means_raw, axis=0),
        "true_raw": np.concatenate(trues_raw, axis=0),
        "mask_nom": np.concatenate(masks_nom, axis=0),
        "mask_nop": np.concatenate(masks_nop, axis=0),
        "mask_pt": np.concatenate(masks_pt, axis=0),
    }


def _masked_flat(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = values.reshape(-1)
    m = mask.reshape(-1) > 0.5
    if m.sum() == 0:
        return np.array([], dtype=np.float32)
    return v[m]


def conformal_q(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, alpha: float = 0.05) -> float:
    err = np.abs(y_true - y_pred)
    vals = _masked_flat(err, mask)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    q_level = min(1.0, math.ceil((vals.size + 1) * (1.0 - alpha)) / vals.size)
    try:
        return float(np.quantile(vals, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(vals, q_level, interpolation="higher"))


def interval_from_mc_group(
    mean_raw: np.ndarray,
    std_raw: np.ndarray,
    schema: FeatureSchema,
    q_nom: float,
    q_nop: float,
    q_pt: float,
    z: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.copy(mean_raw)
    upper = np.copy(mean_raw)

    tdb_mode = (mean_raw.shape[2] == 1)

    # Channel 0 (NoM or TDB): RMD nodes use q_nom, PT nodes use q_pt.
    nom_q = np.full((schema.node_count,), q_nom, dtype=np.float32)
    for i in schema.pt_nom_indices:
        nom_q[i] = q_pt

    half_nom = z * std_raw[:, :, 0, :] + nom_q[None, None, :]
    lower[:, :, 0, :] = np.clip(mean_raw[:, :, 0, :] - half_nom, 0.0, None)
    upper[:, :, 0, :] = mean_raw[:, :, 0, :] + half_nom

    if not tdb_mode:
        half_nop = z * std_raw[:, :, 1, :] + q_nop
        lower[:, :, 1, :] = np.clip(mean_raw[:, :, 1, :] - half_nop, 0.0, None)
        upper[:, :, 1, :] = mean_raw[:, :, 1, :] + half_nop

    return lower, upper


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
    """
    Safe forecasting metrics for sparse / low-volume count time series.

    Primary metrics:
      - RAE
      - RSE
      - Coverage

    Diagnostic:
      - Corr
    """

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()

    # Keep only fully finite rows
    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
        & np.isfinite(lower)
        & np.isfinite(upper)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    lower = lower[mask]
    upper = upper[mask]

    if y_true.size == 0:
        return {
            "RAE": float("nan"),
            "RSE": float("nan"),
            "Corr": float("nan"),
            "Coverage": float("nan"),
        }

    err = y_pred - y_true

    eps = 1e-12

    # RAE: compare absolute error to mean-baseline absolute error
    denom_rae = np.sum(np.abs(y_true - np.mean(y_true)))
    if denom_rae <= eps:
        rae = 0.0 if np.sum(np.abs(err)) <= eps else float("nan")
    else:
        rae = float(np.sum(np.abs(err)) / denom_rae)

    # RSE: compare squared error to mean-baseline squared error
    denom_rse = np.sqrt(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom_rse <= eps:
        rse = 0.0 if np.sqrt(np.sum(err ** 2)) <= eps else float("nan")
    else:
        rse = float(np.sqrt(np.sum(err ** 2)) / denom_rse)

    # Correlation is useful, but not a primary accuracy metric
    if y_true.size > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")

    # Prediction interval metrics
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))

    return {
        "RAE": rae,
        "RSE": rse,
        "Corr": corr,
        "Coverage": coverage,
    }


def evaluate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    t = _masked_flat(y_true, mask)
    p = _masked_flat(y_pred, mask)
    l = _masked_flat(y_lower, mask)
    u = _masked_flat(y_upper, mask)
    return _safe_metrics(t, p, l, u)


def evaluate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    mask_nom: np.ndarray,
    mask_nop: np.ndarray,
    mask_pt: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    # y arrays: [B, H, C, N]
    m_nom = mask_nom
    m_nop = mask_nop
    m_pt = mask_pt

    tdb_mode = (y_true.shape[2] == 1)
    nop_ch = 0 if tdb_mode else 1

    group_nom = evaluate_group_metrics(y_true[:, :, 0, :], y_pred[:, :, 0, :], y_lower[:, :, 0, :], y_upper[:, :, 0, :], m_nom)
    group_nop = evaluate_group_metrics(y_true[:, :, nop_ch, :], y_pred[:, :, nop_ch, :], y_lower[:, :, nop_ch, :], y_upper[:, :, nop_ch, :], m_nop)
    group_pt = evaluate_group_metrics(y_true[:, :, 0, :], y_pred[:, :, 0, :], y_lower[:, :, 0, :], y_upper[:, :, 0, :], m_pt)

    stacked_true = np.concatenate(
        [
            _masked_flat(y_true[:, :, 0, :], m_nom),
            _masked_flat(y_true[:, :, nop_ch, :], m_nop),
            _masked_flat(y_true[:, :, 0, :], m_pt),
        ]
    )
    stacked_pred = np.concatenate(
        [
            _masked_flat(y_pred[:, :, 0, :], m_nom),
            _masked_flat(y_pred[:, :, nop_ch, :], m_nop),
            _masked_flat(y_pred[:, :, 0, :], m_pt),
        ]
    )
    stacked_low = np.concatenate(
        [
            _masked_flat(y_lower[:, :, 0, :], m_nom),
            _masked_flat(y_lower[:, :, nop_ch, :], m_nop),
            _masked_flat(y_lower[:, :, 0, :], m_pt),
        ]
    )
    stacked_up = np.concatenate(
        [
            _masked_flat(y_upper[:, :, 0, :], m_nom),
            _masked_flat(y_upper[:, :, nop_ch, :], m_nop),
            _masked_flat(y_upper[:, :, 0, :], m_pt),
        ]
    )
    overall = _safe_metrics(stacked_true, stacked_pred, stacked_low, stacked_up)

    group_label = "TDB" if tdb_mode else "NoP"
    return {"Overall": overall, "NoM": group_nom, group_label: group_nop, "PT_NoM": group_pt}


def nodewise_metrics(
    schema: FeatureSchema,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    mask_nom: np.ndarray,
    mask_nop: np.ndarray,
) -> pd.DataFrame:
    tdb_mode = (y_true.shape[2] == 1)
    nop_ch = 0 if tdb_mode else 1
    rows = []
    for i, name in enumerate(schema.entity_names):
        if schema.nom_cols[i] is not None:
            m = mask_nom[:, :, i]
            vals = evaluate_group_metrics(
                y_true[:, :, 0, i],
                y_pred[:, :, 0, i],
                y_lower[:, :, 0, i],
                y_upper[:, :, 0, i],
                m,
            )
            ch_label = "TDB" if tdb_mode else "NoM"
            rows.append({"entity": name, "channel": ch_label, "entity_type": schema.entity_types[i], **vals})
        if schema.nop_cols[i] is not None and not tdb_mode:
            m = mask_nop[:, :, i]
            vals = evaluate_group_metrics(
                y_true[:, :, nop_ch, i],
                y_pred[:, :, nop_ch, i],
                y_lower[:, :, nop_ch, i],
                y_upper[:, :, nop_ch, i],
                m,
            )
            rows.append({"entity": name, "channel": "NoP", "entity_type": schema.entity_types[i], **vals})
    return pd.DataFrame(rows)


def save_json(obj: Dict, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metrics_table(path: str | Path, split_name: str, metrics_by_group: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for group, vals in metrics_by_group.items():
        row = {"split": split_name, "group": group}
        row.update(vals)
        rows.append(row)
    ensure_dir(Path(path).parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_node_metrics_csv(path: str | Path, df: pd.DataFrame) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def _col_label(schema: FeatureSchema, idx: int, channel: str) -> str:
    if channel == "NoM" and schema.nom_cols[idx] is not None:
        return schema.nom_cols[idx]
    if channel == "NoP" and schema.nop_cols[idx] is not None:
        return schema.nop_cols[idx]
    return f"{schema.entity_names[idx]}_{channel}"


def save_group_forecast_csv(
    path: str | Path,
    dates: Iterable,
    values: np.ndarray,
    schema: FeatureSchema,
    channel: str,
    indices: List[int],
) -> None:
    ensure_dir(Path(path).parent)
    data = {"Date": pd.to_datetime(list(dates)).strftime("%Y-%m-%d")}
    n_channels = values.shape[1]
    ch = 0 if (channel == "NoM" or n_channels == 1) else 1
    for i in indices:
        data[_col_label(schema, i, channel)] = values[:, ch, i]
    pd.DataFrame(data).to_csv(path, index=False)


def save_required_forecast_files(
    output_data_dir: str | Path,
    dates: Iterable,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    std: np.ndarray,
    schema: FeatureSchema,
) -> None:
    out = Path(output_data_dir)
    ensure_dir(out)

    nom_idx = list(schema.rmd_nom_indices)
    nop_idx = list(schema.nop_indices)
    pt_idx = list(schema.pt_nom_indices)

    save_group_forecast_csv(out / "forecast_NoM_mean.csv", dates, mean, schema, "NoM", nom_idx)
    save_group_forecast_csv(out / "forecast_NoM_lower95.csv", dates, lower, schema, "NoM", nom_idx)
    save_group_forecast_csv(out / "forecast_NoM_upper95.csv", dates, upper, schema, "NoM", nom_idx)
    save_group_forecast_csv(out / "forecast_NoM_mc_std.csv", dates, std, schema, "NoM", nom_idx)

    save_group_forecast_csv(out / "forecast_NoP_mean.csv", dates, mean, schema, "NoP", nop_idx)
    save_group_forecast_csv(out / "forecast_NoP_lower95.csv", dates, lower, schema, "NoP", nop_idx)
    save_group_forecast_csv(out / "forecast_NoP_upper95.csv", dates, upper, schema, "NoP", nop_idx)
    save_group_forecast_csv(out / "forecast_NoP_mc_std.csv", dates, std, schema, "NoP", nop_idx)

    save_group_forecast_csv(out / "forecast_PT_NoM_mean.csv", dates, mean, schema, "NoM", pt_idx)
    save_group_forecast_csv(out / "forecast_PT_NoM_lower95.csv", dates, lower, schema, "NoM", pt_idx)
    save_group_forecast_csv(out / "forecast_PT_NoM_upper95.csv", dates, upper, schema, "NoM", pt_idx)
    save_group_forecast_csv(out / "forecast_PT_NoM_mc_std.csv", dates, std, schema, "NoM", pt_idx)


def _normalize_to_100(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=np.float64)
    if s.size == 0:
        return s
    base = float(s[0])
    if abs(base) < 1e-8:
        return (s - base) + 100.0
    return 100.0 * s / base


def build_gap_forecast(
    dates: Iterable,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    schema: FeatureSchema,
    graph_links: Dict[int, List[int]],
) -> pd.DataFrame:
    rows = []
    date_idx = pd.to_datetime(list(dates))

    for rmd_i in schema.rmd_nom_indices:
        linked_pts = graph_links.get(rmd_i, [])
        if not linked_pts:
            continue

        tdb_mode = (mean.shape[1] == 1)
        has_nop = schema.nop_cols[rmd_i] is not None and not tdb_mode

        # Channel 0 is TDB (NoM+NoP combined) in TDB mode, or NoM in 2-channel mode
        rmd_ch0    = mean[:, 0, rmd_i]
        rmd_ch0_lo = lower[:, 0, rmd_i]
        rmd_ch0_hi = upper[:, 0, rmd_i]

        pt_mean = mean[:, 0, linked_pts].mean(axis=1)
        pt_lo   = lower[:, 0, linked_pts].mean(axis=1)
        pt_hi   = upper[:, 0, linked_pts].mean(axis=1)

        ch0_idx    = _normalize_to_100(rmd_ch0)
        ch0_lo_idx = _normalize_to_100(rmd_ch0_lo)
        ch0_hi_idx = _normalize_to_100(rmd_ch0_hi)
        pt_idx     = _normalize_to_100(pt_mean)
        pt_lo_idx  = _normalize_to_100(pt_lo)
        pt_hi_idx  = _normalize_to_100(pt_hi)

        if has_nop:
            rmd_nop    = mean[:, 1, rmd_i]
            rmd_nop_lo = lower[:, 1, rmd_i]
            rmd_nop_hi = upper[:, 1, rmd_i]
            nop_idx    = _normalize_to_100(rmd_nop)
            nop_lo_idx = _normalize_to_100(rmd_nop_lo)
            nop_hi_idx = _normalize_to_100(rmd_nop_hi)

        gap_ch0 = ch0_idx - pt_idx

        # Column label reflects what channel 0 actually represents
        ch0_label = "TDB" if tdb_mode else "NoM"

        for t, d in enumerate(date_idx):
            row = {
                "Date": d.strftime("%Y-%m-%d"),
                "RMD": schema.entity_names[rmd_i],
                "Linked_PT_Count": len(linked_pts),
                "Linked_PT_List": "|".join(schema.entity_names[j] for j in linked_pts),
                f"RMD_{ch0_label}_Index": float(ch0_idx[t]),
                f"RMD_{ch0_label}_Lower95_Index": float(ch0_lo_idx[t]),
                f"RMD_{ch0_label}_Upper95_Index": float(ch0_hi_idx[t]),
                "PT_NoM_Response_Index": float(pt_idx[t]),
                "PT_NoM_Response_Lower95_Index": float(pt_lo_idx[t]),
                "PT_NoM_Response_Upper95_Index": float(pt_hi_idx[t]),
                f"Gap_{ch0_label}_minus_PT": float(gap_ch0[t]),
            }
            if has_nop:
                row["RMD_NoP_Index"]          = float(nop_idx[t])
                row["RMD_NoP_Lower95_Index"]  = float(nop_lo_idx[t])
                row["RMD_NoP_Upper95_Index"]  = float(nop_hi_idx[t])
                row["Gap_NoP_minus_PT"]        = float(nop_idx[t] - pt_idx[t])
            rows.append(row)

    return pd.DataFrame(rows)


def save_gap_forecast(path: str | Path, gap_df: pd.DataFrame) -> None:
    ensure_dir(Path(path).parent)
    gap_df.to_csv(path, index=False)
