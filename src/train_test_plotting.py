import os
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt

# Global high-res savefig defaults
mpl.rcParams["savefig.dpi"] = 1200
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.05


def _offset_year_month(base_year: int, base_month: int, offset: int, steps_per_year: int) -> tuple[int, int]:
    """Shift (base_year, base_month) forward by offset steps."""
    steps_per_year = max(1, int(steps_per_year))
    total = (int(base_month) - 1) + int(offset)
    year = int(base_year) + (total // steps_per_year)
    month = (total % steps_per_year) + 1
    return year, month


def _get_plot_out_dir(kind: str) -> str:
    """Get the output directory for plots based on kind (Validation/Testing)."""
    kind_clean = kind.split("_")[0]
    return str(Path("model") / kind_clean)


def plot_predicted_actual(
    pred,
    true,
    title,
    kind,
    ci=None,
    base_year=None,
    steps_per_year=None,
    base_month=1,
):
    """Generate and save a plot of predicted vs. actual values."""
    try:
        pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
        true_np = true.detach().cpu().numpy() if isinstance(true, torch.Tensor) else np.asarray(true)
        ci_np = None
        if ci is not None:
            try:
                ci_np = ci.detach().cpu().numpy() if isinstance(ci, torch.Tensor) else np.asarray(ci)
            except Exception:
                ci_np = None

        pred_np = pred_np.ravel()
        true_np = true_np.ravel()
        if ci_np is not None:
            ci_np = ci_np.ravel()

        out_dir = _get_plot_out_dir(kind)
        os.makedirs(out_dir, exist_ok=True)

        x = np.arange(1, len(pred_np) + 1)

        # Blue solid line for Actual
        plt.plot(x, true_np, "b-", label="Actual")
        # Purple dashed line for Predicted
        plt.plot(x, pred_np, "--", color="purple", label="Predicted")

        # Pink confidence band
        if ci_np is not None and len(ci_np) == len(pred_np):
            plt.fill_between(
                x,
                pred_np - ci_np,
                pred_np + ci_np,
                alpha=0.5,
                color="pink",
                label="95% Confidence",
            )

        plt.legend(loc="best", prop={"size": 11})
        plt.grid(True)
        display_title = title.replace("_", " ")
        plt.title(display_title, y=1.03, fontsize=18)
        plt.ylabel("Trend", fontsize=15)
        plt.xlabel("Month", fontsize=15)

        plt.autoscale(enable=True, axis="y", tight=False)
        ax = plt.gca()
        ax.autoscale_view()

        # Dynamic quarterly month labels (MM-YYYY format)
        if base_year is not None and steps_per_year is not None and steps_per_year > 0:
            try:
                by = int(base_year)
                bm = int(base_month) if base_month is not None else 1

                tick_pos = []
                tick_labels = []

                # For monthly data: semiannual markers
                for i in range(len(pred_np)):
                    total_months_from_base = i + (bm - 1)
                    year_num = by + (total_months_from_base // 12)
                    month_num = (total_months_from_base % 12) + 1

                    if month_num in [6, 12]:
                        tick_pos.append(i + 1)
                        tick_labels.append(f"{month_num:02d}-{year_num}")

                if tick_pos:
                    plt.xticks(ticks=tick_pos, labels=tick_labels, rotation="vertical", fontsize=13)
            except Exception as e:
                print(f"[plot_xticks] Error formatting x-ticks: {e}")

        plt.yticks(fontsize=13)

        title_fs = title.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"{title_fs}_{kind}.png"), bbox_inches="tight")
        plt.savefig(
            os.path.join(out_dir, f"{title_fs}_{kind}.pdf"),
            bbox_inches="tight",
            format="pdf",
        )

        plt.close()
    except Exception as e:
        print(f"[plot_predicted_actual] error: {e}")
