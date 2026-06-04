#!/usr/bin/env python3
"""Monitor active MLflow training run: progress, metrics, ETA.

Usage:
    python scripts/training_status.py              # one-shot
    python scripts/training_status.py --watch 10   # refresh every 10s
    python scripts/training_status.py --run <id>   # pin a specific run ID
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"

# Sparkline blocks (lowest → highest)
_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 12) -> str:
    if not values:
        return ""
    sample = values[-width:]
    lo, hi = min(sample), max(sample)
    span = hi - lo or 1e-9
    return "".join(_SPARK[min(int((v - lo) / span * (len(_SPARK) - 1)), len(_SPARK) - 1)] for v in sample)


def _progress_bar(current: int, total: int, width: int = 36) -> str:
    frac = min(current / total, 1.0) if total > 0 else 0.0
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({frac * 100:.1f}%)"


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "—"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _metric_history_values(client, run_id: str, key: str) -> list[float]:
    try:
        history = client.get_metric_history(run_id, key)
        return [m.value for m in sorted(history, key=lambda m: m.step)]
    except Exception:
        return []


def _metric_history_steps(client, run_id: str, key: str) -> list[int]:
    try:
        history = client.get_metric_history(run_id, key)
        return [m.step for m in sorted(history, key=lambda m: m.step)]
    except Exception:
        return []


def find_run(client, run_id: str | None = None):
    if run_id:
        return client.get_run(run_id)

    experiments = client.search_experiments()
    exp_ids = [e.experiment_id for e in experiments]
    if not exp_ids:
        return None

    # Most recently started run across all statuses
    runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=["attribute.start_time DESC"],
        max_results=10,
    )
    if not runs:
        return None

    now_ms = time.time() * 1000
    for run in runs:
        # Skip zombie: RUNNING with no metrics logged and started > 2 hours ago
        if (
            run.info.status == "RUNNING"
            and not run.data.metrics
            and (now_ms - run.info.start_time) > 2 * 3600 * 1000
        ):
            continue
        return run

    return runs[0]  # fallback: return most recent even if zombie


def render_status(client, run) -> str:
    lines: list[str] = []
    add = lines.append

    info = run.info
    params = run.data.params
    metrics = run.data.metrics  # latest value per key

    elapsed_s = (time.time() * 1000 - info.start_time) / 1000.0

    # Look up experiment name
    try:
        exp = client.get_experiment(info.experiment_id)
        exp_name = exp.name
    except Exception:
        exp_name = info.experiment_id

    status_icon = "🟢" if info.status == "RUNNING" else ("✅" if info.status == "FINISHED" else "❌")

    add("")
    add("=" * 62)
    add("  MLflow Training Status")
    add("=" * 62)
    add(f"  Experiment : {exp_name}")
    add(f"  Run ID     : {info.run_id}")
    add(f"  Status     : {status_icon}  {info.status}")
    add(f"  Elapsed    : {_fmt_duration(elapsed_s)}")

    # ── Detect training mode ────────────────────────────────────────
    has_max_steps = "max_steps" in params
    has_max_epochs = "max_epochs" in params
    task = params.get("task", "")

    if has_max_steps and not has_max_epochs:
        # ── PRE-TRAINING (step-based) ───────────────────────────────
        max_steps = int(params["max_steps"])
        loss_steps = _metric_history_steps(client, info.run_id, "train/loss")
        current_step = loss_steps[-1] if loss_steps else 0

        add("")
        add("  ── Progress ────────────────────────────────────────────")
        add(f"  {_progress_bar(current_step, max_steps)}")

        # ETA
        sps = metrics.get("train/steps_per_second")
        if sps and sps > 0 and current_step < max_steps:
            eta_s = (max_steps - current_step) / sps
            add(f"  ETA        : {_fmt_duration(eta_s)}")

        # Metrics
        add("")
        add("  ── Training Metrics ────────────────────────────────────")
        for key in ("train/loss", "train/mlm_loss", "train/learning_rate", "train/steps_per_second"):
            if key in metrics:
                val = metrics[key]
                fmt = f"{val:.2e}" if key == "train/learning_rate" else f"{val:.6f}"
                add(f"  {key:<32} {fmt}")

        # Loss sparkline
        loss_vals = _metric_history_values(client, info.run_id, "train/loss")
        if loss_vals:
            add(f"  Loss trend (last {min(12, len(loss_vals))})   {_sparkline(loss_vals)}")

        # Eval metrics
        eval_keys = [k for k in ("eval/mlm_loss", "eval/perplexity") if k in metrics]
        if eval_keys:
            add("")
            add("  ── Eval Metrics ────────────────────────────────────────")
            for key in eval_keys:
                add(f"  {key:<32} {metrics[key]:.6f}")

    elif has_max_epochs:
        # ── FINE-TUNING (epoch-based) ───────────────────────────────
        max_epochs = int(params["max_epochs"])
        loss_steps = _metric_history_steps(client, info.run_id, "train_loss")
        # steps are 0-indexed epochs; completed epoch = last step + 1
        current_epoch = (loss_steps[-1] + 1) if loss_steps else 0

        add("")
        add("  ── Progress ────────────────────────────────────────────")
        add(f"  {_progress_bar(current_epoch, max_epochs)}")

        # ETA based on wall-clock time per epoch
        if current_epoch > 0 and info.status == "RUNNING":
            secs_per_epoch = elapsed_s / current_epoch
            remaining = max_epochs - current_epoch
            add(f"  ETA        : {_fmt_duration(secs_per_epoch * remaining)}  ({_fmt_duration(secs_per_epoch)}/epoch)")

        # Task-specific metrics
        add("")
        add(f"  ── Fine-tuning Metrics  [task: {task or 'unknown'}] ─────────────")

        # Haplogroup
        for key in ("train_loss", "val_accuracy", "best_val_accuracy"):
            if key in metrics:
                add(f"  {key:<32} {metrics[key]:.6f}")

        # Pathogenicity
        for key in ("val_auroc", "best_val_auroc"):
            if key in metrics:
                add(f"  {key:<32} {metrics[key]:.6f}")

        # Heteroplasmy (fold metrics)
        fold_keys = sorted(k for k in metrics if k.startswith("fold_"))
        for key in fold_keys:
            add(f"  {key:<32} {metrics[key]:.6f}")
        for key in ("mean_r2", "mean_spearman"):
            if key in metrics:
                add(f"  {key:<32} {metrics[key]:.6f}")

        # Loss sparkline
        loss_vals = _metric_history_values(client, info.run_id, "train_loss")
        if loss_vals:
            add(f"  Loss trend (last {min(12, len(loss_vals))})   {_sparkline(loss_vals)}")

    else:
        # Unknown / early run — just dump all metrics
        add("")
        add("  ── Metrics (raw) ───────────────────────────────────────")
        for key, val in sorted(metrics.items()):
            add(f"  {key:<32} {val:.6f}")

    add("=" * 62)
    add("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MLflow training status")
    parser.add_argument("--watch", metavar="N", type=float, default=0,
                        help="Refresh every N seconds (default: one-shot)")
    parser.add_argument("--run", metavar="RUN_ID", default=None,
                        help="Pin a specific MLflow run ID")
    args = parser.parse_args()

    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        sys.exit("mlflow is not installed. Run: uv pip install mlflow")

    client = MlflowClient(tracking_uri=TRACKING_URI)

    def once():
        run = find_run(client, args.run)
        if run is None:
            print("No MLflow runs found in", TRACKING_URI)
            return
        print(render_status(client, run))

    if args.watch > 0:
        try:
            while True:
                # Clear terminal
                os.system("clear" if os.name == "posix" else "cls")
                once()
                print(f"  Refreshing every {args.watch}s — Ctrl-C to stop")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        once()


if __name__ == "__main__":
    main()
