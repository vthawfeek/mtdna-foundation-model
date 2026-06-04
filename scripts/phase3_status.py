"""
Check Phase 3 (haplogroup fine-tuning) training progress.

Usage:
    uv run python scripts/phase3_status.py

Reads from mlflow.db — safe to run while training is in progress.
"""

import time
from datetime import datetime, timedelta

import mlflow

EXPERIMENT_ID = "3"  # mtdna_fm_haplogroup
MAX_EPOCHS = 5
_FALLBACK_TRAIN_H = 6.25  # fallback before any epoch completes (from epoch 2 measurement)
_FALLBACK_VAL_H = 1.06    # fallback before any epoch completes


def measure_epoch_timings(loss_hist, val_by_step, start_ms):
    """Return (avg_train_h, avg_val_h) measured from completed epochs in MLflow history."""
    train_durations, val_durations = [], []
    prev_val_end_ms = None
    for lh in sorted(loss_hist, key=lambda x: x.step):
        vh = val_by_step.get(lh.step)
        if vh is None:
            continue
        if prev_val_end_ms is not None:
            train_durations.append((lh.timestamp - prev_val_end_ms) / 3_600_000)
        val_durations.append((vh.timestamp - lh.timestamp) / 3_600_000)
        prev_val_end_ms = vh.timestamp
    avg_train = sum(train_durations) / len(train_durations) if train_durations else _FALLBACK_TRAIN_H
    avg_val = sum(val_durations) / len(val_durations) if val_durations else _FALLBACK_VAL_H
    return avg_train, avg_val


def fmt_h(hours: float) -> str:
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}h {m:02d}m"


def main() -> None:
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(EXPERIMENT_ID, order_by=["start_time DESC"], max_results=1)
    if not runs:
        print("No runs found in experiment mtdna_fm_haplogroup.")
        return

    r = runs[0]
    run_id = r.info.run_id
    start_ms = r.info.start_time
    status = r.info.status
    now_ms = time.time() * 1000
    elapsed_h = (now_ms - start_ms) / 3_600_000
    started_str = datetime.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d %H:%M")

    loss_hist = client.get_metric_history(run_id, "train_loss")
    val_hist = client.get_metric_history(run_id, "val_accuracy")
    val_by_step = {v.step: v for v in val_hist}

    epochs_done = len(loss_hist)

    print()
    print("Phase 3 — Haplogroup fine-tuning status")
    print("─" * 57)
    print(f"Run ID   : {run_id[:16]}...   Status: {status}")
    print(f"Elapsed  : {fmt_h(elapsed_h):<14} Started: {started_str}")
    print()

    if not loss_hist:
        print("  No epochs complete yet — dataset build or epoch 1 in progress.")
    else:
        print(f"  {'Epoch':<7} {'Train Loss':<12} {'Val Acc':<10} {'Train ended':<14} {'Val ended'}")
        print(f"  {'─'*5:<7} {'─'*10:<12} {'─'*7:<10} {'─'*11:<14} {'─'*9}")
        for lh in sorted(loss_hist, key=lambda x: x.step):
            vh = val_by_step.get(lh.step)
            train_elapsed_h = (lh.timestamp - start_ms) / 3_600_000
            if vh:
                val_elapsed_h = (vh.timestamp - start_ms) / 3_600_000
                val_str = f"+{val_elapsed_h:.2f}h"
                acc_str = f"{vh.value * 100:.1f}%"
            else:
                val_str = "(running…)"
                acc_str = "—"
            print(
                f"  {lh.step + 1:<7} {lh.value:<12.4f} {acc_str:<10} "
                f"+{train_elapsed_h:.2f}h{'':<8} {val_str}"
            )

    print()

    if status != "RUNNING":
        if epochs_done == MAX_EPOCHS:
            print(f"  FINISHED — all {MAX_EPOCHS} epochs complete. Check models/finetune_haplogroup_paper/")
        else:
            print(f"  Run is {status} with {epochs_done}/{MAX_EPOCHS} epochs recorded.")
        return

    # Derive per-epoch timing from actual MLflow timestamps of completed epochs.
    avg_train_h, avg_val_h = measure_epoch_timings(loss_hist, val_by_step, start_ms)
    epoch_h = avg_train_h + avg_val_h

    # Time elapsed since the last epoch boundary (end of most recent val pass).
    if val_by_step and epochs_done > 0:
        last_val_ms = val_by_step[sorted(val_by_step)[-1]].timestamp
        time_into_epoch = (time.time() * 1000 - last_val_ms) / 3_600_000
    else:
        # No epoch complete yet — estimate from run start
        time_into_epoch = elapsed_h

    current_epoch = epochs_done + 1

    if time_into_epoch < avg_train_h:
        pct = min(time_into_epoch / avg_train_h * 100, 100)
        phase = f"epoch {current_epoch}/{MAX_EPOCHS} training pass"
        remaining_train = avg_train_h - time_into_epoch
        remaining_h = remaining_train + avg_val_h + (MAX_EPOCHS - current_epoch) * epoch_h
    else:
        time_into_val = time_into_epoch - avg_train_h
        pct = min(time_into_val / avg_val_h * 100, 100)
        phase = f"epoch {current_epoch}/{MAX_EPOCHS} validation pass"
        remaining_val = max(0.0, avg_val_h - time_into_val)
        remaining_h = remaining_val + (MAX_EPOCHS - current_epoch) * epoch_h

    remaining_h = max(0.0, remaining_h)
    eta_dt = datetime.now() + timedelta(hours=remaining_h)
    nearing = time_into_epoch > epoch_h
    pct_str = "  (nearing completion)" if nearing else f"  (~{pct:.0f}% through)"

    print(f"  Progress  : {phase}{pct_str}")
    print(f"  Per-epoch : ~{fmt_h(avg_train_h)} train  +  ~{fmt_h(avg_val_h)} val  (measured)")
    print(f"  Remaining : ~{fmt_h(remaining_h)}")
    print(f"  ETA       : {eta_dt.strftime('%Y-%m-%d %H:%M')}")
    print()


if __name__ == "__main__":
    main()
