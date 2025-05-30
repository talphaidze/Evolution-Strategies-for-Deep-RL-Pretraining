import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing import event_accumulator

font_path = 'Lato-Regular.ttf'

if os.path.isfile(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rc('font', family='Lato', size=20)
else:
    plt.rcdefaults()

def find_latest_run(log_dir):
    """Find the subdirectory with the most recent TensorBoard event file."""
    pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No event files under {log_dir!r}")
    latest_file = max(files, key=os.path.getmtime)
    return os.path.dirname(latest_file)

def load_scalar(run_dir, tag):
    """Load (relative time, value) pairs for a given scalar tag."""
    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        raise KeyError(f"Tag {tag!r} not in {run_dir}")
    events = ea.Scalars(tag)
    wall_times = np.array([e.wall_time for e in events])
    rel_times  = wall_times - wall_times[0]
    values     = np.array([e.value for e in events], dtype=np.float32)
    return rel_times, values

# ─── CONFIG ───────────────────────────────────────────────────────────────────
runs = [
    ("./tensorboard/dqn/DQN_1",    "DQN",    '#FF0000'),
    ("./tensorboard/dqn_es/DQN_1", "ES + DQN", '#00A79F'),
]
tag_to_plot = "rollout/ep_rew_mean"

# ─── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for log_dir, label, color in runs:
    run_dir = find_latest_run(log_dir)
    times, vals = load_scalar(run_dir, tag_to_plot)
    ax.plot(times, vals, color=color, linewidth=2.5, label=label)


# Style spines
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_visible(True)

# Grid, labels, legend
ax.grid(True, linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('DQN vs. ES + DQN on Flappy Bird', pad=20)
ax.set_xlabel('Cumulative Training Time (s)')
ax.set_ylabel('Mean Episode Reward')
ax.legend(fontsize=15)
ax.set_xlim((0, 600))
# ─── before your plotting loop ───────────────────────────────────────────────
runs_data = []

# ─── PLOT ─────────────────────────────────────────────────────────────────────
for log_dir, label, color in runs:
    run_dir = find_latest_run(log_dir)
    times, vals = load_scalar(run_dir, tag_to_plot)

    ax.plot(times, vals, color=color, linewidth=2.5, label=label)

    # stash the numeric arrays for later
    runs_data.append((times, vals, color))

# ─── then after styling but before tight_layout ───────────────────────────────
for times, vals, color in runs_data:
    # find the x‐coordinate (time) where the reward is max
    x_peak = times[np.argmax(vals)]

    # draw the vertical line
    ax.axvline(
        x_peak,
        linestyle='--',
        linewidth=2,
        color=color
    )

    # label it inside the plot
    ax.text(
        x_peak,
        ax.get_ylim()[1] * 0.34,   # 95% up the y‐axis
        "Peak reward",
        rotation=90,
        va='top',
        ha='right',
        color='black',
        fontsize=19
    )


plt.tight_layout()
plt.show()

