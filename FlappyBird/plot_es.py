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

# ─── Locate the latest TensorBoard event directory ────────────────────────────
def find_latest_run(log_dir):
    pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No event files under {log_dir!r}")
    latest_file = max(files, key=os.path.getmtime)
    return os.path.dirname(latest_file)

# Point to your ES TensorBoard logs folder
log_dir = "./tensorboard/5es16/"  
run_dir = find_latest_run(log_dir)

# ─── Load events ─────────────────────────────────────────────────────────────
ea = event_accumulator.EventAccumulator(
    run_dir,
    size_guidance={event_accumulator.SCALARS: 0}
)
ea.Reload()

tags    = ["ES/MeanReward"]#, "ES/MaxReward", "ES/BestRewardSoFar"]
colors  = ['#FF0000']#, '#00A79F', '#B51F1F']
labels  = ['Mean Reward']#, 'Max Reward', 'Best Reward So Far']

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

for tag, color, label in zip(tags, colors, labels):
    if tag not in ea.Tags().get("scalars", []):
        continue
    events = ea.Scalars(tag)
    wall_times = np.array([e.wall_time for e in events])
    rel_times  = wall_times - wall_times[0]
    values     = np.array([e.value    for e in events], dtype=np.float32)

    # ─── exponential smoothing ────────────────────────────────────────────────
    a = 0.3
    smooth = np.empty_like(values)
    smooth[0] = values[0]
    for i in range(1, len(values)):
        smooth[i] = a * values[i] + (1 - a) * smooth[i-1]

    # ─── plot the smoothed curve ───────────────────────────────────────────────
    ax.plot(rel_times, smooth,
            color=color,
            linewidth=2.5,
            label=f"{label} (α={a})")


# ─── Style the spines ─────────────────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_visible(True)

# ─── Grid, titles, and labels ─────────────────────────────────────────────────
ax.grid(True, linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('ES on Flappy Bird', pad=20)
ax.set_xlabel('Cumulative Training Time (s)')
ax.set_ylabel('Mean Episode Reward')
ax.set_xlim((0, 600))

# ─── Add plateau line and label ───────────────────────────────────────────────
y0 = 110.0
ax.axhline(y0, linestyle='--', linewidth=2, color='grey')  
ax.text(
    ax.get_xlim()[0],  # left edge
    y0,
    "Plateau reward",
    color="black",
    ha="left",
    fontsize=18,
    va="top"           # align text top at y0 → text sits just below the line
)

# ─── Mark pretrained ES policy ────────────────────────────────────────────────
x0 = 193
ymin, ymax = ax.get_ylim()

# red vertical line
# ─── Mark pretrained ES policy inside the plot ────────────────────────────────
x0 = 193

# red vertical line
ax.axvline(x0, color='black', linestyle='--', linewidth=2)

# label inside the plot, halfway up the y-axis
ax.text(
    x0,
    0.24,                                 # halfway up
    "End of pretraining",
    color="black",
    rotation=90,
    va="center",                         # center vertically on the text
    ha="right",
    fontsize=18,
    transform=ax.get_xaxis_transform()   # x in data coords, y in axes fraction
)

plt.tight_layout()
plt.show()