"""Results chart for README / posts (data = results table of the 13 Oct 2026 run)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TYPES = ["Single figure", "Prior year", "YoY %", "Margin", "Follow-up", "Unanswerable\n(said 'not in filing')"]
DATA = {  # model -> % correct per type
    "Qwen2.5-7B (52% overall)": [49, 43, 50, 38, 30, 100],
    "Granite-3.1-8B (51%)": [42, 50, 52, 40, 40, 100],
    "Phi-3.5-mini (45%)": [43, 43, 43, 38, 10, 80],
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
n, group_w = len(DATA), 0.78
bar_w = group_w / n
for i, (model, vals) in enumerate(DATA.items()):
    for j, v in enumerate(vals):
        x = j - group_w / 2 + i * bar_w + 0.012
        w = bar_w - 0.024  # 2px-ish surface gap between bars
        ax.add_patch(FancyBboxPatch((x, 0), w, v, boxstyle="round,pad=0,rounding_size=0.02",
                                    mutation_aspect=40, facecolor=COLORS[i], edgecolor="none"))
        ax.text(x + w / 2, v + 1.5, f"{v}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.bar(0, 0, color=COLORS[i], label=model)  # legend handle
ax.set_xlim(-0.55, len(TYPES) - 0.45); ax.set_ylim(0, 112)
ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(TYPES, fontsize=10.5, color=INK)
ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels([f"{t}%" for t in [0, 25, 50, 75, 100]], color=INK2, fontsize=10)
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#bdbcb6"); ax.tick_params(length=0)
fig.text(0.06, 0.93, "Small LLMs get about half of 10-K facts wrong, even when the answer is in the filing",
         fontsize=15, fontweight="bold", color=INK)
fig.text(0.06, 0.885, "% correct by question type · 182 questions over 10 SEC 10-K filings · BM25 retrieval, top-8 chunks",
         fontsize=10.5, color=INK2)
ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=10.5, labelcolor=INK)
fig.text(0.06, 0.02, "FinHallu-10K · gold answers from SEC XBRL data · github.com/vedanshmathur7/finhallu-10k",
         fontsize=9, color=INK2)
fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.14)
fig.savefig("results_chart.png", facecolor=SURFACE)
print("saved results_chart.png")
