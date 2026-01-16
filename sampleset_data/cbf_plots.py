import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

n_sizes = len(graph_sizes)

pwl_cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n_sizes))
er_cmap  = plt.cm.plasma(np.linspace(0.2, 0.9, n_sizes))

import matplotlib.colors as mcolors
import numpy as np

# def make_contrasted_palette(n, hue_start=0, hue_end=1, s=0.7, v=0.9):
#     """
#     Generuje n kolorów maksymalnie różniących się w obrębie jednego zakresu hue.
#     hue_start, hue_end ∈ [0,1] - zakres barw w kole HSV
#     """
#     if n == 1:
#         hues = [(hue_start + hue_end)/2]
#     else:
#         hues = np.linspace(hue_start, hue_end, n, endpoint=False)
#     return [mcolors.hsv_to_rgb((h, s, v)) for h in hues]

# n_sizes = len(graph_sizes)
# pwl_colors = make_contrasted_palette(n_sizes, hue_start=0.5, hue_end=0.75)
# er_colors  = make_contrasted_palette(n_sizes, hue_start=0.0, hue_end=0.15)
import matplotlib.colors as mcolors
import numpy as np

def generate_contrasted_palettes(n_sizes):
    """
    Generuje dwie wyraźnie różne palety kolorów:
    - pwl_colors: zimne odcienie (cyjan -> niebieski -> fiolet)
    - er_colors: ciepłe odcienie (czerwony -> pomarańcz -> żółty)

    n_sizes: liczba serii / graph_sizes
    Returns: (pwl_colors, er_colors) listy RGB
    """
    def make_palette(n, hue_start, hue_end, s=0.7, v=0.9):
        if n == 1:
            hues = [(hue_start + hue_end)/2]
        else:
            hues = np.linspace(hue_start, hue_end, n, endpoint=False)
        return [mcolors.hsv_to_rgb((h, s, v)) for h in hues]

    # Zimne odcienie dla PWL
    pwl_colors = make_palette(n_sizes, hue_start=0.5, hue_end=0.75, s=0.7, v=0.9)

    # Ciepłe odcienie dla ER
    er_colors  = make_palette(n_sizes, hue_start=0.0, hue_end=0.15, s=0.8, v=0.95)

    return pwl_colors, er_colors

n_sizes = len(graph_sizes)
pwl_colors, er_colors = generate_contrasted_palettes(n_sizes)

import matplotlib.colors as mcolors
import numpy as np


def generate_strongly_contrasted_palettes(n_sizes):
    """
    Generuje dwie palety kolorów:
    - pwl_colors: zimne odcienie (cyjan → niebieski → fiolet)
    - er_colors: ciepłe odcienie (czerwony → pomarańcz → żółty)
    """
    def make_palette(n, hue_start, hue_end, s=0.7, v=0.9):
        if n == 1:
            hues = [(hue_start + hue_end)/2]
        else:
            hues = np.linspace(hue_start, hue_end, n, endpoint=False)
        return [mcolors.hsv_to_rgb((h, s, v)) for h in hues]

    # PWL – zimne kolory (szerszy zakres)
    pwl_colors = make_palette(n_sizes, hue_start=0.45, hue_end=0.85, s=0.7, v=0.9)

    # ER – ciepłe kolory (szerszy zakres)
    er_colors = make_palette(n_sizes, hue_start=0.0, hue_end=0.35, s=0.85, v=0.95)

    return pwl_colors, er_colors

pwl_colors, er_colors = generate_strongly_contrasted_palettes(n_sizes)

num_runs = 19

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for gi, gsize in enumerate(graph_sizes):
    # Power-law
    hierarchical_metadata_pwl = graphs_results_pwl[gsize].hierarchical_metadatas
    cbfs = np.empty((num_runs, ), dtype=object)
    for i in range(num_runs):
        cbfs[i] = np.array([hm.chain_break_fraction for hm in hierarchical_metadata_pwl[i]])
    modularities_pwl = graphs_results_pwl[gsize].modularities
    communities_pwl = graphs_results_pwl[gsize].communities
    cbfs_sums_pwl = graphs_results_pwl[gsize].cbfs_sum_per_run

    # Erdos-Renyi
    hierarchical_metadata_er = graphs_results_er[gsize].hierarchical_metadatas
    cbfs = np.empty((num_runs, ), dtype=object)
    for i in range(num_runs):
        cbfs[i] = np.array([hm.chain_break_fraction for hm in hierarchical_metadata_er[i]])
    modularities_er = graphs_results_er[gsize].modularities
    communities_er = graphs_results_er[gsize].communities
    cbfs_sums_er = graphs_results_er[gsize].cbfs_sum_per_run

    # Power-law 
    mods_pwl = []
    cbf_max_pwl = []
    cbf_avg_pwl = []
    non_zero_cbfs_pwl = []
    chain_strengths_pwl = []
    comms_lens_pwl = []
    energies_pwl = []
    indices_pwl = []

    for idx, sm in enumerate(hierarchical_metadata_pwl):
        cbfs = np.array(sm.chain_break_fraction)
        non_zero_cbfs_idxs = np.where(cbfs > 0)
        cbf_max_pwl.append(cbfs.max())
        cbf_avg_pwl.append(cbfs.mean())
        non_zero_cbfs_pwl.append(cbfs[non_zero_cbfs_idxs])
        indices_pwl.append(non_zero_cbfs_idxs)

        chain_strengths_pwl.append(np.nanmean(np.array(sm.chain_strength, dtype=float)))
        mods_pwl.append(modularities_pwl[idx])
        comms_lens_pwl.append(communities_pwl[idx])
    

    # pwl_color = pwl_cmap[gi]
    # er_color  = er_cmap[gi]
    pwl_color = pwl_colors[gi]
    er_color  = er_colors[gi]
    ax.scatter(cbf_max_pwl, mods_pwl, s=50, color=pwl_color, label=f"PWL N={gsize}", edgecolors='k', linewidths=0.5, zorder=3)
    ax.scatter(cbf_max_er, mods_er, s=50, color=er_color, label=f"ER N={gsize}", edgecolors='k', linewidths=0.5, zorder=3)

    # Erdos-Renyi
    mods_er = []
    cbf_max_er = []
    cbf_avg_er = []
    non_zero_cbfs_er = []
    chain_strengths_er = []
    comms_lens_er = []
    energies_er = []
    indices_er = []

    for idx, sm in enumerate(hierarchical_metadata_er):
        cbfs = np.array(sm.chain_break_fraction)
        non_zero_cbfs_idxs = np.where(cbfs > 0)
        cbf_max_er.append(cbfs.max())
        cbf_avg_er.append(cbfs.mean())
        non_zero_cbfs_er.append(cbfs[non_zero_cbfs_idxs])
        indices_er.append(non_zero_cbfs_idxs)

        chain_strengths_er.append(np.nanmean(np.array(sm.chain_strength, dtype=float)))
        mods_er.append(modularities_er[idx])
        comms_lens_er.append(communities_er[idx])

    # ax[1].scatter(cbf_avg_pwl, mods_pwl, s=100)
    # ax[1].scatter(cbf_avg_er, mods_er, s=100)

ax.set_ylabel("Modularity (Q)")
ax.set_xlabel("Max. CBF")
ax.set_title("Powerlaw graphs Modularity (Q) in function of CBF max. for different graph sizes\neach modularity is a result from a hierarchical run (one of 20)");


# ax.set_yscale("log")

# ax.relim()
# ax.autoscale_view()

# xmin, xmax = ax.get_xlim()

handles, labels = ax.get_legend_handles_labels()
pwl_items = [(h, l) for h, l in zip(handles, labels) if l.startswith("PWL")]
er_items  = [(h, l) for h, l in zip(handles, labels) if l.startswith("ER")]

ordered_handles = [h for h, _ in pwl_items + er_items]
ordered_labels  = [l for _, l in pwl_items + er_items]

xmin, xmax = ax.get_xlim()

# Power-law
for i, gsize in enumerate(graph_sizes):
    ax.hlines(
        y=mods_leid_pwl[i],
        xmin=xmin,
        xmax=xmax,
        colors=pwl_colors[i],
        # linestyles="dashed",
        linewidth=10,
        alpha=1,
        label=f"PWL Louvain max N={gsize}",
    )

# Erdős–Rényi
for i, gsize in enumerate(graph_sizes):
    ax.hlines(
        y=mods_leid_er[i],
        xmin=xmin,
        xmax=xmax,
        colors=er_colors[i],
        # linestyles="dotted",
        linewidth=10,
        alpha=1,
        label=f"ER Louvain max N={gsize}",
    )

for i, gsize in enumerate(graph_sizes):
    ax.hlines(
        y=mods_leid_pwl[i],
        xmin=0,
        xmax=0.05,
        transform=ax.get_yaxis_transform(),
        linestyles="dashed",
        colors=pwl_colors[i],
        linewidth=10,
    )
    ax.hlines(
        y=mods_leid_er[i],
        xmin=0,
        xmax=0.05,
        transform=ax.get_yaxis_transform(),
        linestyles="dashed",
        colors=er_colors[i],
        linewidth=10,
    )

legend1 = fig.legend(
    ordered_handles,
    ordered_labels,
    loc='upper right',
    ncol=2,
    bbox_to_anchor=(1.3, 0.95),
    # frameon=False,
)
ax.add_artist(legend1)

hl_handles = []
hl_labels = []
for i, gsize in enumerate(graph_sizes):
    proxy = Line2D([0], [0], color=pwl_colors[i], lw=5)
    hl_handles.append(proxy)
    hl_labels.append(f"PWL Leiden max N={gsize}")

legend2 = fig.legend(
    hl_handles,
    hl_labels,
    loc='center right',  # You can adjust position
    bbox_to_anchor=(1.37, 0.23),  # outside figure
    title="Referential values of max. Q obtained with Leiden",
    frameon=True
)
ax.add_artist(legend2)

plt.tight_layout()
plt.show()