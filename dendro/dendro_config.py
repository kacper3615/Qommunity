# Config file to be modified by the User
# Sets of configurable Dendrogram plotting style settings

DENDROGRAM_TITLE = "Hierarchical Clustering Dendrogram"
XLABEL_DEFAULT = "Leaf Nodes"
YLABEL_DEFAULT = "Modularity " + r"(max $Q$)"

CLUSTERS_LEGENDBOX_TITLE = "Clusters and their Nodes"


# Main plot Dendrogram settings
DENDROGRAM_PLOT_STYLE = {
    "title": DENDROGRAM_TITLE,
    "xlabel": XLABEL_DEFAULT,
    "ylabel": YLABEL_DEFAULT,
}

# TREE-SKELETON settings (Clades)
TREE_BASE_STYLE = {
    "vline_color": "gray",
    "hline_color": "gray",
    "hier_line_alpha": 0.8,
}

# LEAFS settings
# for display_leafs = True
MODULARITY_BASE_HLINE = "BASE_HLINE"
LEAFS_SCATTER = "LEAFS_SCATTER"
LEAFS_HLINES = "LEAFS_HLINES"
# For horizontal dendrogram
LEAFS_VLINES = "LEAFS_VLINES"

LEAFS_SETTINGS = {
    MODULARITY_BASE_HLINE: {
        "y": 0,
        "color": "gray",
        "linestyle": "--",
        "linewidth": 0.8,
        "label": "Modularity base (0)",
    },

    LEAFS_SCATTER: {
        "s": 50,
        "alpha": 1
    },

    LEAFS_HLINES: {
        "alpha": 0.6,
        "linewidth": 3
    },

    # For horizontal dendrogram
    LEAFS_VLINES: {
        "alpha": 0.6,
        "linewidth": 3
    }
}


# CLUSTERS
# for display_leafs = False
CLUSTER_LEGEND_STYLE = {
    "title": CLUSTERS_LEGENDBOX_TITLE,
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.1),
    "ncol": 2,
}

CLUSTER_HLINES = {
    "alpha": 1,
    "linewidth": 5
}


# Y-AXIS and MODULARITY INCREMENTS settings
# Modularity increments axis at the right
# Only for vertical (standard) dendrogram
MOD_INCREMENTS_STYLE = {
    "line_color": "gray",
    "linestyle": "-",
    "linewidth": 1.5,
    "alpha": 0.6,
    "font_color": "gray",
    "xytext": (40, 0)
}

# This value specifies how to plot the Y-axis
# If set to True, when the user passes yaxis_abs_log=True
# the modularities on the left side of the dendrogram
# will be marked as abs. log values.
# If set to False, the modularities on the left side of
# the dendrogram will remain unmodified.
Y_AXIS_WITH_RESPECT_TO_ABS_LOG = False


# FIGURE SIZE settings
# Specified for vertical dendrogram
# For horizontal dendrogram these values are swapped
DEFAULT_FIG_WIDTH = 20
DEFAULT_FIG_HEIGHT = 10
DEFAULT_FIGSIZE = (DEFAULT_FIG_WIDTH, DEFAULT_FIG_HEIGHT)

# Settings of autoscaling the figure width
# For vertical dendrogram only as it is hard to adjust
# the value for the horizontal dendrogram.

# The horizontal dendrogram is not autoscaled
# the Ax and Fig or figsize must be specified by the User
FIG_AUTOSCALING = {
    "default_fig_width": 10,
    "node_factor": 0.4,
    "max_fig_width": 100
}