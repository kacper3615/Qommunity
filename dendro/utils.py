import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

from .dendro_config import FIG_AUTOSCALING


def nodes_to_communities(communities: list) -> dict:
    result = {}
    for i in range(len(communities)):
        for node in communities[i]:
            result[node] = i
    return result


def is_valid_color(color):
    try:
        # Converts any valid color to RGBA
        mcolors.to_rgba(color)
        return True
    except ValueError as ve:
        print(ve)
        return False


def get_colorlist(
    cmap: mcolors.ListedColormap | sns.palettes._ColorPalette | list, n_communities: int
):
    length_err_msg = (
        "The number of colors in the colormap must be "
        "at least equal to the number of communities."
    )
    if isinstance(cmap, list):
        # Too little colors to paint all communities
        if len(cmap) < n_communities:
            raise ValueError(length_err_msg)
        validated = [is_valid_color(color) for color in cmap]
        # Invalid colors
        if not all(validated):
            invalid = np.array(cmap)[np.array(validated) == False].tolist()
            raise ValueError(f"Invalid colors in the colormap: {invalid}")
        return cmap

    elif isinstance(cmap, mcolors.ListedColormap):
        if len(cmap.colors) < n_communities:
            raise ValueError(length_err_msg)
        return cmap.colors

    elif isinstance(cmap, sns.palettes._ColorPalette):
        if len(cmap) < n_communities:
            raise ValueError(length_err_msg)
        return cmap
    else:
        raise ValueError(
            "Unsupported colormap type. The supported types are: "
            "list, matplotlib.colors.ListedColormap and "
            "seaborn.palettes._ColorPalette."
        )


def autoscale_fig_width(num_nodes: int) -> float:
    default_fig_width = FIG_AUTOSCALING["default_fig_width"]
    node_factor = FIG_AUTOSCALING["node_factor"]
    max_fig_width = FIG_AUTOSCALING["max_fig_width"]
    width = default_fig_width + num_nodes * node_factor
    return min(width, max_fig_width)
