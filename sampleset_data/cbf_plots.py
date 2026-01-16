import numpy as np
import matplotlib.colors as mcolors


def generate_strongly_contrasted_palettes(n_sizes):

    def make_palette(n, hue_start, hue_end, s=0.7, v=0.9):
        if n == 1:
            hues = [(hue_start + hue_end) / 2]
        else:
            hues = np.linspace(hue_start, hue_end, n, endpoint=False)
        return [mcolors.hsv_to_rgb((h, s, v)) for h in hues]

    # PWL - cooler shades
    pwl_colors = make_palette(n_sizes, hue_start=0.45, hue_end=0.85, s=0.7, v=0.9)

    # ER – warm shades
    er_colors = make_palette(n_sizes, hue_start=0.0, hue_end=0.35, s=0.85, v=0.95)

    return pwl_colors, er_colors
