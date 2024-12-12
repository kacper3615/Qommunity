from typing import Any
import warnings
import matplotlib.axes
from matplotlib.colors import ListedColormap
import matplotlib.figure
import networkx as nx
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import seaborn as sns

from .utils import nodes_to_communities, autoscale_fig_width, get_colorlist
from .dendro_config import (
    CLUSTER_HLINES,
    DEFAULT_FIG_HEIGHT,
    DEFAULT_FIGSIZE,
    MOD_INCREMENTS_STYLE,
    MODULARITY_BASE_HLINE,
    TREE_BASE_STYLE,
    CLUSTER_LEGEND_STYLE,
    LEAFS_SETTINGS,
    Y_AXIS_WITH_RESPECT_TO_ABS_LOG,
)


class Dendrogram:
    def __init__(
        self,
        G: nx.Graph,
        communities: list[list],
        division_modularities: list[float],
        division_tree: list[list[list]],
    ) -> None:
        self.G = G
        self.communities = communities
        self.division_modularities = division_modularities
        self.division_tree = division_tree

        self._set_default_colormap()

        # Result dict. Inspired by Scipy's dendrogram
        self.R: dict = {}

    def _set_default_colormap(self):
        self._cluster_colors_list = [
            tuple(rgb) for rgb in np.random.rand(len(self.communities) + 1, 3)
        ]
        self._cluster_colors_dict = {
            hash(tuple(cluster)): color
            for cluster, color in zip(self.communities, self._cluster_colors_list)
        }

    def _set_random_colors_with_seed(self, seed: int):
        if seed is not None:
            np.random.seed(seed)

        self._set_default_colormap()

    def _get_colormap(self, cmap: ListedColormap | list) -> tuple[list, dict]:
        cluster_colors_list = get_colorlist(cmap, len(self.communities))
        cluster_colors_dict = {
            hash(tuple(cluster)): color
            for cluster, color in zip(self.communities, cluster_colors_list)
        }

        return cluster_colors_list, cluster_colors_dict

    def _calculate_Y_levels(self, yaxis_abs_log: bool):
        """
        Calculates essential values to plot on Y axis:
            - `mod increments` - the absolute increase in the modularity metric
            that each hierarchical level has produced;
            undergone abs. of natural logarithm transform if `yaxis_abs_log` is True.

            - `Y_levels` - cumulative sum of increments in modularity at consecutive hierarchy levels,
            starting at the bottom of hierarchy (final clustering).

        Args:
            yaxis_abs_log (bool):
                decides whether take absolute value of the natural logarithm of modularity increments
                (abs of logarithmic transform). Allows to effectively log the Y axis
                (by handling the negative values of the logarithm, due to the
                usually small increases in modularity).

        Returns:
            Y_levels: list[float] - to be marked on Y axis.
            mod_increments: list[float] - the increment in modularity each recursive split has provided
        """
        division_modularities = self.division_modularities

        # Modularity increments
        mod_increments = [
            division_modularities[i + 1] - division_modularities[i]
            for i in range(0, len(division_modularities) - 1)
        ]

        # Apply abs of natural logarithmic transform
        if yaxis_abs_log:
            mod_increments = [abs(math.log(mi)) for mi in mod_increments]

        # Y axis markers
        Y_levels = list(
            reversed(
                [sum(mod_increments[-i - 1 :]) for i, _ in enumerate(mod_increments)]
            )
        )

        self.R["Y_levels"] = Y_levels
        self.R["yaxis_abs_log"] = yaxis_abs_log
        self.R["mod_increments"] = mod_increments

        return Y_levels, mod_increments

    def draw(
        self,
        display_leafs: bool = True,
        yaxis_abs_log: bool = False,
        with_labels: bool = True,
        *,
        node_labels_mapping: dict[int | Any, Any] | None = None,
        communities_labels: list[str] | None = None,
        xlabel_rotation: float | None = None,
        with_communities_legend: bool = True,
        fig_saving_path: str | None = None,
        show_plot: bool = True,
        color_seed: int | None = None,
        cmap: ListedColormap | sns.palettes._ColorPalette | list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        figsize: tuple | None = None,
        decimal_precision: int = 4,
        tight_layout: bool = True,
        **kwargs,
    ):
        """
        Plot the hierarchical community search as a dendrogram.

        The dendrogram illustrates the increments in the metric of modularity
        on each recursion (hierarchy) level of the hierarchical community search.
        The increments are usually quite small as the recursive search progresses,
        hence it is recommended to set `yaxis_abs_log` to True for to incrase readability.

        The dendrogram plots leafs on the X axis, which represent the nodes of
        the input graph and visualises their membership in the detected communities,
        i.e. final clustering division.
        For larger graphs it is recommended to set display_leafs to False
        to increase readibility.

        The leafs represent the elements of the communities detected,
        the clades represent the course of the recursive splits (divisions).

        The dendrogram is plotted in a vertical orientation - leafs on the X axis,
        correspondent modularity increase on the Y axis (default). To plot the dendrogram
        horizontally, use the `draw_horizontal` method.

        Args:
            display_leafs (bool, optional):
                ``True`` (default)
                Plot graph nodes and their community memberships.

                ``False``
                Plot the outline of communities detected.

            yaxis_abs_log (bool, optional): Defaults to False.
                Take absolute value of natural logarithm of modularity increments;
                effectively apply logarithmic transform of Y axis.

            with_labels (bool, optional):.
                Display labels on the X axis. Defaults to True.

            node_labels_mapping (dict[int, Any] | None, optional):
                Set custom labels to leafs. Defaults to None - the default numbering.
                The mapping must be a dict of pairs (node, label), where nodes are original
                graph nodes and labels are the custom labels.

            communities_labels (list[str] | None, optional):
                Set custom labels to detected communities. Defaults to None - the default
                numbering. The labels must be a list of labels as they appear in the detected
                communities order (`communities``).

            xlabel_rotation (float | None, optional):
                Specifies the angle (in degrees) to rotate the leaf labels
                (when ``display_leafs``= True) or the communities labels
                (when ``display_leafs``= False).
                Defaults to None (do not rotate).

            with_communities_legend (bool, optional):
                Show legend with nodes and their community memeberships.
                Applicable (and recommended) when ``display_leafs`` = False.
                Defaults to True.

            fig_saving_path (str | None, optional):
                Path to save the figure.

                ``None``
                Do not save the figure (default).

                ``str``
                Save figure to the given path.

            show_plot (bool, optional):
                Display the plot. Defaults to True.
                Recommended to set it to false in the case of creating and saving
                many dendrograms (in a loop) - big data serialization.

            color_seed (int | None, optional):
                Seed of the random color map generator. Defaults to None.

            cmap (ListedColormap | sns.palettes._ColorPalette | list | None, optional):
                Color map applied to the communities of nodes.
                If `None`, the default color map will be used, characterized by
                the `color_seed` parameter.

                If specified, must be of length at least equal to the number of detected communities.
                If longer, the colors will be truncated to the number of communities.
                Communities will be assigned colors in the order they appear in the list.                

                Defaults to None.

            ax (matplotlib.axes.Axes | None, optional):
                Axes to plot the dendrogram on. Highly recommended to be set by the user.
                If `None`, a new figure and axes instance will be created (default).
                This can be useful if the proper figsize scaling is difficult to achieve.
                The user can experiment with different plot sizes and chose the best one.

            fig (matplotlib.figure.Figure | None, optional):
                Goes together with the ax param. Highly recommended to be set by the user.
                If `None` and `ax` is None, a new figure and axes instance will be created
                (default). If exactly one of them is not None, a ValueError will be raised.

            figsize (tuple | None, optional):
                If not `None`, a new fig and ax will be created with a given figsize.
                If `None`, will be ignored and other parameters (ax, fig) will be taken into
                account when establishing new ax and fig.
                Allows the user to specify the size of the plot in a more friendly way
                than passing ax and fig.

                Will be ignored if `ax` and `fig` are specified.

            decimal_precision (int, optional):
                Precision of decimals on the Y axis. Default set to 4.

            tight_layout (bool, optional):
                Apply tight layout to the plot. Defaults to True.
        """
        Y_levels, _ = self._calculate_Y_levels(yaxis_abs_log)

        # Color maps (cmap)
        if color_seed:
            self._set_random_colors_with_seed(color_seed)

        # Default color options
        colors = self._cluster_colors_list
        cluster_colors = self._cluster_colors_dict

        if cmap:
            colors, cluster_colors = self._get_colormap(cmap)

        # Define the mapping between nodes and their position on the plot
        nodes = np.array(self.G.nodes)
        leafs_clustering_ordering = [c for cluster in self.communities for c in cluster]
        node_positions = {
            leaf: node for node, leaf in zip(nodes, leafs_clustering_ordering)
        }

        if ax and fig and figsize:
            warnings.warn("`ax` and `fig` and `figsize` are specified,"
                          "`figsize` will be ignored.")
        elif figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = self._get_ax_fig(ax, fig, kwargs, horizontal=False)

        # Plot the base
        self._draw_tree_base(
            ax=ax,
            display_leafs=display_leafs,
            Y_levels=Y_levels,
            node_positions=node_positions,
            cluster_colors=cluster_colors,
        )
        # Mark modularity increments at the right side of the plot
        self._mark_modularity_increments(
            ax,
            Y_levels,
            nodes,
            with_respect_to_yaxis_abs_log=Y_AXIS_WITH_RESPECT_TO_ABS_LOG,
            round_decimals=decimal_precision,
        )

        xlabel_rot_angle = xlabel_rotation if xlabel_rotation else 0

        # Draw leafs
        if display_leafs:
            ax.set_xticks(nodes)  # Show ticks
            self._set_leafs_as_xlabels(
                ax=ax,
                leafs_clustering_ordering=leafs_clustering_ordering,
                node_labels_mapping=node_labels_mapping,
                cluster_colors=cluster_colors,
                xlabel_rot=xlabel_rot_angle,
            )

        # Draw communities with labels
        elif with_labels:
            ax.set_xticks([])  # Hide ticks
            self._set_communities_as_xlabels(
                ax=ax,
                node_positions=node_positions,
                communities_labels=communities_labels,
                cluster_colors=cluster_colors,
                xlabel_rot=xlabel_rot_angle,
            )

            # Add legend
            if with_communities_legend:
                legend_handles = self._get_communities_legend_handles(
                    cluster_colors, communities_labels
                )

                legend_title = CLUSTER_LEGEND_STYLE["title"]
                legend_loc = CLUSTER_LEGEND_STYLE["loc"]
                legend_bbox_to_anchor = CLUSTER_LEGEND_STYLE["bbox_to_anchor"]
                legend_ncol = CLUSTER_LEGEND_STYLE["ncol"]

                legend = ax.legend(
                    handles=legend_handles,
                    title=legend_title,
                    loc=legend_loc,
                    bbox_to_anchor=legend_bbox_to_anchor,
                    ncol=legend_ncol,
                )
                for i, el in enumerate(legend.get_texts()):
                    el.set_color(colors[i])

        # Otherwise - make sure everything is neat and hidden
        else:
            for label in ax.get_xticklabels():
                label.set_visible(False)

        # Set yticks
        ax.set_yticks(np.sort(np.array([0] + Y_levels)))
        ax.set_yticklabels(
            [round(m, decimal_precision) for m in self.division_modularities[::-1]]
        )
        # Draw the baseline at 0
        modularity_base_hline_style = LEAFS_SETTINGS[MODULARITY_BASE_HLINE]
        ax.axhline(
            y=0,
            color=modularity_base_hline_style["color"],
            linestyle=modularity_base_hline_style["linestyle"],
            linewidth=modularity_base_hline_style["linewidth"],
            label=modularity_base_hline_style["label"],
        )

        # Hide plot box borders (spines)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if not display_leafs:
            ax.spines["bottom"].set_visible(False)
            ax.xaxis.set_visible(False)
        ax.spines["left"].set_visible(True)

        # Finalize the plot
        self._set_title(kwargs, ax)
        self._set_xaxis_label(kwargs, ax)
        self._set_yaxis_label(kwargs, ax)

        if tight_layout:
            plt.tight_layout()

        if fig_saving_path:
            try:
                fig.savefig(fig_saving_path)
            except Exception as e:
                print(f"Exception occured while saving figure: {e}")

            # For serialization purposes
            # - in case of producing dendrograms in a loop
            # so that mplib does not "hang" on a particular figure
            # resulting in process interruption
            if not show_plot:
                plt.close(fig)

        if show_plot:
            plt.show()

        self.R["fig"] = fig
        self.R["ax"] = ax

        # return self.R

    def draw_horizontal(
        self,
        yaxis_abs_log: bool = False,
        *,
        node_labels_mapping: dict[int | Any, Any] | None = None,
        ylabel_rotation: float | None = None,
        fig_saving_path: str | None = None,
        show_plot: bool = True,
        color_seed: int | None = None,
        cmap: ListedColormap | sns.palettes._ColorPalette | list | None = None,
        ax: matplotlib.axes.Axes | None = None,
        fig: matplotlib.figure.Figure | None = None,
        figsize: tuple | None = None,
        decimal_precision: int = 4,
        tight_layout: bool = True,
        **kwargs,
    ):
        """
        Plotting the dendrgram in a horizontal orientation.
        Axes X and Y are inverted - leafs are plotted on the Y axis, modularity increments
        on the X axis.

        Args:
            yaxis_abs_log (bool, optional): Defaults to False.
                Take absolute value of natural logarithm of modularity increments;
                In the case of horizontal orientation, the axes are inverted, hence
                the transform is marked on the X axis. The "y" in the name stays for
                conventional terminology reasons.

            node_labels_mapping (dict[int, Any] | None, optional):
                Set custom labels to leafs. Defaults to None - the default numbering.
                The mapping must be a dict of pairs (node, label), where nodes are original
                graph nodes and labels are the custom labels.

            xlabel_rotation (float | None, optional):
                Specifies the angle (in degrees) to rotate the leaf labels
                (when ``display_leafs``= True) or the communities labels
                (when ``display_leafs``= False).
                Defaults to None (do not rotate).

            fig_saving_path (str | None, optional):
                Path to save the figure.

                ``None``
                Do not save the figure (default).

                ``str``
                Save figure to the given path.

            show_plot (bool, optional):
                Display the plot. Defaults to True.
                Recommended to set it to false in the case of creating and saving
                many dendrograms (in a loop) - big data serialization.

            color_seed (int | None, optional):
                Seed of the random color map generator. Defaults to None.

            cmap (ListedColormap | sns.palettes._ColorPalette | list | None, optional):
                Color map applied to the communities of nodes.
                If `None`, the default color map will be used, characterized by
                the `color_seed` parameter.

                If specified, must be of length at least equal to the number of detected communities.
                If longer, the colors will be truncated to the number of communities.
                Communities will be assigned colors in the order they appear in the list.                

                Defaults to None.

            ax (matplotlib.axes.Axes | None, optional):
                Axes to plot the dendrogram on. Highly recommended to be set by the user.
                If `None`, a new figure and axes instance will be created (default).
                This can be useful if the proper figsize scaling is difficult to achieve.
                The user can experiment with different plot sizes and chose the best one.

            fig (matplotlib.figure.Figure | None, optional):
                Goes together with the ax param. Highly recommended to be set by the user.
                If `None` and `ax` is None, a new figure and axes instance will be created
                (default). If exactly one of them is not None, a ValueError will be raised.

            figsize (tuple | None, optional):
                If not `None`, a new fig and ax will be created with a given figsize.
                If `None`, will be ignored and other parameters (ax, fig) will be taken into
                account when establishing new ax and fig.
                Allows the user to specify the size of the plot in a more friendly way
                than passing ax and fig.

                Will be ignored if `ax` and `fig` are specified.

            decimal_precision (int, optional):
                Precision of decimals on the Y axis. Default set to 4.

            tight_layout (bool, optional):
                Apply tight layout to the plot. Defaults to True.
        """
        Y_levels, _ = self._calculate_Y_levels(yaxis_abs_log)

        # Color maps (cmap)
        if color_seed:
            self._set_random_colors_with_seed(color_seed)

        # Default color options
        cluster_colors = self._cluster_colors_dict

        if cmap:
            _, cluster_colors = self._get_colormap(cmap)

        # Orders in which leafs (nodes) appear in the final clustering
        # for dendrogram readability/esthetics purposes
        nodes = np.array(self.G.nodes)
        # Dict of pairs (leaf - order in final clustering, initial node number)
        leafs_clustering_ordering = [c for cluster in self.communities for c in cluster]
        node_positions = {
            leaf: node for node, leaf in zip(nodes, leafs_clustering_ordering)
        }

        if ax and fig and figsize:
            warnings.warn("`ax` and `fig` and `figsize` are specified,"
                          "`figsize` will be ignored.")

        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = self._get_ax_fig(ax, fig, kwargs, horizontal=True)

        # Invert axes
        ax.invert_yaxis()
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        # Plot the base
        self._draw_tree_base_horizontal(
            ax=ax,
            Y_levels=Y_levels,
            node_positions=node_positions,
            cluster_colors=cluster_colors,
        )

        ylabel_rot_angle = ylabel_rotation if ylabel_rotation else 0

        self._set_title(kwargs, ax)
        self._set_yaxis_label_inverted(kwargs, ax)
        self._set_xaxis_label_inverted(kwargs, ax)

        # Leafs go to Y axis
        ax.set_yticks(nodes)
        self._set_leafs_as_ylabels(
            ax=ax,
            leafs_clustering_ordering=leafs_clustering_ordering,
            node_labels_mapping=node_labels_mapping,
            cluster_colors=cluster_colors,
            ylabel_rot=ylabel_rot_angle,
        )

        # Set modularity ticks on X axis
        ax.set_xticks(np.sort(np.array([0] + Y_levels)))
        ax.set_xticklabels(
            [round(m, decimal_precision) for m in self.division_modularities[::-1]]
        )

        modularity_base_hline_style = LEAFS_SETTINGS[MODULARITY_BASE_HLINE]
        ax.axvline(
            x=0,
            color=modularity_base_hline_style["color"],
            linestyle=modularity_base_hline_style["linestyle"],
            linewidth=modularity_base_hline_style["linewidth"],
        )

        # Hide plot box borders (spines)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

        if tight_layout:
            plt.tight_layout()

        if fig_saving_path:
            try:
                fig.savefig(fig_saving_path)
            except Exception as e:
                print(f"Exception occured while saving figure: {e}")

            # For serialization purposes
            # - in case of producing dendrograms in a loop
            # so that mplib does not "hang" on a particular figure
            if not show_plot:
                plt.close(fig)

        if show_plot:
            plt.show()

        self.R["fig"] = fig
        self.R["ax"] = ax

        # return self.R

    def _draw_tree_base(
        self,
        ax: matplotlib.axes.Axes,
        display_leafs: bool,
        Y_levels: list[float],
        node_positions: dict[int, int],
        cluster_colors: dict[int, tuple],
        **kwargs,
    ):
        division_tree = self.division_tree

        # Get esthetics settings
        vline_color = TREE_BASE_STYLE["vline_color"]
        hline_color = TREE_BASE_STYLE["hline_color"]
        hier_line_alpha = TREE_BASE_STYLE["hier_line_alpha"]

        leafs_scatter_settings = LEAFS_SETTINGS["LEAFS_SCATTER"]
        leafs_hlines_settings = LEAFS_SETTINGS["LEAFS_HLINES"]

        horizontal_coords = {}
        # Traverse the division tree bottom->up
        for level in reversed(range(len(division_tree))):
            level_clustering = division_tree[level]

            # CASE OF FINAL CLUSTERING (LEAFS):
            # For every cluster in the final clustering:
            #   For every leaf in a cluster:
            #       1. scatter it
            #   2. then draw one horizontal line for visually agglomerating a cluster
            if level == len(division_tree) - 1:
                for cluster in level_clustering:
                    cluster_leafs_positioned = [
                        node_positions[node] for node in cluster
                    ]

                    base_modularity_zero = self.division_modularities[0]

                    # x (horizontal) corrds of the cluster - to know where to start and stop drawing
                    xmin = min(cluster_leafs_positioned)
                    xmax = max(cluster_leafs_positioned)
                    xmid = (xmin + xmax) / 2

                    # In the case of final clustering ymin is the ground/base of 0
                    ymin = base_modularity_zero
                    ymax = Y_levels[level - 1]
                    ax.vlines(
                        x=xmid,
                        ymin=ymin,
                        ymax=ymax,
                        colors=vline_color,
                        alpha=hier_line_alpha,
                    )
                    # Mark x coord mids to know where to simplify plotting the next dendrogram hierarchy level
                    horizontal_coords[hash(tuple(cluster))] = xmid

                    if display_leafs:
                        # 1.
                        for x in cluster_leafs_positioned:
                            ax.scatter(
                                x,
                                base_modularity_zero,
                                color=cluster_colors[hash(tuple(cluster))],
                                s=leafs_scatter_settings["s"],
                                alpha=leafs_scatter_settings["alpha"],
                            )

                        # 2.
                        ax.hlines(
                            y=base_modularity_zero,
                            xmin=xmin,
                            xmax=xmax,
                            colors=cluster_colors[hash(tuple(cluster))],
                            alpha=leafs_hlines_settings["alpha"],
                            linewidth=leafs_hlines_settings["linewidth"],
                        )

                    else:
                        # 2.
                        ax.hlines(
                            y=base_modularity_zero,
                            xmin=xmin,
                            xmax=xmax,
                            colors=cluster_colors[hash(tuple(cluster))],
                            alpha=CLUSTER_HLINES["alpha"],
                            linewidth=CLUSTER_HLINES["linewidth"],
                        )

            # CASE OF NON-FINAL DIVISION TREE LEVELS (CLADES):
            else:
                # In a currently considered hierarchy level of the division tree:
                # A. For each cluster in the clustering of that level:
                #   1. find subclusters of this cluster (the later division of that cluster),
                #      called "subsequent clusters" below;
                #      to know, where to start and end plotting lines (calculate positions);
                #      paying attention to: the order of the clusters in each level of the division tree
                #      shall not be taken for granted, hence we do a double "for" loop;
                #   2. draw appropriate plot elements corresponding to this level of the division tree:
                #      - vertical line connecting "subsequent" (check glossary below) divisions
                #        with the current division;
                #      - horizontal line (Y axis markers) correspondant to that level;

                # B. IMPORTANT: Note that on each hierarchy level of the hierarchical search method
                # a further division of a given cluster may or may not occur. (A community may
                # or may not be further divided in later hierarchical search calls.)
                #
                # That's why a considered community may have:
                #   1. only one subcluster ("simplicifolious") - itself
                #      (in a case of no further divisions),
                #   2. two subclusters (in the case of further binary division).
                # The two cases are handled.

                # GLOSSARY:
                # "Subsequent" - meaning:
                # one hierarchy level lower (deeper) in the division tree <=> ...
                # ... <=> one hierarchy level higher in a reversed division tree
                subsequent_clusters = division_tree[level + 1]
                subclusters = {}

                # A.1
                for subsequent_cluster in subsequent_clusters:
                    for cluster in level_clustering:
                        if set(subsequent_cluster).issubset(set(cluster)):
                            key_clus = hash(tuple(cluster))
                            if key_clus not in subclusters.keys():
                                subclusters[key_clus] = subsequent_cluster
                            else:
                                subclusters[key_clus] = (
                                    subclusters.get(key_clus),
                                    subsequent_cluster,
                                )
                # A.2
                for cluster in level_clustering:
                    y = Y_levels[level]
                    key_clus = hash(tuple(cluster))

                    # CASE B.2:
                    # two subcommunities - tuple (of lists)
                    if type(subclusters[key_clus]) == tuple:
                        c0, c1 = subclusters[key_clus]
                        key1, key2 = hash(tuple(c0)), hash(tuple(c1))
                        # Coords for plotting
                        if (
                            key1 in horizontal_coords.keys()
                            and key2 in horizontal_coords.keys()
                        ):
                            mid_c0 = horizontal_coords[key1]
                            mid_c1 = horizontal_coords[key2]
                            mid = (mid_c0 + mid_c1) / 2
                            horizontal_coords[hash(tuple(cluster))] = mid

                            # Draw vertical line for this level
                            ymin = y
                            ymax = Y_levels[level - 1]
                            if level != 0:
                                ax.vlines(
                                    x=mid,
                                    ymin=ymin,
                                    ymax=ymax,
                                    colors=vline_color,
                                    alpha=hier_line_alpha,
                                )

                        # Connect clusters with horizontal line
                        ax.hlines(
                            y=y,
                            xmin=mid_c0,
                            xmax=mid_c1,
                            colors=hline_color,
                            alpha=hier_line_alpha,
                        )

                    # CASE B.1:
                    # only one subcommunity - list
                    elif type(subclusters[key_clus]) == list and level != 0:
                        # Draw vertical line for this level
                        mid_c0 = horizontal_coords[hash(tuple(cluster))]
                        ymin = y
                        ymax = Y_levels[level - 1]

                        ax.vlines(
                            x=mid_c0,
                            ymin=ymin,
                            ymax=ymax,
                            colors=vline_color,
                            alpha=hier_line_alpha,
                        )

    def _draw_tree_base_horizontal(
        self,
        ax: matplotlib.axes.Axes,
        Y_levels: list[float],
        node_positions: dict[int, int],
        cluster_colors: dict[int, tuple],
        **kwargs,
    ):
        division_tree = self.division_tree

        # Esthetics
        vline_color = TREE_BASE_STYLE["vline_color"]
        hline_color = TREE_BASE_STYLE["hline_color"]
        hier_line_alpha = TREE_BASE_STYLE["hier_line_alpha"]

        leafs_scatter_settings = LEAFS_SETTINGS["LEAFS_SCATTER"]
        leafs_vlines_settings = LEAFS_SETTINGS["LEAFS_VLINES"]

        # Store positions of the nodes
        vertical_coords = {}

        # Iterate the division tree bottom -> up
        for level in reversed(range(len(division_tree))):
            level_clustering = division_tree[level]

            if level == len(division_tree) - 1:
                for i, cluster in enumerate(level_clustering):
                    cluster_leafs_positioned = [
                        node_positions[node] for node in cluster
                    ]

                    base_modularity_zero = self.division_modularities[0]

                    # Adjust coordinates for the rotated plot
                    ymin = min(cluster_leafs_positioned)
                    ymax = max(cluster_leafs_positioned)
                    ymid = (ymin + ymax) / 2

                    xmin = base_modularity_zero
                    xmax = Y_levels[level - 1]

                    # Change vlines to hlines for horizontal dendrogram
                    ax.hlines(
                        y=ymid,
                        xmin=xmin,
                        xmax=xmax,
                        colors=vline_color,
                        alpha=hier_line_alpha,
                    )

                    vertical_coords[hash(tuple(cluster))] = ymid

                    for j, y in enumerate(cluster_leafs_positioned):
                        # Scatter the points (rotate coordinates)
                        ax.scatter(
                            base_modularity_zero,
                            y,
                            color=cluster_colors[hash(tuple(cluster))],
                            s=leafs_scatter_settings["s"],
                            alpha=leafs_scatter_settings["alpha"],
                        )

                    # Change hlines to vlines
                    ax.vlines(
                        x=base_modularity_zero,
                        ymin=ymin,
                        ymax=ymax,
                        colors=cluster_colors[hash(tuple(cluster))],
                        alpha=leafs_vlines_settings["alpha"],
                        linewidth=leafs_vlines_settings["linewidth"],
                    )

            else:
                subsequent_clusters = division_tree[level + 1]
                subclusters = {}

                for i, subsequent_cluster in enumerate(subsequent_clusters):
                    for j, cluster in enumerate(level_clustering):
                        if set(subsequent_cluster).issubset(set(cluster)):
                            if j not in subclusters.keys():
                                subclusters[j] = subsequent_cluster
                            else:
                                subclusters[j] = (
                                    subclusters.get(j),
                                    subsequent_cluster,
                                )

                for i, cluster in enumerate(level_clustering):
                    x = Y_levels[level]

                    if type(subclusters[i]) == tuple:
                        c0, c1 = subclusters[i]
                        key1, key2 = hash(tuple(c0)), hash(tuple(c1))

                        if (
                            key1 in vertical_coords.keys()
                            and key2 in vertical_coords.keys()
                        ):
                            mid_c0 = vertical_coords[key1]
                            mid_c1 = vertical_coords[key2]
                            mid = (mid_c0 + mid_c1) / 2
                            vertical_coords[hash(tuple(cluster))] = mid

                            xmin = x
                            xmax = Y_levels[level - 1]

                            if level != 0:
                                ax.hlines(
                                    y=mid,
                                    xmin=xmin,
                                    xmax=xmax,
                                    colors=vline_color,
                                    alpha=hier_line_alpha,
                                )

                        ax.vlines(
                            x=x,
                            ymin=mid_c0,
                            ymax=mid_c1,
                            colors=hline_color,
                            alpha=hier_line_alpha,
                        )

                    elif type(subclusters[i]) == list and level != 0:
                        mid_c0 = vertical_coords[hash(tuple(cluster))]
                        xmin = x
                        xmax = Y_levels[level - 1]
                        ax.hlines(
                            y=mid_c0,
                            xmin=xmin,
                            xmax=xmax,
                            colors=vline_color,
                            alpha=hier_line_alpha,
                        )

    def _mark_modularity_increments(
        self,
        ax: matplotlib.axes.Axes,
        Y_levels: list[float],
        nodes: list[int],
        with_respect_to_yaxis_abs_log: bool,
        round_decimals: int = 6,
    ):
        """
        Annotates modularity increments on a given matplotlib Axes object.
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The matplotlib Axes object where the annotations will be drawn.
        Y_levels : list[float]
            A list of levels to plot on the Y axis, corresponding either to modularity increments
            or to absolute values of natural log of modularity increments.
        nodes : list[int]
            A list of graph nodes.
        with_respect_to_yaxis_abs_log : bool
            If True, annotations will be based on the absolute natural logarithm of y-axis values.
            If False, annotations will be based on modularity increments.
        round_decimals : int, optional
            The number of decimal places to round the annotation values to (default is 6).
        Returns:
        --------
        None
        """
        x_annotation_line = max(nodes) + 2

        Y_levels_reversed = list(reversed(Y_levels))
        ymin = self.division_modularities[0]  # 0

        mod_increments = list(
            np.array(self.division_modularities[1:])
            - np.array(self.division_modularities[:-1])
        )[::-1]

        # Esthetics
        MIS = MOD_INCREMENTS_STYLE
        line_color = MIS["line_color"]
        linewidth = MIS["linewidth"]
        alpha = MIS["alpha"]
        font_color = MIS["font_color"]
        xytext = MIS["xytext"]

        for i, y in enumerate(Y_levels_reversed):
            ymax = y
            dash_margin = 0

            ax.vlines(
                x=x_annotation_line,
                ymin=ymin + dash_margin,
                ymax=ymax - dash_margin,
                color=line_color,
                linestyle="--",
                linewidth=linewidth,
                alpha=alpha,
            )

            horizontal_line_length = 0.6
            ax.hlines(
                y=ymin + dash_margin,
                xmin=x_annotation_line - horizontal_line_length,
                xmax=x_annotation_line,
                color=line_color,
                linestyle="-",
                linewidth=linewidth,
                alpha=alpha,
            )
            ax.hlines(
                y=ymax - dash_margin,
                xmin=x_annotation_line - horizontal_line_length,
                xmax=x_annotation_line,
                color=line_color,
                linestyle="-",
                linewidth=linewidth,
                alpha=alpha,
            )

            # The with_respect_to_yaxis_abs_log param decides whether to annotate
            # abs-logged modularity increments or the increments themselves
            y_increment_value = ymax - ymin

            if with_respect_to_yaxis_abs_log:
                annotation_value = y_increment_value
            else:
                annotation_value = mod_increments[i]

            ax.annotate(
                f"{annotation_value:.{round_decimals}f}",
                xy=(x_annotation_line + 0.35, (ymin + ymax) / 2),
                xytext=xytext,
                textcoords="offset points",
                ha="right",
                va="center",
                color=font_color,
            )

            ymin = ymax

    def _set_leafs_as_xlabels(
        self,
        ax: matplotlib.axes.Axes,
        leafs_clustering_ordering: list,
        node_labels_mapping: dict[int, str],
        cluster_colors: dict[int, tuple],
        xlabel_rot: float,
    ):
        node_to_cluster_id = nodes_to_communities(self.communities)
        cluster_id_to_hash = {
            i: hash(tuple(cluster)) for i, cluster in enumerate(self.communities)
        }

        if node_labels_mapping:
            ax.set_xticklabels(
                [node_labels_mapping[n] for n in leafs_clustering_ordering],
                rotation=xlabel_rot,
            )
            labels_to_nodes = {v: k for k, v in node_labels_mapping.items()}

            for label in ax.get_xticklabels():
                node_num = int(labels_to_nodes[label.get_text()])
                cluster_id = node_to_cluster_id[node_num]
                label.set_color(cluster_colors[cluster_id_to_hash[cluster_id]])
                label.set_fontweight("bold")

        else:
            ax.set_xticklabels(
                [str(n) for n in leafs_clustering_ordering],
                rotation=xlabel_rot,
            )

            for label in ax.get_xticklabels():
                node_num = int(label.get_text())
                cluster_id = node_to_cluster_id[node_num]
                label.set_color(cluster_colors[cluster_id_to_hash[cluster_id]])
                label.set_fontweight("bold")

        ax.xaxis.set_ticks_position("bottom")

    def _set_leafs_as_ylabels(
        self,
        ax: matplotlib.axes.Axes,
        leafs_clustering_ordering: list,
        node_labels_mapping: dict[int, str],
        cluster_colors: dict[int, tuple],
        ylabel_rot: float,
    ):
        node_to_cluster_id = nodes_to_communities(self.communities)
        cluster_id_to_hash = {
            i: hash(tuple(cluster)) for i, cluster in enumerate(self.communities)
        }

        ax.yaxis.set_ticks_position("left")

        if node_labels_mapping:
            ax.set_yticklabels(
                [node_labels_mapping[n] for n in leafs_clustering_ordering],
                rotation=ylabel_rot,
            )
            labels_to_nodes = {v: k for k, v in node_labels_mapping.items()}

            for label in ax.get_yticklabels():
                node_num = int(labels_to_nodes[label.get_text()])
                cluster_id = node_to_cluster_id[node_num]
                label.set_color(cluster_colors[cluster_id_to_hash[cluster_id]])
                label.set_fontweight("bold")

        else:
            ax.set_yticklabels(
                [str(n) for n in leafs_clustering_ordering],
                rotation=ylabel_rot,
            )

            for label in ax.get_yticklabels():
                node_num = int(label.get_text())
                cluster_id = node_to_cluster_id[node_num]
                label.set_color(cluster_colors[cluster_id_to_hash[cluster_id]])
                label.set_fontweight("bold")

    def _set_communities_as_xlabels(
        self,
        ax: matplotlib.axes.Axes,
        node_positions: dict[int, int],
        communities_labels: list,
        cluster_colors: dict[int, tuple],
        xlabel_rot: float,
    ):
        for i, cluster in enumerate(self.communities):
            cluster_leafs_positioned = [node_positions[node] for node in cluster]
            xmid = (min(cluster_leafs_positioned) + max(cluster_leafs_positioned)) / 2

            label = communities_labels[i] if communities_labels else f"{i}"
            ax.text(
                xmid,  # midpoint of the community
                -0.02,  # slightly below the x-axis
                label,
                color=cluster_colors[hash(tuple(cluster))],
                ha="center",
                va="top",  # Align the label vertically above the axis
                fontweight="bold",
                fontsize=12,
                rotation=xlabel_rot,
                transform=ax.get_xaxis_transform(),  # for proper positioning
            )

    def _get_communities_legend_handles(
        self, cluster_colors: dict[int, tuple], communities_labels: list
    ):
        clusters = {hash(tuple(cluster)): cluster for cluster in self.communities}
        legend_handles = []
        for i, (clus_hash_key, cluster) in enumerate(clusters.items()):
            if communities_labels:
                clus_label = f"{communities_labels[i]}: {cluster}"
            else:
                clus_label = f"{i}: {cluster}"
            color = cluster_colors[clus_hash_key]
            handle = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=clus_label,
                markerfacecolor=color,
                markersize=10,
            )
            legend_handles.append(handle)

        return legend_handles

    def _get_ax_fig(
        self,
        ax: matplotlib.axes.Axes | None,
        fig: matplotlib.figure.Figure | None,
        kwargs: dict,
        horizontal: bool,
    ):
        """
        Extract ax and fig if both passed (if both not None). If not, "
        "determine the figsize and create new ax and fig.

        Args:
            ax (matplotlib.axes.Axes): ax | None
            fig (matplotlib.figure.Figure): fig | None
            kwargs (dict): kwargs dict from the main function
            horizontal (bool): whether the dendrogram is to be horizontal
            important to flip the ax sizes.

        Raises:
            ValueError: When ax and fig are not passed both.

        Returns:
            tuple of matplotlib.axes.Axes and matplotlib.figure.Figure: ax, fig
        """
        if ax and fig:
            return fig, ax

        def xor(a, b):
            return (a and not b) or (b and not a)

        if xor(fig is None, ax is None):
            raise ValueError("ax and fig must be passed both")

        x_width, y_height = self._determine_figsize(kwargs)
        if horizontal:
            x_width, y_height = y_height, x_width
        fig, ax = plt.subplots(figsize=(x_width, y_height))

        return fig, ax

    def _determine_figsize(self, kwargs: dict):
        if "figsize" in kwargs:
            figsize = kwargs.get("figsize", DEFAULT_FIGSIZE)
            x_width, y_height = figsize[0], figsize[1]
        else:
            x_width = autoscale_fig_width(len(self.G.nodes))
            y_height = DEFAULT_FIG_HEIGHT

        return x_width, y_height

    def _set_yaxis_label(self, kwargs, ax):
        ylabel_default = "Modularity " + r"(max $Q$)"
        ylabel = kwargs.get("ylabel", ylabel_default)
        ax.set_ylabel(ylabel)

    def _set_xaxis_label_inverted(self, kwargs, ax):
        ylabel_default = "Modularity " + r"(max $Q$)"
        ylabel = kwargs.get("ylabel", ylabel_default)
        ax.set_xlabel(ylabel)

    def _set_xaxis_label(self, kwargs, ax):
        xlabel = kwargs.get("xlabel", " ")
        ax.set_xlabel(xlabel)

    def _set_yaxis_label_inverted(self, kwargs, ax):
        xlabel = kwargs.get("xlabel", " ")
        ax.set_ylabel(xlabel)

    def _set_title(self, kwargs, ax):
        title = kwargs.get("title", " ")
        ax.set_title(title)
