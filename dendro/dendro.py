from typing import Any
import matplotlib.axes
import matplotlib.figure
import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

from .utils import nodes_to_communities, autoscale_fig_width


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

        # TODO - Result tree dict
        # self.R: dict = {}

    def _set_default_colormap(self):
        self._cluster_colors_list = [
            tuple(rgb) for rgb in np.random.rand(len(self.communities) + 1, 3)
        ]
        self._cluster_colors_dict = {
            hash(tuple(cluster)): color
            for cluster, color in zip(self.communities, self._cluster_colors_list)
        }

    def _set_random_colors_with_seed(self, seed):
        if seed is not None:
            np.random.seed(seed)

        self._set_default_colormap()

    def _caluclate_Y_levels(self, yaxis_abs_log: bool):
        """
        Calculates essential values to plot on Y axis:
            - `mod increments` - the absolute increase in the modularity metric
            that each hierarchical recursion level has provided;
            undergone abs. of natural logarithm if `yaxis_abs_log` is True.

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
        ax: matplotlib.axes.Axes | None = None,  # May be moved to kwargs later
        fig: matplotlib.figure.Figure | None = None,  # May be moved to kwargs later
        figsize: tuple | None = None,
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

            ax (matplotlib.axes.Axes | None, optional):
                Axes to plot the dendrogram on.
                If `None`, a new figure and axes instance will be created (default).
                This can be useful if the proper figsize scaling is difficult to achieve.
                The user can experiment with different plot sizes and chose the best one.

            fig (matplotlib.figure.Figure | None, optional):
                Goes together with the ax param.
                If `None` and `ax` is None, a new figure and axes instance will be created
                (default). If exactly one of them is not None, a ValueError will be raised.

            figsize (tuple | None, optional):
                If not `None`, a new fig and ax will be created with a given figsize.
                If `None`, will be ignored and other parameters (ax, fig) will be taken into
                account when establishing new ax and fig.
                Allows the user to specify the size of the plot in a more friendly way
                than passing ax and fig.
        """
        Y_levels, _ = self._caluclate_Y_levels(yaxis_abs_log)

        # TODO - implement color maps (cmaps)
        if color_seed:
            self._set_random_colors_with_seed(color_seed)
        # Default color options
        colors = self._cluster_colors_list
        cluster_colors = self._cluster_colors_dict

        # Define the mapping between nodes and their position on the plot
        nodes = np.array(self.G.nodes)
        leafs_clustering_ordering = [c for cluster in self.communities for c in cluster]
        node_positions = {
            leaf: node for node, leaf in zip(nodes, leafs_clustering_ordering)
        }

        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = self._handle_ax_fig_situation(ax, fig, kwargs)

        # Plot the base
        self._draw_tree_base(
            ax=ax,
            display_nodes=display_leafs,
            Y_levels=Y_levels,
            node_positions=node_positions,
            cluster_colors=cluster_colors,
        )
        self._mark_modularity_increments_style2(ax, Y_levels, nodes, yaxis_abs_log)

        xlabel_rot_angle = xlabel_rotation if xlabel_rotation else 0

        # Draw leafs
        if display_leafs:
            ax.set_xticks(nodes)
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
                legend = ax.legend(
                    handles=legend_handles,
                    title="Clusters and their Nodes",
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.1),
                    ncol=2,
                )
                for i, el in enumerate(legend.get_texts()):
                    el.set_color(colors[i])

        # Otherwise - make sure everything is neat and hidden
        else:
            for label in ax.get_xticklabels():
                label.set_visible(False)

        # Set yticks
        ax.set_yticks(np.sort(np.array([0]+Y_levels)))
        ax.set_yticklabels([round(m,4) for m in self.division_modularities[::-1]])
        # Draw the base of modularity = 0
        ax.axhline(
            y=0,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="Modularity base (0)",
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
        self._set_yaxis_label(yaxis_abs_log, kwargs, ax)

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

    def draw_horizontal(
        self,
        yaxis_abs_log: bool = False,
        node_labels_mapping: dict[int | Any, Any] | None = None,
        ylabel_rotation: float | None = None,
        fig_saving_path: str | None = None,
        show_plot: bool = True,
        color_seed: int | None = None,
        ax: matplotlib.axes.Axes | None = None,  # May be moved to kwargs later
        fig: matplotlib.figure.Figure | None = None,  # May be moved to kwargs later
        figsize: tuple | None = None,
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

            ax (matplotlib.axes.Axes | None, optional):
                Axes to plot the dendrogram on.
                If `None`, a new figure and axes instance will be created (default).
                This can be useful if the proper figsize scaling is difficult to achieve.
                The user can experiment with different plot sizes and chose the best one.

            fig (matplotlib.figure.Figure | None, optional):
                Goes together with the ax param.
                If `None` and `ax` is None, a new figure and axes instance will be created
                (default). If exactly one of them is not None, a ValueError will be raised.

            figsize (tuple | None, optional):
                If not `None`, a new fig and ax will be created with a given figsize.
                If `None`, will be ignored and other parameters (ax, fig) will be taken into
                account when establishing new ax and fig.
                Allows the user to specify the size of the plot in a more friendly way
                than passing ax and fig.
        """
        Y_levels, _ = self._caluclate_Y_levels(yaxis_abs_log)

        # TODO - implement color maps (cmaps)
        if color_seed:
            self._set_random_colors_with_seed(color_seed)
        # Default color options
        colors = self._cluster_colors_list
        cluster_colors = self._cluster_colors_dict

        # Orders in which leafs (nodes) appear in the final clustering
        # for dendrogram readability/esthetics purposes
        nodes = np.array(self.G.nodes)
        # Dict of pairs (leaf - order in final clustering, initial node number)
        leafs_clustering_ordering = [c for cluster in self.communities for c in cluster]
        node_positions = {
            leaf: node for node, leaf in zip(nodes, leafs_clustering_ordering)
        }

        if figsize:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = self._handle_ax_fig_situation(ax, fig, kwargs)

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
        self._set_xaxis_label_inverted(yaxis_abs_log, kwargs, ax)

        # Leafs go to Y axis
        ax.set_yticks(nodes)
        self._set_leafs_as_ylabels(
            ax=ax,
            leafs_clustering_ordering=leafs_clustering_ordering,
            node_labels_mapping=node_labels_mapping,
            cluster_colors=cluster_colors,
            ylabel_rot=ylabel_rot_angle,
        )

        ax.set_xticks(np.sort(np.array([0]+Y_levels)))
        ax.set_xticklabels([round(m,4) for m in self.division_modularities[::-1]])
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

        # Hide plot box borders (spines)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)

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

    def _draw_tree_base(
        self,
        ax,
        display_nodes: bool,
        Y_levels: list[float],
        node_positions: dict[int, int],
        cluster_colors: dict[int, tuple],
        **kwargs,
    ):
        division_tree = self.division_tree

        # Esthetics
        vline_color = "gray"
        hline_color = "gray"
        hier_line_alpha = 0.8

        horizontal_coords = {}
        # Traverse the division tree bottom->up
        for level in reversed(range(len(division_tree))):
            level_clustering = division_tree[level]

            # CASE OF FINAL CLUSTERING (LEAFS):
            # For every cluster in the final clustering:
            #   For every leaf in a cluster:
            #       1. scatter it and draw a vertical line coming from it
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

                    if display_nodes:
                        # 1.
                        for x in cluster_leafs_positioned:
                            ax.scatter(
                                x,
                                base_modularity_zero,
                                color=cluster_colors[hash(tuple(cluster))],
                                s=50,
                                alpha=1,
                            )

                        # 2.
                        ax.hlines(
                            y=base_modularity_zero,
                            xmin=xmin,
                            xmax=xmax,
                            colors=cluster_colors[hash(tuple(cluster))],
                            alpha=0.6,
                            linewidth=3,
                        )

                    else:
                        # 2.
                        ax.hlines(
                            y=base_modularity_zero,
                            xmin=xmin,
                            xmax=xmax,
                            colors=cluster_colors[hash(tuple(cluster))],
                            alpha=1,
                            linewidth=5,
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
                # a further division of a given cluster may or may not occur. (A community might,
                # but might not be further divided in later hierarchical search calls.)
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
        ax,
        Y_levels: list[float],
        node_positions: dict[int, int],
        cluster_colors: dict[int, tuple],
        **kwargs,
    ):
        division_tree = self.division_tree

        # Esthetics
        vline_color = "gray"
        hline_color = "gray"
        hier_line_alpha = 0.8

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
                            s=50,
                            alpha=1,
                        )

                    # Change hlines to vlines
                    ax.vlines(
                        x=base_modularity_zero,
                        ymin=ymin,
                        ymax=ymax,
                        colors=cluster_colors[hash(tuple(cluster))],
                        alpha=0.6,
                        linewidth=3,
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

    def _mark_modularity_increments(self, ax, Y_levels, nodes):
        x_annotation_line = max(nodes) + 2
        Y_levels_reversed = list(reversed(Y_levels))
        ymin = self.division_modularities[0]  # 0.0
        color = "gray"

        for y in Y_levels_reversed:
            ymax = y
            y_increment_value = ymax - ymin
            dash_margin = y_increment_value * 0.99
            ax.vlines(
                x=x_annotation_line,
                ymin=ymin + dash_margin,
                ymax=ymax - dash_margin,
                color="gray",
                linestyle="-",
                linewidth=1.5,
                alpha=0.6,
            )

            ax.annotate(
                f"{y_increment_value:.4f}",
                xy=(x_annotation_line + 0.35, (ymin + ymax) / 2),
                xytext=(40, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color=color,
                arrowprops=dict(arrowstyle="-[", lw=1.0, color="black"),
            )

            ymin = ymax

    def _mark_modularity_increments_style2(self, ax, Y_levels, nodes, yaxis_abs_log):
        x_annotation_line = max(nodes) + 2

        Y_levels_reversed = list(reversed(Y_levels))
        ymin = self.division_modularities[0]  # 0
        inc_ms = list(
            np.array(self.division_modularities[1:])-np.array(self.division_modularities[:-1])
        )[::-1]

        for i, y in enumerate(Y_levels_reversed):
            ymax = y
            y_increment_value = ymax - ymin
            dash_margin = 0
            color = "gray"

            ax.vlines(
                x=x_annotation_line,
                ymin=ymin + dash_margin,
                ymax=ymax - dash_margin,
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
            )

            horizontal_line_length = 0.6
            ax.hlines(
                y=ymin + dash_margin,
                xmin=x_annotation_line - horizontal_line_length,
                xmax=x_annotation_line,
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.6,
            )
            ax.hlines(
                y=ymax - dash_margin,
                xmin=x_annotation_line - horizontal_line_length,
                xmax=x_annotation_line,
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.6,
            )
            
            ax.annotate(
                f"{inc_ms[i]:.4f}",
                xy=(x_annotation_line + 0.35, (ymin + ymax) / 2),
                xytext=(40, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                color="gray",
            )

            ymin = ymax

    def _set_leafs_as_xlabels(
        self,
        ax,
        leafs_clustering_ordering,
        node_labels_mapping,
        cluster_colors,
        xlabel_rot,
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
        ax,
        leafs_clustering_ordering,
        node_labels_mapping,
        cluster_colors,
        ylabel_rot,
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
        self, ax, node_positions, communities_labels, cluster_colors, xlabel_rot
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

    def _get_communities_legend_handles(self, cluster_colors, communities_labels):
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

    # TEMPORARY PATCH for autoscaling
    def _handle_ax_fig_situation(self, ax, fig, kwargs):
        if ax and fig:
            return fig, ax

        def xor(a, b):
            return (a and not b) or (b and not a)

        if xor(fig is None, ax is None):
            raise ValueError("ax and fig must be passed both")

        x_width, y_height = self._determine_figsize(kwargs)
        fig, ax = plt.subplots(figsize=(x_width, y_height))

        return fig, ax

    # TODO
    def _determine_figsize(self, kwargs):
        if "figsize" in kwargs:
            figsize = kwargs.get("figsize", (20, 10))
            x_width, y_height = figsize[0], figsize[1]
        else:
            x_width = autoscale_fig_width(np.array(self.G.nodes))
            y_height = 10  # TODO autoscale_fig_height

        return x_width, y_height

    def _set_yaxis_label(self, yaxis_abs_log, kwargs, ax):
        ylabel_default = (
            "Modularity "+r'(max $Q$)'
        )
        ylabel = kwargs.get("ylabel", ylabel_default)
        ax.set_ylabel(ylabel)

    def _set_xaxis_label_inverted(self, yaxis_abs_log, kwargs, ax):
        ylabel_default = (
            "Modularity "+r'(max $Q$)'
        )
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