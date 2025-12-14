import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(self, id, values, left=None, right=None, parent=None):
        self.id = id
        self.values = values
        self.left = left
        self.right = right
        self.parent = parent

    def __str__(self):
        return f"Node(id={self.id}, values={self.values})"
    
    def __repr__(self):
        return f"Node(values={self.values}, left={self.left}, right={self.right})"
    
    def __eq__(self, node):
        if not isinstance(node, Node):
            return False
        return self.id == node.id and sorted(self.values) == sorted(node.values)
    

class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = [root]


def recover_ordering(G, division_tree):
    root = Node(id=hash(tuple(G.nodes)), values=list(G.nodes), left=None, right=None, parent=None)
    nodes = {}

    for level in reversed(range(len(division_tree))):
        level_clustering = division_tree[level]

        if level == len(division_tree) - 1:
            for cluster in level_clustering:
                leaf_node = Node(id=hash(tuple(cluster)), values=cluster, left=None, right=None, parent=None)
                nodes[leaf_node.id] = leaf_node
        else:
            subsequent_clusters = division_tree[level + 1]
            subclusters = {}

            for subsequent_cluster in subsequent_clusters: # children clusters
                for cluster in level_clustering: # parent clusters
                    if set(subsequent_cluster).issubset(set(cluster)): # if subsequent cluster is child of the parent
                        key_clus = hash(tuple(cluster))
                        
                        if key_clus not in subclusters.keys():
                            subclusters[key_clus] = subsequent_cluster
                        else:
                            subclusters[key_clus] = (
                                subclusters.get(key_clus),
                                subsequent_cluster,
                            )                    

            for cluster in level_clustering:
                key_clus = hash(tuple(cluster))

                if type(subclusters[key_clus]) == tuple:
                    c0, c1 = subclusters[key_clus]
                    key1, key2 = hash(tuple(c0)), hash(tuple(c1))

                    child_node_1 = nodes[key1]
                    child_node_2 = nodes[key2]

                    parent_node = Node(id=key_clus, values=cluster, left=child_node_1, right=child_node_2, parent=None)
                    nodes[parent_node.id] = parent_node

                    child_node_1.parent = parent_node
                    child_node_2.parent = parent_node
                    
                    # Sanity checks
                    assert child_node_1.parent == parent_node
                    assert child_node_2.parent == parent_node


                # only one subcommunity - list
                elif type(subclusters[key_clus]) == list and level != 0:
                    pass # Nothing to do in here

    root = nodes[root.id]
    return nodes, root


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def build_graph(G, node, division_modularities, level=0):
    """Recursively traverse the binary tree and add edges."""
    if node is None:
        return

    def divide_label(values, len_per_line=6) -> str:
        lines = []
        for i in range(0, len(values), len_per_line):
            line = values[i:i+len_per_line]
            lines.append(", ".join(map(str, line)))
        return "\n".join(lines)

    label = divide_label(node.values)
    mod_value = division_modularities[level]
    color = "#4c72b0" if node.left or node.right else "#d35d6e"  # blue for internal, red for leaves
    values = node.values

    # pr_id = comm_hash_to_pr_id[hash(tuple(values))]
    # chain_strength = pr_id_to_chain_strength[pr_id]
    # chain_strength = comm_hash_to_ch_str[hash(tuple(values))]
    chain_strength = 0.93

    G.add_node(node.id, label=label, mod_value=mod_value, color=color, values=values, level=level, chain_strength=chain_strength)

    if node.left:
        G.add_edge(node.id, node.left.id)
        build_graph(G, node.left, division_modularities, level + 1)
    if node.right:
        G.add_edge(node.id, node.right.id)
        build_graph(G, node.right, division_modularities, level + 1)

    
def center_parents(G, pos):
    """Center parents above children."""
    for n, data in sorted(G.nodes(data=True), key=lambda x: x[1]['level'], reverse=True):
        children = list(G.successors(n))
        if len(children) == 2:
            x1, _ = pos[children[0]]
            x2, _ = pos[children[1]]
            _, y = pos[n]
            pos[n] = ((x1 + x2)/2, y)
    return pos


def print_tree(node, level=0):
    if node is not None:
        print_tree(node.right, level + 1)
        print(' ' * 8 * level + '->', node)
        print_tree(node.left, level + 1)


from QHyper.solvers.base import SamplesetData


class DivisionNode:
    def __init__(self, community_hash: str | int, node: Node, sampleset_data: SamplesetData | None):
        self.community_hash: int | str = community_hash
        self.node: Node = node
        self.sampleset_data: SamplesetData | None = sampleset_data
        
    @property
    def id(self):
        return self.node.id

    @property
    def values(self):
        return self.node.values
    
    @property
    def left(self):
        return self.node.left
    
    @property
    def right(self):
        return self.node.right
    
    @property
    def parent(self):
        return self.node.parent
    
    @property
    def problem_id(self):
        return self.sampleset_data.problem_id if self.sampleset_data else None
    
    @property
    def chain_strength(self):
        return self.sampleset_data.chain_strength if self.sampleset_data else None
    
    @property
    def chain_break_fraction(self):
        return self.sampleset_data.chain_break_fraction if self.sampleset_data else None
    
    @property
    def chain_break_method(self):
        return self.sampleset_data.chain_break_method if self.sampleset_data else None
    
    def __str__(self):
        return f"ExtendedNode(id={self.id}, community={self.values}, left={self.left}, right={self.right}, community_hash={self.community_hash}, problem_id={self.problem_id}, chain_strength={self.chain_strength}, chain_break_fraction={self.chain_break_fraction}, chain_break_method={self.chain_break_method})"
    
    def __repr__(self):
        return f"ExtendedNode(id={self.id}, community={self.values}, left={self.left}, right={self.right}, community_hash={self.community_hash}, problem_id={self.problem_id}, chain_strength={self.chain_strength}, chain_break_fraction={self.chain_break_fraction}, chain_break_method={self.chain_break_method})"
    
    def __eq__(self, node):
        if not isinstance(node, DivisionNode):
            return False
        return self.id == node.id and sorted(self.values) == sorted(node.values)


def recover_info_ordering(root, nodes, sampleset_metadata):
    extended_nodes = {nodes_k: DivisionNode(community_hash=node.id, node=node, sampleset_data=sampleset_metadata[nodes_k]) if nodes_k in sampleset_metadata.community_hash else DivisionNode(community_hash=node.id, node=node, sampleset_data=None) for nodes_k, node in nodes.items()}
    for _, v in extended_nodes.items():
        if v.left:
            v.node.left = extended_nodes[v.left.id]
        if v.right:
            v.node.right = extended_nodes[v.right.id]
        # if v.parent:
            #     v.node.parent = extended_nodes[v.parent.id]
   
    root_extended = DivisionNode(community_hash=root.id, node=root, sampleset_data=sampleset_metadata[root.id])
    return extended_nodes, root_extended


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def build_graph_extended(G, division_modularities, node, level=0):
    """Recursively traverse the binary tree and add edges."""

    if node is None:
        return

    def divide_label(values, len_per_line=6) -> str:
        lines = []
        for i in range(0, len(values), len_per_line):
            line = values[i:i+len_per_line]
            lines.append(", ".join(map(str, line)))
        return "\n".join(lines)
    
    community_hash = node.community_hash
    community = node.values
    problem_id = node.problem_id
    chain_strength = node.chain_strength
    chain_break_fraction = node.chain_break_fraction

    label = divide_label(node.values)
    mod_value = division_modularities[level]
    # color = "#4c72b0" if node.left or node.right else "#d35d6e"  # blue for internal, red for leaves


    G.add_node(node.id, label=label, mod_value=mod_value, values=community, level=level, chain_strength=chain_strength, chain_break_fraction=chain_break_fraction, community_hash=community_hash, problem_id=problem_id)

    if node.left:
        G.add_edge(node.id, node.left.id)
        build_graph_extended(G, division_modularities, node.left, level + 1)
    if node.right:
        G.add_edge(node.id, node.right.id)
        build_graph_extended(G, division_modularities, node.right, level + 1)


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm


ROUND_DIGITS = 4


def center_parents(G, pos):
    """Center parents above children."""
    for n, data in sorted(G.nodes(data=True), key=lambda x: x[1]['level'], reverse=True):
        children = list(G.successors(n))
        if len(children) == 2:
            x1, _ = pos[children[0]]
            x2, _ = pos[children[1]]
            _, y = pos[n]
            pos[n] = ((x1 + x2)/2, y)
    return pos


def plot_tree_extended(root, division_modularities, figsize=(14,8), cmap=plt.cm.viridis, value=None):
    """Create and visualize the binary tree graph with a modern aesthetic."""
    G = nx.DiGraph()
    build_graph_extended(G, division_modularities, root)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    center_parents(G, pos)

    # Node classification
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    internals = [n for n in G.nodes() if G.out_degree(n) > 0]
    labels = nx.get_node_attributes(G, 'label')

    # Modern color scheme
    internal_color = "#b8d3e9"
    leaf_color = "#f2a7a7"

    # plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use("seaborn-v0_8-white")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                           nodelist=internals,
                           node_color=internal_color,
                           node_size=2800,
                           ax=ax,
                           edgecolors="#2b3a42",
                           linewidths=0.8)

    nx.draw_networkx_nodes(G, pos,
                           nodelist=leaves,
                           node_color=leaf_color,
                           node_size=2800,
                           ax=ax,
                           edgecolors="#2b3a42",
                           linewidths=0.8)

    # Draw edges — subtle and transparent
    nx.draw_networkx_edges(G, pos,
                           arrows=False,
                           edge_color="#7d8ca3",
                           width=1.0,
                           ax=ax,
                           alpha=0.6)
    


    cmap = cmap
    values = [G.nodes[n]['chain_strength'] for n in G.nodes()]
    valid_values = [v for v in values if v is not None]
    norm = mcolors.Normalize(vmin=min(valid_values), vmax=max(valid_values))
    # none_color = "#d3d3d3"
    none_color = "#CE6060"
    colors = {v: mcolors.to_hex(cmap(norm(v))) if v is not None else none_color for v in values}

    import math

    def is_bright(rgb, threshold=240):
        r, g, b = rgb
        lum = 0.2126*r + 0.7152*g + 0.0722*b
        return lum >= threshold
    
    def is_close_to_white(rgb, threshold=30):
        """
        rgb: (R, G, B) tuple, 0–255
        threshold: max distance from white to be considered 'close'
        """
        r, g, b = rgb
        # Euclidean distance to white
        dist = math.sqrt((255 - r)**2 + (255 - g)**2 + (255 - b)**2)
        return dist < threshold
    
    def is_close_to_gray(color_hex, sat_threshold=0.15):
        """
        Returns True if color is low saturation (grayish/whitish)
        color_hex: "#rrggbb"
        sat_threshold: maximum saturation allowed to still be considered 'gray'
        """
        rgb = mcolors.to_rgb(color_hex)
        h, s, v = mcolors.rgb_to_hsv(rgb)
        return s < sat_threshold

    # Draw labels with clean, modern typography
    for node, (x, y) in pos.items():
        node_data = G.nodes[node]
        mod_value = node_data['mod_value']

        color = colors[node_data['chain_strength']]
        color_rgb = mcolors.to_rgb(color)
        rgb_255 = tuple(int(255 * x) for x in color_rgb)

        # if is_close_to_white(rgb_255, 30):
        # if is_bright(rgb_255, 200):
        if is_close_to_gray(color, sat_threshold=0.4):
            text_color = "#1e2a36"  # dark text for bright backgrounds
        else:
            text_color = "white"  # light text for dark backgrounds


        values = node_data['values']
        chain_strength = node_data['chain_strength']
        chain_break_fraction = node_data['chain_break_fraction']

        # label = f"{len(values)}\n{mod_value:.2f}"
        # text_color = "white" if node in leaves else "#1e2a36"
        # text_color = "white"
        font_weight = "bold" if node in leaves else "semibold"
        # label1 = str(len(values))
        # label2 = f"{mod_value:.2f}"
        label1 = f"Pr.size: {len(values)}"
        ch_str_label = f"{chain_strength:.3f}" if chain_strength is not None else chain_strength
        label2 = f"Ch.str.: {ch_str_label}"
        cbf_label = f"{chain_break_fraction:.3f}" if chain_break_fraction is not None else chain_break_fraction
        label3 = f"CBF: {cbf_label}"
        # label = label1 + "\n" + label2
        label = label1 + "\n" + label2 + "\n" + label3

        ax.text(x, y,
                 label,
                 ha="center", va="center",
                #  ha="center",
                 fontsize=9,
                 color=text_color,
                 fontweight=font_weight,
                 wrap=True,
                 bbox=dict(facecolor=color,
                           boxstyle="round,pad=0.4",
                           edgecolor="#e6e6e6",
                           lw=0.8,
                           alpha=0.85)
        )

    levels = sorted(set(nx.get_node_attributes(G, 'level').values()))
    modularities = [division_modularities[l] for l in levels]

    # Annotate modularity per level at the right of the plot
    xmax = max(x for x, y in pos.values())
    pos_values_sorted = sorted(pos.values(), key=lambda x: (x[1], x[0]))
    last_1, last_2 = pos_values_sorted[-1], pos_values_sorted[-2]
    dx = abs(last_2[0] - last_1[0])
    for lvl, m in zip(levels, modularities):
        # Find nodes at this level
        nodes_at_level = [n for n, d in G.nodes(data=True) if d['level'] == lvl]
        if nodes_at_level:
            # Use mean X position of nodes at this level for placement
            plt.text(xmax + dx/2, np.mean([pos[n][1] for n in nodes_at_level]), f"{m:.4f}",
                     ha="left", va="center", fontsize=9, color="#444444", fontweight="semibold")
            
            plt.hlines(y=np.mean([pos[n][1] for n in nodes_at_level]),
                      xmin=(xmax+dx) - (xmax+dx), xmax=xmax+dx/2,
                      colors="#bbbbbb", linestyles="dashed", lw=0.8, alpha=0.6)
            
    legend_elements = [
        Patch(facecolor=internal_color, edgecolor="#2b3a42", label="Internal nodes"),
        Patch(facecolor=leaf_color, edgecolor="#2b3a42", label="Leaf nodes"),
    ]

    plt.legend(handles=legend_elements,
            # loc="upper left",   # or "lower right" depending on your layout
            frameon=False,
            fontsize=10)
    


    # extended_colors = [none_color] + [cmap(i) for i in np.linspace(0, 1, 256)]
    # extended_cmap = ListedColormap(extended_colors)
    
    # sm = plt.cm.ScalarMappable(cmap=extended_cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    # ticks = ["None"] + sorted(set(valid_values))
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels([f"{t:.4f}" if t is not None else "None" for t in ticks])
    # cbar.set_label("Chain Strength", fontsize=10)
    
    # color_array = [cmap(i) for i in np.linspace(0, 1, 256)]
    # extended_colors = [mcolors.to_rgba(none_color)] + color_array
    # extended_cmap = ListedColormap(extended_colors)

    # # Normalization: 0 = None, numeric values from 0.0001 upwards
    # vmin, vmax = 0, max(valid_values)
    # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # # Create ScalarMappable
    # sm = plt.cm.ScalarMappable(cmap=extended_cmap, norm=norm)
    # sm.set_array([])

    # # Plot colorbar
    # cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
    # cbar = plt.colorbar(sm, ax=ax, cax=cbar_ax, orientation="horizontal")

    # # Numeric ticks only! Map None to 0
    # ticks = [0] + list(np.linspace(min(valid_values), max(valid_values), 5))
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(["None"] + [f"{t:.3f}" for t in ticks[1:]])
    # cbar.set_label("Chain strength", fontsize=10)

    valid_values_sorted = sorted(valid_values)

    # Create colormap
    cmap_vals = plt.cm.PuBu(np.linspace(0, 1, len(valid_values_sorted)))
    colors = np.vstack([mcolors.to_rgba(none_color), cmap_vals])
    discrete_cmap = ListedColormap(colors)

    # Create boundaries: first bin = None
    bounds = [-0.01] + valid_values_sorted
    norm = BoundaryNorm(bounds, discrete_cmap.N)

    # ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=norm)
    sm.set_array([])

    # Colorbar
    # fig, ax = plt.subplots(figsize=(6,1))
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")

    # Ticks
    ticks = [0] + valid_values_sorted
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(["None"] + [f"{t:.3f}" for t in valid_values_sorted])
    cbar.set_label("Chain Strength", fontsize=10, labelpad=20)

    plt.title("Chain strength & problem size in the hierarchical division tree", fontsize=16, fontweight="bold", color="#2b3a42", pad=20)
    # plt.axis("off")
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Modularity")
    ax.yaxis.set_label_position("right") 
    plt.tight_layout()
    # plt.show()

from matplotlib import colormaps
cmap = colormaps.get_cmap("PuBu")
cmap