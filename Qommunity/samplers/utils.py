def communities_to_list(sample, communities_number) -> list:
    communities = []
    for k in range(communities_number):
        subcommunity = []
        for i in sample:
            if sample[i] == k:
                subcommunity.append(i)
        communities.append(subcommunity)

    return communities


def communities_to_dict(communities) -> dict:
    result = {}
    for i in range(len(communities)):
        for j in communities[i]:
            result[f"x{j}"] = i

    return result

def from_networkx_to_graphtool(G):
    from graph_tool.all import Graph

    # Create graph-tool graph
    gtG = Graph(directed=G.is_directed())
    gtG.add_vertex(len(G.nodes))
    eprop_weight = gtG.new_edge_property("double")

    # Add edges and properties
    for (u, v, data) in G.edges(data=True):
        e = gtG.add_edge(u, v)
        if "weight" in data:
            eprop_weight[e] = data["weight"]
    gtG.edge_properties["weight"] = eprop_weight

    return gtG, eprop_weight