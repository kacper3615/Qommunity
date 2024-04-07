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
