from Qommunity.samplers.hierarchical.hierarchical_sampler import HierarchicalSampler
import networkx as nx


class HierarchicalCommunitySearcher:
    def __init__(self, sampler: HierarchicalSampler) -> None:
        self.sampler = sampler

    def single_community_search(
        self, verbosity: int = 0, community: list | None = None
    ) -> list:
        if not community:
            community = [*range(self.sampler.G.number_of_nodes())]

        if verbosity >= 1:
            print("Starting community detection")
        if verbosity >= 2:
            print("===========================================")
            print("Calculations for graph with", len(community), "nodes in community")
            print("===========================================")

        sample = self.sampler.sample_qubo_to_dict()

        c0, c1 = self._split_dict_to_lists(sample, community)

        if verbosity >= 2:
            print("Base community:", community, sep="\n")
            print("Community division:", c0, c1, sep="\n")
            print("===========================================\n")
        if verbosity >= 1:
            print("Stopping community detection")

        if c0 and c1:
            return [c0] + [c1]
        elif c0:
            return [c0]
        else:
            return [c1]

    def hierarchical_community_search(
        self,
        verbosity: int = 0,
        max_depth: int | None = None,
        division_tree: bool = False,
        return_modularities: bool = False,
    ) -> list:
        if verbosity >= 1:
            print("Starting community detection")

        if max_depth == None:
            if division_tree == False:
                division_tree = None
            else:
                division_tree = []

            result = self._hierarchical_search_recursion(
                verbosity=verbosity,
                level=1,
                max_depth=max_depth,
                division_tree=division_tree,
            )

            if division_tree:
                for i in range(1, len(division_tree)):
                    # Flatten the list
                    lower_list_elements = self._flatten_list_to_set(division_tree[i])

                    # Rewrite unincluded communities
                    for sublist in division_tree[i - 1]:
                        if not set(sublist).issubset(lower_list_elements):
                            unique_to_sublist = set(sublist) - lower_list_elements
                            if unique_to_sublist:
                                division_tree[i].append(sublist)

                # Remove the last division if it repeats itself
                higher_list_elements = self._flatten_list_to_set(
                    division_tree[len(division_tree) - 2]
                )
                lower_list_elements = self._flatten_list_to_set(
                    division_tree[len(division_tree) - 1]
                )

                if higher_list_elements == lower_list_elements:
                    division_tree.pop(len(division_tree) - 1)

            if division_tree and return_modularities:
                division_modularities = []
                for division in division_tree:
                    division_modularity = nx.community.modularity(
                        G=self.sampler.G,
                        communities=division,
                        resolution=self.sampler.resolution,
                    )
                    division_modularities.append(division_modularity)

            elif return_modularities:
                division_modularities = nx.community.modularity(
                    G=self.sampler.G,
                    communities=result,
                    resolution=self.sampler.resolution,
                )

            if verbosity >= 1:
                print("Stopping community detection")
                print("Result: ")
                print(result)
                if division_tree:
                    print("Division tree")
                    for division in division_tree:
                        print(division)

            if division_tree and return_modularities:
                return result, division_tree, division_modularities
            if division_tree:
                return result, division_tree
            if return_modularities:
                return result, division_modularities
            else:
                return result
        elif max_depth < 1:
            print("Max depth value must be equal or greater than one!")
            return []

        return []

    def _hierarchical_search_recursion(
        self,
        verbosity: bool,
        max_depth: int,
        level: int,
        community: list | None = None,
        division_tree: list | None = None,
    ):
        if not community:
            community = [*range(self.sampler.G.number_of_nodes())]

        if len(community) == 1:
            return [community]

        if level == 1 and division_tree == []:
            division_tree.append([community])

        if verbosity >= 2:
            print("===========================================")
            print(
                "Calculations for graph with",
                len(community),
                "nodes, level of recursion:",
                level,
            )
            print("===========================================")

        if level != 1:
            self.sampler.__init__(
                self.sampler.G, self.sampler.resolution, community
            )

        sample = self.sampler.sample_qubo_to_dict()

        c0, c1 = self._split_dict_to_lists(sample, community)

        if verbosity >= 2:
            print("Base community:", community, sep="\n")
            print("Community division:", c0, c1, sep="\n")
            print("===========================================")

        if division_tree:
            if len(division_tree) < level + 1:
                division_tree.append([])

            if c0 and c1:
                division_tree[level].append(c0)
                division_tree[level].append(c1)
            else:
                division_tree[level].append(community)

        if level == max_depth:
            if c0 and c1:
                return [c0] + [c1]
            elif c0:
                return [c0]
            else:
                return [c1]

        else:
            if c0 and c1:
                return self._hierarchical_search_recursion(
                    verbosity,
                    max_depth,
                    level=level + 1,
                    community=c0,
                    division_tree=division_tree,
                ) + self._hierarchical_search_recursion(
                    verbosity,
                    max_depth,
                    level=level + 1,
                    community=c1,
                    division_tree=division_tree,
                )
            elif c0:
                return [c0]
            else:
                return [c1]

    def _split_dict_to_lists(self, dictionary, community):
        c0, c1 = [], []
        for i in community:
            value = dictionary.get(f"x{i}", None)
            if value == 0:
                c0.append(i)
            else:
                c1.append(i)
        return c0, c1

    def _flatten_list_to_set(self, list) -> set:
        result = set()
        for sublist in list:
            for item in sublist:
                result.add(item)

        return result