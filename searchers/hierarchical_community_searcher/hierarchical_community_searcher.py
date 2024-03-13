from samplers.hierarchical.hierarchical_sampler import HierarchicalSampler


class HierarchicalCommunitySearcher:
    def __init__(self, sampler: HierarchicalSampler) -> None:
        self.sampler = sampler

    def single_community_search(self, verbosity: int = 0, community: list | None = None) -> list:
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
            print("Base community:", community, sep='\n')
            print("Community division:", c0, c1, sep='\n')
            print("===========================================\n")
        if verbosity >= 1:
                print("Stopping community detection")

        if c0 and c1:
            return [c0] + [c1]
        elif c0:
            return [c0]
        else:
            return [c1]

    def hierarchical_community_search(self, verbosity: int = 0, max_depth: int | None = None) -> list:
        if verbosity >= 1:
            print("Starting community detection")

        if max_depth == None:
            result = self._hierarchical_search_recursion(verbosity=verbosity, level=1, max_depth=max_depth)

            if verbosity >= 1:
                print("Stopping community detection")
                print("Result: ")
                print(result)
            
            return result
        elif max_depth < 1:
            print("Max depth value must be equal or greater than one!")
            return []
        

    def _hierarchical_search_recursion(self, verbosity: bool, max_depth: int, level: int, community: list | None = None):
        if not community:
            community = [*range(self.sampler.G.number_of_nodes())]

        if len(community) == 1:
            return community
        
        if verbosity >= 2:
            print("===========================================")
            print("Calculations for graph with", len(community), "nodes, level of recursion:", level)
            print("===========================================")

        if level != 1:
            self.sampler.__init__(self.sampler.G, self.sampler.time, self.sampler.resolution, community)

        sample = self.sampler.sample_qubo_to_dict()

        c0, c1 = self._split_dict_to_lists(sample, community)

        if verbosity >= 2:
            print("Base community:", community, sep='\n')
            print("Community division:", c0, c1, sep='\n')
            print("===========================================")
        
        if level == max_depth:
            if c0 and c1:
                return [c0] + [c1]
            elif c0:
                return [c0]
            else:
                return [c1]
            
        else: 
            if c0 and c1:
                return (self._hierarchical_search_recursion(verbosity, max_depth, level=level+1, community=c0) + 
                        self._hierarchical_search_recursion(verbosity, max_depth, level=level+1, community=c1))
            elif c0:
                return [c0]
            else:
                return [c1]
            
    def _split_dict_to_lists(self, dictionary, community):
        c0, c1 = [], []
        for i in community:
            value = dictionary.get(f"x{i}" , None)
            if value == 0:
                c0.append(i)
            else:
                c1.append(i)
        return c0, c1
    
