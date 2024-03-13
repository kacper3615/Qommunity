from samplers.regular.regular_sampler import RegularSampler

class CommunitySearcher:
    def __init__(self, sampler: RegularSampler) -> None:
        self.sampler = sampler

    def community_search(self, verbosity: int = 0, return_list: bool = True, community: list | None = None):
        if not community:
            community = [*range(self.sampler.G.number_of_nodes())]

        if verbosity >= 1:
            print("Starting community detection")
        if verbosity >= 2:
            print("===========================================")
            print("Calculations for graph with", len(community), "nodes in community")
            print("===========================================")

        if return_list:
            sample = self.sampler.sample_qubo_to_list()
        else:
            sample = self.sampler.sample_qubo_to_dict()

        if verbosity >= 2:
            print("Base community:", community, sep='\n')
            print("Community division:")
            if return_list:
                for subcommunity in sample:
                    print(subcommunity)
            else:
                subcommunities = self._communities_to_list(sample)
                for subcommunity in subcommunities:
                    print(subcommunity)
            print("===========================================")
        if verbosity >= 1:
                print("Stopping community detection")

        return sample


    def _communities_to_list(self, sample) -> list:
        communities = []
        for k in range(self.sampler.communities_number):
            subcommunity = []
            for i in sample:
                if sample[i] == k:
                    subcommunity.append(i)
            communities.append(subcommunity)

        return communities