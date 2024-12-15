from Qommunity.samplers.hierarchical.hierarchical_sampler import (
    HierarchicalSampler,
)
from Qommunity.samplers.regular.regular_sampler import RegularSampler
from Qommunity.iterative_searcher.iterative_hierarchical_searcher import (
    IterativeHierarchicalSearcher,
)
from Qommunity.iterative_searcher.iterative_regular_searcher import (
    IterativeRegularSearcher,
)


class IterativeSearcher:
    def __new__(
        cls, sampler: HierarchicalSampler | RegularSampler
    ) -> "IterativeHierarchicalSearcher | IterativeRegularSearcher":
        if isinstance(sampler, HierarchicalSampler):
            return IterativeHierarchicalSearcher(sampler)
        elif isinstance(sampler, RegularSampler):
            return IterativeRegularSearcher(sampler)
