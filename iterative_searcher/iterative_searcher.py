from Qommunity.samplers.hierarchical.hierarchical_sampler import HierarchicalSampler
from Qommunity.samplers.regular.regular_sampler import RegularSampler
from iterative_searcher.iterative_searcher_hierarchical import (
    IterativeSearcherHierarchical,
)
from iterative_searcher.iterative_searcher_regular import IterativeSearcherRegular


class IterativeSearcher:
    def __new__(
        cls, sampler: HierarchicalSampler | RegularSampler
    ) -> "IterativeSearcherHierarchical | IterativeSearcherRegular":
        if isinstance(sampler, HierarchicalSampler):
            return IterativeSearcherHierarchical(sampler)
        elif isinstance(sampler, RegularSampler):
            return IterativeSearcherRegular(sampler)
