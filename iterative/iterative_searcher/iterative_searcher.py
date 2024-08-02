from Qommunity.searchers.community_searcher import CommunitySearcher
from Qommunity.searchers.hierarchical_community_searcher import (
    HierarchicalCommunitySearcher,
)
from iterative_searcher.iterative_searcher_hierarchical import (
    IterativeSearcherHierarchical,
)
from iterative_searcher.iterative_searcher_regular import IterativeSearcherRegular


class IterativeSearcher:
    def __new__(
        cls, searcher: HierarchicalCommunitySearcher | CommunitySearcher
    ) -> "IterativeSearcherHierarchical | IterativeSearcherRegular":
        if isinstance(searcher, HierarchicalCommunitySearcher):
            return IterativeSearcherHierarchical(searcher)
        elif isinstance(searcher, CommunitySearcher):
            return IterativeSearcherRegular(searcher)
