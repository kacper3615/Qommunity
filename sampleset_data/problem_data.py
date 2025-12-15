from dataclasses import dataclass, field
from datetime import datetime, timezone
from datetime import timedelta
from typing import Callable, Optional
from dwave.inspector.storage import get_problem, problemdata_bag, problemdata



@dataclass
class ProblemData:
    time_created: datetime
    time_received: datetime
    time_solved: datetime
    time_resolved: datetime
    parse_time: float
    created_at: float | None = None

    service_time_determination_method: Optional[Callable[['ProblemData'], timedelta]] = None
    
    _service_time: timedelta = field(init=False)
    
    def __post_init__(self) -> None:
        if self.service_time_determination_method:
            self._service_time = self.service_time_determination_method(self)
        else:
            self._service_time = self.__calulcate_service_time_default()

    def __calulcate_service_time_default(self) -> datetime:
        return self.time_solved - self.time_received
    
    @property
    def service_time(self) -> timedelta:
        return self._service_time
    
    def apply_other_service_time_computing_method(self, method_handler: Callable[['ProblemData'], timedelta]) -> None:
        self._service_time = method_handler(self)

    
    def save_to_pickle(self, filename: str) -> None:
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


def recover_problem_dwave_inspector(problem_id: str) -> None:
    problem_dwave_inspector = get_problem(problem_id)
    return problem_dwave_inspector






