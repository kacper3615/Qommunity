from dwave.inspector.storage import get_problem, problemdata_bag, problemdata


def recover_dwave_problem(problem_id: str) -> None:
    problem = get_problem(problem_id)
    


save_response_with_solver(problem.response, "response_full_with_solver.json")
response_obj = parse_response_json("response_full_with_solver.json")


serialize_qubo_problem(problem.problem, "problem_qubo.json")
problem_reconstructed = deserialize_qubo_problem("problem_qubo.json")

from dwave.cloud.client import Client
from dwave.cloud.solver import StructuredSolver


solver_reconstructed = StructuredSolver(client=Client, data=response_obj.solver["data"])
solver_reconstructed


# from dwave.inspector.storage import ProblemDataTimestamped, ProblemData


# pr_rec = ProblemDataTimestamped(problem=problem_reconstructed, solver=StructuredSolver(client=Client, data=response_obj.solver["data"]), response=response_obj)

import json

with open("solver_full.json", "r") as f:
    solver_dict = json.load(f)

from dwave.cloud.solver import SolverConfiguration

# Convert dict back to SolverConfiguration
solver_config = SolverConfiguration.model_validate(solver_dict)

from dwave.cloud.solver import StructuredSolver

solver = StructuredSolver(client=Client, data=solver_config)
problem.solver.max_num_reads()