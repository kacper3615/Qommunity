import os
from saving_utils import save_response_with_solver, parse_response_json # for response
from saving_utils import serialize_qubo_problem, deserialize_qubo_problem
from problem_data import ProblemData, recover_problem_dwave_inspector


def recover_dwave_inspector_problems_from_runs(results, base_path):
    for iter, result_run in enumerate(results):
        samplesets_data = result_run.samplesets_data
        problem_ids = samplesets_data.problem_id
        problems_dwave_inspector = [recover_problem_dwave_inspector(pid) for pid in problem_ids]

        for dwave_problem in problems_dwave_inspector:
            problem = dwave_problem.problem
            created_at = dwave_problem.created_at
            response = dwave_problem.response
            solver = dwave_problem.solver

            problem_data = ProblemData(
                time_created=response.time_created,
                time_received=response.time_received,
                time_solved=response.time_solved,
                time_resolved=response.time_resolved,
                parse_time=response.parse_time,
                created_at=created_at,
                )

            problem_id = response.id
            serialize_qubo_problem(problem, f"{base_path}/iter_{iter}_problem_id={problem_id}_problem.json")
            save_response_with_solver(response, f"{base_path}/iter_{iter}_problem_id={problem_id}_response.json")
    
            problem_data.save_to_pickle(f"{base_path}/iter_{iter}_problem_id={problem_id}_timestamps.pkl")