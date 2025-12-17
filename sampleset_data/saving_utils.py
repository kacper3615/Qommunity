import json
import datetime

def make_serializable_solver(solver):
    """
    Convert solver object to JSON-serializable dict using our previous logic.
    """
    data = {}
    
    for name in dir(solver):
        if name.startswith("_"):
            continue
        try:
            value = getattr(solver, name)
        except Exception:
            data[name] = "<unreadable>"
            continue
        if callable(value):
            continue
        # use to_serializable if available
        if hasattr(value, "to_serializable") and callable(value.to_serializable):
            try:
                data[name] = value.to_serializable()
            except Exception:
                data[name] = "<to_serializable failed>"
        elif isinstance(value, (datetime.datetime, datetime.date)):
            data[name] = value.isoformat()
        else:
            try:
                json.dumps(value)
                data[name] = value
            except TypeError:
                data[name] = str(value)
    
    return data

def save_response_with_solver(response, filename):
    """
    Save all public attributes of problem.response to JSON, using solver serialization
    for the 'solver' attribute.
    """
    data = {}
    
    for name in dir(response):
        if name.startswith("_"):
            continue
        
        try:
            value = getattr(response, name)
        except Exception:
            data[name] = "<unreadable>"
            continue
        
        if callable(value):
            continue
        
        # special case: solver
        if name == "solver":
            data[name] = make_serializable_solver(value)
        # use to_serializable if available (e.g., SampleSet)
        elif hasattr(value, "to_serializable") and callable(value.to_serializable):
            try:
                data[name] = value.to_serializable()
            except Exception:
                data[name] = "<to_serializable failed>"
        # datetime -> ISO
        elif isinstance(value, (datetime.datetime, datetime.date)):
            data[name] = value.isoformat()
        else:
            try:
                json.dumps(value)
                data[name] = value
            except TypeError:
                data[name] = str(value)
    
    # save to JSON
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving response to {filename}: {e}")


import json
import datetime
from types import SimpleNamespace
import dimod

def parse_response_json(filename):
    """
    Load a response JSON saved with save_response_with_solver
    and reconstruct a simple object with same public fields.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    
    obj = SimpleNamespace()
    
    for name, value in data.items():
        # handle datetime
        if isinstance(value, str):
            try:
                # próba parsowania ISO datetime
                dt = datetime.datetime.fromisoformat(value)
                setattr(obj, name, dt)
                continue
            except ValueError:
                pass
        
        # handle SampleSet (jeśli to dict z to_serializable)
        if name == "sampleset" and isinstance(value, dict):
            try:
                ss = dimod.SampleSet.from_serializable(value)
                setattr(obj, name, ss)
                continue
            except Exception:
                pass
        
        # handle solver (możesz zostawić jako dict lub odtworzyć StructuredSolver, jeśli chcesz)
        if name == "solver" and isinstance(value, dict):
            setattr(obj, name, value)  # zostawiamy jako dict
        
        # wszystko inne
        setattr(obj, name, value)
    
    return obj


import json
import numpy as np

def serialize_qubo_problem(problem_dict, filename):
    """
    Serialize a QUBO problem dict to JSON safely:
    - converts numpy.float64 to float
    - converts tuple keys in 'quadratic' to strings
    """
    serializable = problem_dict.copy()
    
    # linear: convert np.float64 to float
    serializable["linear"] = {k: float(v) for k, v in problem_dict["linear"].items()}
    
    # quadratic: convert tuple keys to "i,j" strings, values to float
    serializable["quadratic"] = {f"{i},{j}": float(v) 
                                 for (i,j), v in problem_dict["quadratic"].items()}
    
    # offset
    if "offset" in serializable:
        serializable["offset"] = float(serializable["offset"])
    
    # params: leave as is
    # label, undirected_biases: leave as is
    
    try:
        with open(filename, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"Error saving QUBO problem to {filename}: {e}")


import json
import numpy as np

def deserialize_qubo_problem(filename):
    """
    Load a QUBO problem from JSON and reconstruct original format:
    - tuple keys in 'quadratic'
    - numpy.float64 values for linear, quadratic, offset
    """
    with open(filename, "r") as f:
        data = json.load(f)
    
    # linear
    linear = {int(k): np.float64(v) for k, v in data["linear"].items()}
    
    # quadratic
    quadratic = {}
    for k, v in data["quadratic"].items():
        i, j = map(int, k.split(","))
        quadratic[(i,j)] = np.float64(v)
    
    # offset
    offset = np.float64(data.get("offset", 0.0))
    
    # params, label, undirected_biases
    params = data.get("params", {})
    label = data.get("label")
    undirected_biases = data.get("undirected_biases", True)
    
    problem_reconstructed = {
        "type_": "qubo",
        "linear": linear,
        "quadratic": quadratic,
        "offset": offset,
        "params": params,
        "label": label,
        "undirected_biases": undirected_biases
    }
    
    return problem_reconstructed