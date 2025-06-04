import cobra
from cobra.flux_analysis import flux_variability_analysis, geometric_fba, pfba


def run_simulation(model, method="fba", **kwargs):
    """
    Run a metabolic simulation on a COBRApy model using the specified method.

    Parameters:
      model : cobra.Model
          The metabolic model.
      method : str
          Simulation method to use. Supported options:
            - "fba": Standard Flux Balance Analysis (model.optimize())
            - "pfba": Parsimonious FBA (cobra.flux_analysis.pfba(model, **kwargs))
            - "geometric": Geometric FBA (cobra.flux_analysis.geometric_fba(model, **kwargs))
            - "slim": Slim optimization (model.slim_optimize(**kwargs))
                     [Note: returns only the objective value]
            - "fva": Flux Variability Analysis (cobra.flux_analysis.flux_variability_analysis(model, **kwargs))
      **kwargs:
          Additional arguments to pass to the simulation function.

    Returns:
      The simulation result. For "fba", "pfba", and "geometric", a cobra.Solution object is returned.
      For "slim", a float (objective value) is returned.
      For "fva", a pandas DataFrame is returned.
    """
    method = method.lower()
    if method == "pfba":
        solution = pfba(model, **kwargs)
    elif method == "geometric":
        solution = geometric_fba(model, **kwargs)
    elif method == "slim":
        solution = model.slim_optimize(**kwargs)
    elif method == "fva":
        solution = flux_variability_analysis(model, **kwargs)
    elif method == "fba":
        solution = model.optimize(**kwargs)
    else:
        raise ValueError(
            f"Unsupported simulation method '{method}'. Supported methods: 'fba', 'pfba', 'geometric', 'slim', 'fva'."
        )

    return solution
