# --- START OF FILE src/utils.py ---

def parse_hyperparameters(exp_id: str) -> dict:
    """
    Parses a string-based experiment ID into a dictionary of hyperparameters.

    This allows for defining experiments with varying parameters directly from the
    SimpleHTGC UI's experiment list.

    Example:
        exp_id = "resnet20_lr-0.05_mu-0.01_sp-0.2"
        parses to: {'lr': 0.05, 'mu': 0.01, 'sparsity_ratio': 0.2, ...}

    Args:
        exp_id (str): The experiment ID string, e.g., "run1_lr-0.01_sp-0.1".

    Returns:
        dict: A dictionary containing hyperparameters for the DeepZero optimizer.
    """
    # Define the default hyperparameters for the DeepZero optimizer
    params = {
        'lr': 0.01,
        'mu': 0.005,
        'sparsity_ratio': 0.1,
        'p_rge': 128,
        'k_sparse': 1,
        'momentum': 0.9
    }

    # Split the experiment ID into parts based on the underscore delimiter
    parts = exp_id.split('_')

    # Iterate over the parts to find key-value pairs
    for part in parts:
        if '-' in part:
            try:
                key, value = part.split('-', 1)
                # Map shorthand keys to the full parameter names
                key_map = {
                    'lr': 'lr',
                    'mu': 'mu',
                    'sp': 'sparsity_ratio',
                    'p': 'p_rge',
                    'k': 'k_sparse',
                    'mom': 'momentum'
                }
                
                if key in key_map:
                    param_name = key_map[key]
                    # Convert value to the correct type (float or int)
                    if '.' in value:
                        params[param_name] = float(value)
                    else:
                        params[param_name] = int(value)
            except ValueError:
                # Ignore parts that are not valid key-value pairs (e.g., "resnet20")
                print(f"[utils.py] INFO: Ignoring non-hyperparameter part '{part}' in exp_id.")
                continue

    print(f"[utils.py] Parsed hyperparameters for exp_id '{exp_id}': {params}")
    return params

# --- END OF FILE src/utils.py ---
