import argparse
import json
import runner  # Import your core logic module

def main():
    """
    This is a wrapper script. Its sole purpose is to parse command-line
    arguments and delegate the actual work to the `runner.py` module.
    """
    # 1. --- Argument Parsing ---
    # Sets up the script to receive arguments from the shell script.
    parser = argparse.ArgumentParser(
        description="A wrapper to launch an experiment from the `runner` module."
    )
    parser.add_argument(
        "exp_id",
        type=str,
        help="The unique identifier for this experiment run."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the staged dataset on the node's local disk."
    )
    args = parser.parse_args()

    # 2. --- Delegation ---
    # Call the main function from the runner module and pass the parsed arguments.
    # The runner will handle all the core logic.
    print(f"--- [main.py] Delegating work to runner.py for experiment: {args.exp_id} ---")
    results = runner.execute_experiment(args)
    print(f"--- [main.py] Received results from runner.py ---")

    # 3. --- Output Handling ---
    # Take the dictionary returned by the runner and print it in a structured way.
    # This will be captured in your HTCondor .out file.
    # Using json.dumps makes the output clean and easy to parse later.
    print("\n" + "="*50)
    print("           FINAL EXPERIMENT RESULTS")
    print("="*50)
    print(json.dumps(results, indent=4))
    print("="*50 + "\n")


if __name__ == '__main__':
    # This standard Python construct ensures that main() is called only when
    # the script is executed directly (not when imported).
    main()
