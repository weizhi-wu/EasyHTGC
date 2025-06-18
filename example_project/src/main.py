import argparse
import json
import csv
import runner

def main():
    """
    Wrapper script to parse args, delegate work, and handle I/O.
    Now correctly handles a list of dictionaries to write a multi-row CSV.
    """
    # 1. --- Argument Parsing (Unchanged) ---
    parser = argparse.ArgumentParser(description="Wrapper to launch an experiment.")
    parser.add_argument("exp_id", type=str, help="Unique experiment identifier.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to staged dataset.")
    args = parser.parse_args()

    # 2. --- Delegation (Unchanged) ---
    print(f"--- [main.py] Delegating work to runner.py for experiment: {args.exp_id} ---")
    # `results_list` is now expected to be a list of dictionaries
    results_list = runner.execute_experiment(args)
    print(f"--- [main.py] Received results list from runner.py ---")

    # 3. --- Write Results to CSV File (Updated Logic) ---
    csv_filename = f"results_{args.exp_id}.csv"
    print(f"--- [main.py] Writing time-series results to {csv_filename} ---")
    
    # Check if the results list is not empty
    if results_list:
        try:
            # Get the headers from the keys of the FIRST dictionary in the list
            headers = results_list[0].keys()
            
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                # Use writerows() to write all dictionaries in the list
                writer.writerows(results_list)
                
            print(f"--- [main.py] Successfully wrote {len(results_list)} rows to {csv_filename} ---")
        except Exception as e:
            print(f"--- [main.py] ERROR: Failed to write CSV file. Reason: {e} ---")
    else:
        print("--- [main.py] WARNING: Results list from runner was empty. No CSV file written. ---")

    # 4. --- Output Final Summary to .out log (Good Practice) ---
    # It's still useful to print a final summary to the .out file for a quick look.
    final_summary = results_list[-1] if results_list else {"status": "No results"}
    print("\n" + "="*50)
    print("           FINAL EPOCH SUMMARY (JSON)")
    print("="*50)
    print(json.dumps(final_summary, indent=4))
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
