# --- START OF FILE download_dataset.py ---

import torchvision
import tarfile
import os
import shutil
import argparse

# This configuration makes it easy to adapt the script for other datasets
CONFIG = {
    "dataset_name": "CIFAR-10",
    "torchvision_dataset": torchvision.datasets.CIFAR10,
    "archive_name": "dataset.tar.gz",
    "internal_folder_name": "cifar10_dataset" # CRITICAL: This must match the path in runner.py
}

def main(args):
    """
    Downloads a dataset using torchvision and compresses it into a .tar.gz archive
    suitable for the HTCondor workflow.
    """
    temp_download_dir = f"./temp_{CONFIG['internal_folder_name']}_download"
    
    if os.path.exists(CONFIG['archive_name']) and not args.replace:
        print(f"--- INFO: '{CONFIG['archive_name']}' already exists. Use --replace to overwrite. ---")
        return

    print(f"--- Starting download for {CONFIG['dataset_name']} ---")
    
    try:
        # 1. Download the dataset to a temporary directory
        if os.path.exists(temp_download_dir):
            shutil.rmtree(temp_download_dir)
        os.makedirs(temp_download_dir)
        
        print(f"Downloading to temporary directory: {temp_download_dir}")
        CONFIG['torchvision_dataset'](root=temp_download_dir, train=True, download=True)
        CONFIG['torchvision_dataset'](root=temp_download_dir, train=False, download=True)
        print("--- Download complete. ---")

        # 2. Create the .tar.gz archive
        print(f"--- Creating archive: {CONFIG['archive_name']} ---")
        with tarfile.open(CONFIG['archive_name'], "w:gz") as tar:
            # The `arcname` parameter is crucial. It sets the name of the top-level
            # folder inside the archive, ensuring a clean and predictable path.
            source_dir = os.path.join(temp_download_dir, CONFIG['dataset_name'].lower().replace('-', ''))
            tar.add(source_dir, arcname=CONFIG['internal_folder_name'])
        
        print(f"--- Successfully created '{CONFIG['archive_name']}'. Project is ready. ---")

    except Exception as e:
        print(f"--- ERROR: An error occurred during dataset preparation: {e} ---")
    
    finally:
        # 3. Clean up the temporary download directory
        if os.path.exists(temp_download_dir):
            print(f"--- Cleaning up temporary directory: {temp_download_dir} ---")
            shutil.rmtree(temp_download_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and prepare a dataset for the HTCondor workflow.")
    parser.add_argument("--replace", action="store_true", help="Overwrite the existing dataset.tar.gz if it exists.")
    args = parser.parse_args()
    main(args)

# --- END OF FILE download_dataset.py ---
