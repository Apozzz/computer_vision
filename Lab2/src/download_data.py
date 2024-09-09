import os
import shutil
from dotenv import load_dotenv
import roboflow

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
rf = roboflow.Roboflow(api_key=API_KEY)

def download_oxford_pets(dataset_type='by-breed', version=1, target_folder="data"):
    """
    Downloads the Oxford Pets dataset using Roboflow API.
    
    Args:
        dataset_type (str): Either 'by-breed' or 'by-species'.
        version (int): Dataset version (default is 1).
        target_folder (str): Path to save the dataset.

    Returns:
        dataset_path (str): The path to the downloaded dataset.
    """

    try:
        if os.path.exists(target_folder):
            print(f"Folder {target_folder} already exists. Removing the old folder...")
            shutil.rmtree(target_folder)

        project = rf.workspace("brad-dwyer").project("oxford-pets")
        print(f"Exporting the Oxford Pets dataset ({dataset_type})...")

        dataset_info = project.version(version).download(model_format="multiclass", location=target_folder)

        print(f"Dataset info: {dataset_info}")
        print(f"Dataset successfully downloaded to {target_folder}")

    except Exception as e:
        print(f"Error during download: {e}") 
    
    return os.path.join(target_folder, "train")