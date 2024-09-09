from download_data import download_oxford_pets

dataset_path = download_oxford_pets(dataset_type='by-breed')
print(f"Dataset downloaded to: {dataset_path}")