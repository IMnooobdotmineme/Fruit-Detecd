import kagglehub

print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("swoyam2609/fresh-and-stale-classification")
print("Dataset downloaded successfully to:")
print(path)