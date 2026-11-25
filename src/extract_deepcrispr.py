import tarfile
import os

print("Extracting DeepCRISPR classification dataset...")

# Path to the DeepCRISPR archive (download DeepCRISPR separately; see README)
tar_path = os.path.join("DeepCRISPR-master", "paper_data-classification.tar.gz")

# Extract to data folder
os.makedirs("data", exist_ok=True)

with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall("data/")
    print(" Extraction complete!")

# List what we got
print("\n Files in data folder:")
for root, dirs, files in os.walk("data/"):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / (1024*1024)  # MB
        print(f"  {file} ({size:.2f} MB)")