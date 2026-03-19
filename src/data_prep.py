import os
import shutil
from cleanvision import Imagelab

def prepare_data():
    raw_train_path = "data/dataset/vehicle_dataset"
    clean_base_path = "data/vehicle_dataset_cleaned/train"
    
    # 1. Run Cleanvision Audit
    print("Starting data audit...")
    imagelab = Imagelab(data_path=raw_train_path)
    imagelab.find_issues()
    
    # 2. Filter for Clean Images
    issue_df = imagelab.issues
    issue_cols = [col for col in issue_df.columns if col.endswith('_issue')]
    # Keeps rows where the sum of all issue flags is 0
    clean_images_df = issue_df[issue_df[issue_cols].sum(axis=1) == 0]
    
    print(f"Audit Complete. Found {len(clean_images_df)} clean images.")

    # 3. Create Cleaned Directory
    if os.path.exists(clean_base_path):
        shutil.rmtree(clean_base_path)
    os.makedirs(clean_base_path, exist_ok=True)

    # 4. Move Files
    for index, _ in clean_images_df.iterrows():
        rel_path = os.path.relpath(index, raw_train_path)
        dest_path = os.path.join(clean_base_path, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(index, dest_path)
    
    print(f"Cleaned dataset ready at: {clean_base_path}")

if __name__ == "__main__":
    prepare_data()

