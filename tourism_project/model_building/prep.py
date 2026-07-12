# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/naveenaggarwal1989/tourism-package-data/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier and any unnamed columns
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# Map 'Fe Male' to 'Female' in 'Gender' as it looks like a typo only
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# Map 'Unmarried' to 'Single' in 'MaritalStatus' as both of them are same
if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="naveenaggarwal1989/tourism-package-data",
        repo_type="dataset",
    )
