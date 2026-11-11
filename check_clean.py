import yaml
from src.data import load_data
from src.preprocessing import clean_dataset

config = yaml.safe_load(open("config_adult.yaml"))
df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])

for col in ["workclass", "occupation", "native_country"]:
    print(col, "| missing:", df[col].isna().sum(), "| '?' que sobram:", (df[col] == "?").sum())