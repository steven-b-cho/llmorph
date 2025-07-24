from relations.func_db import DBSNLI, DBSquad2, DBSST2, DBReDocRED
from file_handler import save_json
import os

# downloads test data
def main():
    dir = "data/source-inputs-with-labels"
    data_getters = [
        DBSNLI(),
        DBSquad2(),
        DBSST2(),
        DBReDocRED(),
    ]
    dataset_names = [
        "snli", 
        "squad2", 
        "sst2", 
        "re-docred"
    ]
    for name, dg in zip(dataset_names, data_getters):
        print(f"Getting {name}...")
        data = dg.get_dataset_with_labels()
        save_json(data, os.path.join(dir, f"{name}.json"))


if __name__ == "__main__":
    main()
