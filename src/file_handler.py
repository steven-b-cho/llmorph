import json
import datetime
import os
import csv

def save_json(output, file_path):
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    json_data = json.dumps(output, indent=4, ensure_ascii=False)
    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_data)

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data

def get_timestamp():
    from config_data import config_data # Needs to be imported this late as the file itself is loaded before the config, which causes config to be empty

    format = config_data.get("use_d-m-y_time", True)
    current_time = datetime.datetime.now()

    if(format):
        timestamp = current_time.strftime("%d-%m-%Y_%H-%M-%S")
    else:
        timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

def file_exists(filename):
    return os.path.exists(filename)

def load_csv(csv_filepath):
    with open(csv_filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data # returns List[Dict[str, str]]

def save_csv(data, file_path, fieldnames=None):
    if not fieldnames:
        fieldnames = data[0].keys()
    try:
        with open(file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except IOError:
        print(f"Error writing to file: {file_path}")