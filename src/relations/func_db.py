from .func_base import FuncDB
from file_handler import load_json
from datasets import load_dataset
import random
import re

DATASET_SIZE = 1000 # default size for these test datasets

class DBJSON(FuncDB):
    # json file is a simple list of inputs
    def __init__(self, file_path):
        self.file_path = file_path
    
    def get_dataset(self):
        return load_json(self.file_path)

class DBHF(FuncDB):
    # Huggingface dataset
    def __init__(self, dataset_name, size=None, seed=42):
        self.dataset_name = dataset_name
        self.size = size
        self.seed = seed

    def load_limited_dataset(self, dataset_name, num_samples=100):
        dataset = load_dataset(dataset_name, streaming=True)

        limited_dataset = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            limited_dataset.append(sample)
        
        return limited_dataset

    def get_dataset(self):
        random.seed(self.seed)
        dataset = load_dataset(self.dataset_name, streaming=True)

        all_values = []
        for split in dataset.keys():
            all_values.extend([item for item in dataset[split]])
        
        if self.size is None:
            return all_values
        
        sampled_values = random.sample(all_values, min(self.size, len(all_values)))
        return sampled_values
    
class DBSquad2(DBHF):
    def __init__(self):
        super().__init__("squad_v2", DATASET_SIZE)

    def get_dataset(self):
        dataset = super().get_dataset()
        dataset_formatted = [[d["context"], d["question"]] for d in dataset]
        return dataset_formatted
    
    def get_dataset_with_labels(self):
        dataset = super().get_dataset()
        dataset_formatted = []
        for d in dataset:
            if len(d["answers"]["text"]) > 0:
                text = d["answers"]["text"][0]
            else:
                text = "unknown"
            dataset_formatted.append([d["context"], d["question"], text])
        return dataset_formatted

class DBSNLI(DBHF):
    def __init__(self):
        super().__init__("snli", DATASET_SIZE)

    def get_dataset(self):
        dataset = super().get_dataset()
        dataset_formatted = [[d["premise"], d["hypothesis"]] for d in dataset]
        return dataset_formatted
    
    def get_dataset_with_labels(self):
        dataset = super().get_dataset()
        dataset_formatted = [[d["premise"], d["hypothesis"], d["label"]] for d in dataset]
        return dataset_formatted


class DBSST2(DBHF):
    def __init__(self):
        super().__init__("sst2", DATASET_SIZE)

    def get_dataset(self):
        dataset = super().get_dataset()
        dataset_formatted = [d["sentence"] for d in dataset]
        return dataset_formatted
    
    def get_dataset_with_labels(self):
        dataset = super().get_dataset()
        dataset_formatted = [[d["sentence"], d["label"]] for d in dataset]
        return dataset_formatted



class DBDocRED(DBHF):
    def __init__(self):
        super().__init__("docred", DATASET_SIZE)

    def list_to_string(self, list_of_sentences):
        return ' '.join([' '.join(sentence) for sentence in list_of_sentences])
    
    def list_to_string_cleaned(self, list_of_sentences):
        text = self.list_to_string(list_of_sentences)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        return text
    
    def extract_names(self, data):
        names = [item["name"] for sublist in data for item in sublist]
        return names
    
    def extract_names_first(self, data):
        names = [sublist[0]["name"] for sublist in data]
        return names

    def get_random_pair(self, lst):
        return random.sample(lst, 2)
    
    # In DocRED dataset, each row contains:
    # - a sentence
    # - the list of possible entities in that sentence
    # This chooses a random pair to test
    def format_data_single(self, data, seed=42):
        random.seed(seed)
        output_data = []
        for d in data:
            text = self.list_to_string_cleaned(d["sents"])
            entities = self.extract_names_first(d["vertexSet"])
            entity_pair = self.get_random_pair(entities)
            output_data.append([text] + entity_pair)
        return output_data

    def get_dataset(self):
        dataset = super().get_dataset()
        dataset_formatted = self.format_data_single(dataset)
        return dataset_formatted


class DBReDocRED(DBHF):
    def __init__(self):
        super().__init__("re_docred", DATASET_SIZE)

    def list_to_string(self, list_of_sentences):
        return ' '.join([' '.join(sentence) for sentence in list_of_sentences])
    
    def list_to_string_cleaned(self, list_of_sentences):
        text = self.list_to_string(list_of_sentences)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        return text
    
    def extract_names_random(self, data):
        names = [random.choice(sublist)["name"] for sublist in data]
        return names

    def get_random_pair(self, lst):
        return random.sample(lst, 2)
    
    def format_data_single(self, data, seed=42):
        random.seed(seed)
        output_data = []
        for d in data:
            text = self.list_to_string_cleaned(d["sents"])
            entity_names = self.extract_names_random(d["vertexSet"])
            relation = random.choice(d["labels"])
            head = entity_names[relation["h"]]
            tail = entity_names[relation["t"]]
            output_data.append([text, head, tail])
        return output_data
    
    def get_dataset_from_files(self, file_names):
        all_data = []
        for f in file_names:
            data = load_json(f)
            all_data.extend(data)
        return all_data
    
    def get_sampled_values(self, data):
        random.seed(self.seed)
        return random.sample(data, min(self.size, len(data)))

    def get_dataset(self):
        file_names = [
            "./datasets/Re-DocRED-main/data/dev_revised.json",
            "./datasets/Re-DocRED-main/data/train_revised.json",
            "./datasets/Re-DocRED-main/data/test_revised.json",
        ]
        dataset = self.get_dataset_from_files(file_names)
        viable_data = [d for d in dataset if len(d["labels"]) > 0] #remove anything with no relations
        sampled_dataset = self.get_sampled_values(viable_data)
        dataset_formatted = self.format_data_single(sampled_dataset)
        return dataset_formatted

    def format_data_single_with_label(self, data, seed=42):
        random.seed(seed)
        output_data = []
        rel_mapping = load_json("./datasets/Re-DocRED-main/relation_mapping.json")
        for d in data:
            text = self.list_to_string_cleaned(d["sents"])
            entity_names = self.extract_names_random(d["vertexSet"])
            relation = random.choice(d["labels"])
            head = entity_names[relation["h"]]
            tail = entity_names[relation["t"]]
            rel_name = rel_mapping.get(relation["r"])
            output_data.append([text, head, tail, rel_name])
        return output_data

    def get_dataset_with_labels(self):
        file_names = [
            "./datasets/Re-DocRED-main/data/dev_revised.json",
            "./datasets/Re-DocRED-main/data/train_revised.json",
            "./datasets/Re-DocRED-main/data/test_revised.json",
        ]
        dataset = self.get_dataset_from_files(file_names)
        viable_data = [d for d in dataset if len(d["labels"]) > 0] # remove anything with no relations
        sampled_dataset = self.get_sampled_values(viable_data)
        dataset_formatted = self.format_data_single_with_label(sampled_dataset)
        return dataset_formatted


