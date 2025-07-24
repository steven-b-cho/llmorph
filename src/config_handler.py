from file_handler import load_json, file_exists
import config_data
import os
import copy
import glob

CONFIG_DIR = "src/config"
TEMPLATE_DIR = "src/config/templates"

def load_config_with_templates(main_config_path, template_dir):
    config = load_json(main_config_path)
    templates_by_category = load_templates_by_category(template_dir)
    replace_templates_with_config(config, templates_by_category)
    return config

def load_templates_by_category(template_dir):
    templates = {}
    for filename in os.listdir(template_dir):
        if filename.endswith('.json'):
            category = filename.replace('.json', '')
            templates[category] = load_json(os.path.join(template_dir, filename))
    return templates

def replace_templates_with_config(config, templates, category_suffix="_templates"):
    for _, value in list(config.items()):
        if isinstance(value, dict):
            if "$template" in value:
                category, template_name = value["$template"].split(":")
                template = templates.get(category + category_suffix, {}).get(template_name, {})
                value.pop("$template")
                value.update(template)
            else:
                replace_templates_with_config(value, templates)

def get_config_by_filename(filename, config_dir=CONFIG_DIR, template_dir=TEMPLATE_DIR):
    file_path = os.path.join(config_dir, filename + ".json",)
    config = load_config_with_templates(file_path, template_dir)
    return config

def get_raw_run_config_from_json(filename="run_config"):
    config = get_config_by_filename(filename)
    return config

def get_run_config_from_json(filename="run_config"):
    config = get_config_by_filename(filename)
    config = add_defaults_to_run_config(config)
    return config

def add_defaults_to_run_config(config):
    # uses defaults in config
    default_config = get_config_by_filename("run_config_default")

    # using defaults for missing keys
    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    # use first LLM in the list if not specified
    if not config["llm_for_transformation"]:
        config["llm_for_transformation"] = config["llm_list"][0] if config["llm_list"] else None

    # set dirs based on base dirs
    default_dir = config.get("dir_base_default")
    keys = ["checkpoints", "logs", "source_inputs", "source_outputs", "followup_inputs", "vsi", "vso", "vfi"]
    key_names = ["checkpoints", "results", "source_inputs", "source_outputs", "followup_inputs", "verify_source_inputs", "verify_source_outputs", "verify_followup_inputs"]
    for key, key_name in zip(keys, key_names):
        base_dir = config.get(f"dir_base_{key}", default_dir)
        config.setdefault(f"dir_{key}", os.path.join(base_dir, key_name))

    return config

def store_run_config(config):
    config_data.config_data = config


def merge_general_relations(original_data, replacement_rules=[]):
    data = copy.deepcopy(original_data)
    general_props = data.get("general", {})
    
    for task_name in [k for k in data.keys() if k != "general"]:
        specific_section = data[task_name]
        
        # Add missing IDs from general
        for gen_id, gen_props in general_props.items():
            if gen_id not in specific_section and ("exceptions" not in gen_props or task_name not in gen_props["exceptions"]):
                specific_section[gen_id] = copy.deepcopy({k: v for k, v in gen_props.items() if k != "exceptions"})

        # Merge properties
        for sp_id, sp_props in specific_section.items():
            if sp_id in general_props:
                gen_props = copy.deepcopy(general_props[sp_id])
                for gen_prop, gen_val in gen_props.items():
                    if gen_prop not in sp_props:
                        sp_props[gen_prop] = gen_val
            
            # Apply replacement rules if defined for the section
            if task_name in replacement_rules:
                for replace_key, replace_val in replacement_rules[task_name].items():
                    if replace_key == "replace":
                        # replace_val is a list of dicts {prop_name, from_class, to_value}
                        # hacky
                        for replace_data in replace_val:
                            p_name = replace_data["prop_name"]
                            if p_name in sp_props and sp_props[p_name]["class"] == replace_data["from_class"]:
                                sp_props[p_name] = replace_data["to_value"]
                    elif replace_key == "add_to":
                        # adds the kwarg to the prop if it does not exist in kwargs
                        for prop, add_val in replace_val.items():
                            if prop in sp_props and "kwargs" in sp_props[prop]:
                                for k, v in add_val.items():
                                    if k not in sp_props[prop]["kwargs"]:
                                        sp_props[prop]["kwargs"][k] = v
                            elif prop in sp_props:
                                sp_props[prop]["kwargs"] = add_val
    
    # remove 'exceptions' from all (including general)
    for task_name in data.keys():
        for _, props in data[task_name].items():
            if "exceptions" in props:
                props.pop("exceptions")
    
    return data

def remove_replacement_rules(original_tasks_data):
    rules_data = {}
    tasks_data = copy.deepcopy(original_tasks_data)
    for key, props in tasks_data.items():
        if "replacement_rules" in props:
            rules = props.pop("replacement_rules")
            rules_data[key] = rules
    return rules_data, tasks_data

def get_tasks_relations():
    raw_tasks = get_config_by_filename("list_tasks")
    raw_relations = get_config_by_filename("list_relations")
    replacement_rules, tasks_data = remove_replacement_rules(raw_tasks)
    relations_data = merge_general_relations(raw_relations, replacement_rules)
    return tasks_data, relations_data


def get_checkpoint(checkpoint_dir, checkpoint_filepath = None) -> dict | None:
    if checkpoint_filepath is None:
        return get_latest_checkpoint(checkpoint_dir)
    if not file_exists(checkpoint_filepath):
        print(f"Warning: Checkpoint {checkpoint_filepath} does not exist")
        return None
    print(f"Checkpoint found at {checkpoint_filepath}.")
    return load_json(checkpoint_filepath)

def get_latest_checkpoint(checkpoint_dir) -> dict | None:
    files = glob.glob(os.path.join(checkpoint_dir, '*'))
    if not files:
        print(f"Warning: No checkpoint found. Continuing without checkpoint...")
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"Checkpoint found at {latest_file}.")
    return load_json(latest_file)