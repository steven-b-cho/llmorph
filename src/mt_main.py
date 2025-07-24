import os
from config_handler import get_raw_run_config_from_json, add_defaults_to_run_config, store_run_config, get_tasks_relations, get_checkpoint
from mt_config_run_handler import do_run_tests

# main entry point
def run_using_config(run_config: dict):
    # current_working_directory = os.getcwd()
    # print("Current working directory:", current_working_directory)

    run_config = add_defaults_to_run_config(run_config)
    store_run_config(run_config)

    init_folders(run_config)

    tasks_list, relations_list = get_tasks_relations()
    _, checkpoint = get_config_and_cp(run_config)

    print(run_config)
    
    llm_list = run_config["llm_list"]
    for llm_name in llm_list:
        if checkpoint is not None and checkpoint["llm_name"] and checkpoint["llm_name"] != llm_name:
            continue # skip if checkpoint is not for this llm
        print()
        print(f"Running tests for {llm_name}...")
        do_run_tests(llm_name, run_config, tasks_list, relations_list, checkpoint)
        checkpoint = None # only use checkpoint for the first test (if any)

def get_config_and_cp(run_config: dict):
    checkpoint = None
    if run_config["continue_from_checkpoint"]:
        checkpoint = get_checkpoint(run_config["dir_checkpoints"])
        if checkpoint:
            run_config = checkpoint["run_config"]
    return run_config, checkpoint


# if folders do not exist, create them
def init_folders(config: dict):
    folders = [
        config.get("dir_checkpoints"),
        config.get("dir_logs"),
        config.get("dir_source_inputs"),
        config.get("dir_source_outputs"),
        config.get("dir_followup_inputs"),
        config.get("dir_vsi"),
        config.get("dir_vso"),
        config.get("dir_vfi"),
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def save_tasks_relations():
    from file_handler import save_json
    tasks, relations = get_tasks_relations()
    save_json(tasks, "analysis/tasks.json")
    save_json(relations, "analysis/relations.json")



def main():
    run_config = get_raw_run_config_from_json()
    run_using_config(run_config)

if __name__ == '__main__':
    main()
    # save_tasks_relations()