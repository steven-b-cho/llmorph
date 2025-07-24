import importlib
from mt_run import run_test
from relations.mt_mr_main import Relation

def do_run_tests(llm_name: str, run_config: dict, tasks_list: dict, relations_list: dict, checkpoint: dict | None = None) -> None:
    tests_to_run = get_tests_to_run(run_config, tasks_list, relations_list)
    if checkpoint:
        tests_to_run = get_cp_tests_to_run(tests_to_run, checkpoint)

    for test_to_run in tests_to_run:
        task_name, relation_name = test_to_run
        run_that_test(llm_name, task_name, relation_name, tasks_list, relations_list, run_config, checkpoint)
        checkpoint = None # only use checkpoint for the first test (if any)


def get_cp_tests_to_run(tests_to_run: list[tuple[str,str]], checkpoint: dict) -> list[tuple[str,str]]:
    checkpoint_test = (checkpoint["task_name"], checkpoint["relation_name"])
    
    if checkpoint["checkpoint_type"] in ["source"]:
        checkpoint_tests_to_run = get_cp_source_sublist(tests_to_run, checkpoint_test)
    else:
        checkpoint_tests_to_run = get_cp_sublist(tests_to_run, checkpoint_test)
    
    if checkpoint_tests_to_run:
        return checkpoint_tests_to_run
    else:
        print("Checkpoint test not found. Running tests as specified...")
        return tests_to_run

def get_cp_sublist(lst, target):
    for i, elm in enumerate(lst):
        if elm == target:
            return lst[i:]
    return []

def get_cp_source_sublist(lst, target):
    for i, (first_elm, _) in enumerate(lst):
        if first_elm == target[0]:
            return lst[i:]
    return []


# Returns a list of [(task_name, relation_names), ...]
def get_tests_to_run(run_config: dict, tasks_list: dict, relations_list: dict) -> list[tuple[str,str]]:
    tasks_to_run = get_tasks_to_run(run_config, tasks_list, relations_list)
    tests_to_run = []

    for task_name, relation_names in tasks_to_run.items():
        for relation_name in relation_names:
            test = (task_name, relation_name)
            tests_to_run.append(test)
    return tests_to_run

def get_tasks_to_run(run_config: dict, tasks_list: dict, relations_list: dict):
    tasks_to_run = get_tasks_from_config(run_config, tasks_list)
    populated_tasks_to_run = populate_relations(tasks_to_run, tasks_list, relations_list)
    return populated_tasks_to_run

def get_tasks_from_config(run_config: dict, tasks_list: dict) -> dict[str, list[str]]:
    if run_config["run_all"]:
        return {task_name: [] for task_name in tasks_list.keys()}
    return run_config["tasks"]

def populate_relations(tasks_to_run: dict[str, list[str]], tasks_list: dict, relations_list: dict) -> dict[str, list[str]]:
    populated_tasks_to_run = {}
    for task_name, relation_names in tasks_to_run.items():
        if task_name in tasks_list:
            relations_to_run = get_relations_to_run(task_name, relation_names, relations_list)
            populated_tasks_to_run[task_name] = relations_to_run
        else:
            print(f"Task '{task_name}' configuration not found.")
    return populated_tasks_to_run

def get_relations_to_run(task_name: str, relation_names: list, relations_list: dict) -> list[str]:
    # if relations is [], get all relations
    if not relation_names:
        return relations_list[task_name].keys() 
    
    relations_to_run = []
    for relation_name in relation_names:
        if relation_name in relations_list[task_name]:
            relations_to_run.append(relation_name)
        else:
            print(f"Relation '{relation_name}' not found for task '{task_name}'.")
    return relations_to_run


def run_that_test(llm_name: str, task_name: str, relation_name: str, tasks_list: dict, relations_list: dict, run_config: dict, checkpoint: dict | None) -> None:
    task_config = tasks_list[task_name]
    relation_config = relations_list[task_name][relation_name]
    relation_test = instantiate_relation_test(llm_name, task_name, relation_name, task_config, relation_config)
    run_relation_test(relation_test, run_config, checkpoint)




def instantiate_relation_test(llm_name, task_name, relation_name, task_config, relation_config) -> Relation:
    components = {}
    combined_config = {**task_config, **relation_config}
    for key, config in combined_config.items():
        components[key] = instantiate_class(config["class"], *(config.get("args") or []), **(config.get("kwargs") or {}))
    return Relation(llm_name, task_name, relation_name, **components)

def instantiate_class(path, *args, **kwargs):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_(*args, **kwargs)

def run_relation_test(test: Relation, run_config: dict, checkpoint: dict | None = None) -> None:
    run_test(
        llm_name = test.llm_name,
        task_name = test.task_name,
        relation_name = test.relation_name,
        get_dataset = test.get_dataset,
        input_transformation = test.input_transformation,
        run_sut = test.run_sut, 
        output_relation = test.output_relation,
        verify_source_input = test.verify_source_input,
        verify_source_output = test.verify_source_output,
        verify_followup_input = test.verify_followup_input,
        run_config = run_config,
        checkpoint = checkpoint
    )
