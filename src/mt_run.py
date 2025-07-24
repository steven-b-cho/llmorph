from file_handler import save_json, load_json, get_timestamp, file_exists
from relations.func_base import FuncDB, FuncIT, FuncOR, FuncSUT, FuncVerify
from tqdm import tqdm
import os

def run_test(llm_name : str, task_name : str, relation_name : str, get_dataset : FuncDB, input_transformation : FuncIT, run_sut : FuncSUT, output_relation : FuncOR, run_config : dict, checkpoint : dict | None = None, verify_source_input : FuncVerify = None, verify_source_output : FuncVerify = None, verify_followup_input : FuncVerify = None):
    # print("Running test...")
    print("Loading dataset...")

    dataset = get_save_source_input_data(get_dataset, task_name, run_config)
    
    # downloaded dataset size given by value in func_db.py, but tested data is given by config to limit tests done
    dataset_old_len = len(dataset)
    do_limit = run_config["data_start_index"] is not None and run_config["data_end_index"] is not None and run_config["data_start_index"] >= 0 and run_config["data_end_index"] >= 0
    if do_limit:
        dataset = dataset[run_config["data_start_index"]:run_config["data_end_index"]]

    source_input_verification = get_save_vsi(dataset, verify_source_input, task_name, relation_name, run_config)
    source_output_data = get_source_output_data(dataset, run_sut, llm_name, task_name, run_config, checkpoint)

    # if source_output_data is cached from old dataset, limit it to the new dataset
    if do_limit and len(source_output_data) == dataset_old_len:
        source_output_data = source_output_data[run_config["data_start_index"]:run_config["data_end_index"]]

    # generate only source data
    if run_config.get("only_generate_source"):
        print("FINISHED SOURCE GENERATION")
        return

    source_output_verification = get_save_vso(source_output_data, source_input_verification, verify_source_output, task_name, relation_name, run_config)
    followup_input_data = get_followup_input_data(dataset, input_transformation, source_input_verification, task_name, relation_name, run_config, checkpoint)
    followup_input_verification = get_save_vfi(followup_input_data, source_input_verification, verify_followup_input, task_name, relation_name, run_config)

    # generate only follow-up data
    if run_config.get("only_generate_followup_inputs"):
        print("FINISHED FOLLOW-UP INPUT GENERATION")
        return

    processed_data = process_dataset(
        dataset, 
        input_transformation, 
        run_sut, 
        output_relation, 
        verify_source_input, 
        verify_source_output, 
        verify_followup_input, 
        llm_name, 
        task_name, 
        relation_name, 
        run_config, 
        source_output_cache=source_output_data,
        followup_input_cache=followup_input_data,
        source_input_verification_cache=source_input_verification,
        source_output_verification_cache=source_output_verification,
        followup_input_verification_cache=followup_input_verification,
        checkpoint=checkpoint
    )
    final_data = {"llm_name": llm_name, "task_name": task_name, "relation_name": relation_name, "data": processed_data}
    log_data(final_data, run_config)


def get_save_source_input_data(get_dataset: FuncDB, task_name: str, run_config: dict):
    if run_config["use_existing_source_inputs"]:
        filename = run_config["existing_source_inputs"]
        if not file_exists(filename):
            raise FileNotFoundError(f"Source inputs file not found: {filename}")
        return load_json(filename)

    # caching source inputs
    if run_config["cache_source_inputs"]:
        filename = os.path.join(run_config["dir_source_inputs"], f"source_inputs__{task_name}.json")
        if not file_exists(filename):
            print(f"Cached source inputs not found. Getting source inputs...")
            dataset = get_dataset()
            save_json(dataset, filename)
            return dataset
        print(f"Cached source inputs found at {filename}")
        return load_json(filename)
    return get_dataset()


def get_save_verify(data: list, to_verify: list[bool], verify_func: FuncVerify, verification_name: str, uid: str, verification_dir: str):
    # caching verification
    if verify_func is None:
        return None
    
    filename = os.path.join(verification_dir, f"{verification_name}__{uid}.json")
    if not file_exists(filename):
        print(f"Cached verification not found. Doing {verification_name}...")
        verify_list = do_verify(data, to_verify, verify_func)
        save_json(verify_list, filename)
        return verify_list
    print(f"Cached verification found at {filename}")
    return load_json(filename)

def do_verify(data: list, to_verify: list[bool], verify_func: FuncVerify):
    return [verify_func(d) if do_verif else None for d, do_verif in tqdm(zip(data, to_verify), total=len(data))]

def get_save_vsi(source_inputs: list, verify_source_input: FuncVerify, task_name: str, relation_name: str, run_config: dict):
    source_inputs = [i[0] for i in source_inputs]
    to_verify = [True] * len(source_inputs)
    return get_save_verify(source_inputs, to_verify, verify_source_input, "verify_source_input", f"{task_name}__{relation_name}", run_config["dir_vsi"])

def get_save_vso(source_outputs: list, source_input_verification: list[bool|None], verify_source_output: FuncVerify, task_name: str, relation_name: str, run_config: dict):
    return get_save_verify(source_outputs, source_input_verification, verify_source_output, "verify_source_output", f"{task_name}__{relation_name}", run_config["dir_vso"])

def get_save_vfi(followup_inputs: list, source_input_verification: list[bool|None],  verify_followup_input: FuncVerify, task_name: str, relation_name: str, run_config: dict):
    return get_save_verify(followup_inputs, source_input_verification, verify_followup_input, "verify_followup_input", f"{task_name}__{relation_name}", run_config["dir_vfi"])




def get_source_output_data(dataset, run_sut : FuncSUT, llm_name: str, task_name: str, run_config: dict, checkpoint : dict | None):
    # caching source outputs
    if run_config["cache_source_outputs"]:
        output_data_filename = os.path.join(run_config["dir_source_outputs"], f"source_outputs__{llm_name}__{task_name}.json")
        if not file_exists(output_data_filename):
            print(f"Cached source outputs not found. Generating source outputs...")
            run_and_save_source_outputs(dataset, run_sut, output_data_filename, llm_name, task_name, run_config, checkpoint)
        else:
            print(f"Cached source outputs found at {output_data_filename}")
        output_data = load_json(output_data_filename)
    else:
        output_data = None
    return output_data

def run_and_save_source_outputs(dataset, run_sut : FuncSUT, filename : str, llm_name : str, task_name : str, run_config : dict, checkpoint : dict | None = None):
    if checkpoint and checkpoint["checkpoint_type"] != "source_outputs": checkpoint = None
    outputs = [] if not checkpoint else checkpoint["data"]
    checkpoint_interval = run_config["checkpoint_interval"]

    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f"Generating source outputs for ({llm_name}, {task_name})"):
        if checkpoint and i < checkpoint["next_id"]: # start from checkpoint
            continue
        if not isinstance(data, list):
            data = [data]
        outputs.append(run_sut(data))
        if checkpoint_interval and checkpoint_interval > 0 and i % checkpoint_interval == checkpoint_interval - 1:
            save_checkpoint(outputs, checkpoint_type="source_outputs", next_id=i+1, llm_name=llm_name, task_name=task_name, relation_name="", run_config=run_config)
    
    print(f"Source outputs cached to {filename}")
    save_json(outputs, filename)

def get_followup_input_data(dataset, input_transformation : FuncIT, source_input_verification : list, task_name: str, relation_name: str, run_config: dict, checkpoint : dict | None):
    # caching follow-up inputs
    if run_config["cache_followup_inputs"]:
        input_data_filename = os.path.join(run_config["dir_followup_inputs"], f"followup_inputs__{task_name}__{relation_name}.json")
        if not file_exists(input_data_filename):
            print(f"Cached follow-up inputs not found. Generating follow-up inputs...")
            run_and_save_followup_inputs(dataset, input_transformation, source_input_verification, input_data_filename, task_name, relation_name, run_config, checkpoint)
        else:
            print(f"Cached follow-up inputs found at {input_data_filename}")
        input_data = load_json(input_data_filename)
    else:
        input_data = None
    return input_data

def run_and_save_followup_inputs(dataset, input_transformation : FuncIT, source_input_verification : list, filename : str, task_name : str, relation_name : str, run_config : dict, checkpoint : dict | None = None):
    if checkpoint and checkpoint["checkpoint_type"] != "followup_inputs": checkpoint = None
    inputs = [] if not checkpoint else checkpoint["data"]
    checkpoint_interval = run_config["checkpoint_interval"]

    for i, data in tqdm(enumerate(dataset), total=len(dataset), desc=f"Generating follow-up inputs for ({task_name}, {relation_name})"):
        if checkpoint and i < checkpoint["next_id"]: # start from checkpoint
            continue
        if not isinstance(data, list):
            data = [data]

        # verify source input
        if source_input_verification and not source_input_verification[i]:
            followup_inputs = None
        else:
            followup_inputs = input_transformation(data)
        inputs.append(followup_inputs)

        if checkpoint_interval and checkpoint_interval > 0 and i % checkpoint_interval == checkpoint_interval - 1:
            save_checkpoint(inputs, checkpoint_type="followup_inputs", next_id=i+1, llm_name="", task_name=task_name, relation_name=relation_name, run_config=run_config)
    
    print(f"Follow-up inputs cached to {filename}")
    save_json(inputs, filename)


def process_dataset(dataset,
                    input_transformation : FuncIT, run_sut : FuncSUT, output_relation : FuncOR, verify_source_input : FuncVerify, verify_source_output : FuncVerify, verify_followup_input : FuncVerify, 
                    llm_name : str, task_name : str, relation_name : str, run_config : dict, 
                    source_output_cache: list = None, followup_input_cache: list = None,
                    source_input_verification_cache: list = None, source_output_verification_cache: list = None, followup_input_verification_cache: list = None,
                    checkpoint : dict | None = None):
    
    if checkpoint and checkpoint["checkpoint_type"] != "followup_outputs": checkpoint = None
    data_list = [] if not checkpoint else checkpoint["data"]
    
    checkpoint_interval = run_config["checkpoint_interval"]

    source_output_cache = check_and_initialize_cache(dataset, source_output_cache, "source outputs")
    followup_input_cache = check_and_initialize_cache(dataset, followup_input_cache, "follow-up inputs")
    source_input_verification_cache = check_and_initialize_cache(dataset, source_input_verification_cache, "source input verification")
    source_output_verification_cache = check_and_initialize_cache(dataset, source_output_verification_cache, "source output verification")
    followup_input_verification_cache = check_and_initialize_cache(dataset, followup_input_verification_cache, "follow-up input verification")

    total_length = min(len(dataset), len(source_output_cache), len(followup_input_cache))
    for i, (source_input, source_output, followup_input) in tqdm(enumerate(zip(dataset, source_output_cache, followup_input_cache)), total=total_length, desc=f"Running test for ({llm_name}, {task_name}, {relation_name})"):
        if checkpoint and i < checkpoint["next_id"]: # start from checkpoint
            continue

        do_limit = run_config["data_start_index"] is not None and run_config["data_end_index"] is not None and run_config["data_start_index"] >= 0 and run_config["data_end_index"] >= 0
        start_index = run_config["data_start_index"] if do_limit else 0
        
        data = process_single_input(
            source_input, 
            input_transformation, 
            run_sut, 
            output_relation, 
            source_input_verification_cache[i], 
            source_output_verification_cache[i], 
            followup_input_verification_cache[i], 
            source_output=source_output, 
            followup_inputs=followup_input,
            id=start_index + i
        )
        data_list.append(data)
        
        if checkpoint_interval and checkpoint_interval > 0 and i % checkpoint_interval == checkpoint_interval - 1:
            save_checkpoint(data_list, checkpoint_type="followup_outputs", next_id=i+1, llm_name=llm_name, task_name=task_name, relation_name=relation_name, run_config=run_config)

    return data_list

def check_and_initialize_cache(dataset, cache, cache_name):
    if cache is None:
        cache = [None] * len(dataset)
    if len(cache) != len(dataset):
        print(f"Warning: Mismatch between cached {cache_name} (len: {len(cache)}) and source inputs (len: {len(dataset)})")
    return cache


def print_process(inputs, outputs, relations):
    print("----------------------------------")
    print()
    # print("input_source:\n", "\n".join(str(item) for item in [in_s]))
    # print()
    print("input_follow:\n" + "\n".join(str(item) for item in inputs))
    print()
    print("outputs:\n" + "\n".join(str(item) for item in outputs))
    print()
    print("relations:\n" + "\n".join(str(item) for item in relations))
    print()
    print("----------------------------------")
    print()

# verification caches are required
def process_single_input(source_input, 
                         input_transformation : FuncIT, run_sut : FuncSUT, 
                         output_relation : FuncOR, 
                         source_input_verification: bool, source_output_verification: bool, followup_input_verification: bool,
                         source_output: str = None, followup_inputs: list = None,
                         id: int = None) -> dict:
    
    # outputs: {source_input, source_output, followup_inputs, followup_outputs, error = "si", "so", "fi", None}
    d = {
        "id": id, # int
        "source_input": source_input if isinstance(source_input, list) else [source_input], # list
        "source_output": source_output, # str
        "followup_inputs": followup_inputs, # list
        "followup_outputs": None, # list
        "relations": None, # bool
        "verif_failure": None, # str
    }

    # verify source input
    if not source_input_verification:
        d["verif_failure"] = "si"
        return d

    # get source output
    if d["source_output"] is None:
        d["source_output"] = run_sut(d["source_input"])

    # verify source output
    if not source_output_verification:
        d["verif_failure"] = "so"
        return d
    
    # get follow-up inputs
    if d["followup_inputs"] is None:
        d["followup_inputs"] = input_transformation(d["source_input"])

    # verify follow-up inputs
    if not followup_input_verification:
        d["verif_failure"] = "fi"
        return d
        
    # get follow-up outputs
    d["followup_outputs"] = [run_sut(input) for input in d["followup_inputs"]]

    # no follow-up output verification for the implemented MRs

    # get relations
    d["relations"] = [output_relation(d["source_output"], output) for output in d["followup_outputs"]]
    
    # if any errors in source or follow-up outputs
    if None in d["relations"]:
        d["verif_failure"] = "sfoe"

    return d


def log_data(data, run_config):
    try:
        llm_name = data["llm_name"]
        task_name = data["task_name"]
        relation_name = data["relation_name"]
        filename = f"log__{llm_name}__{task_name}__{relation_name}__{get_timestamp()}.json"
        file_path = os.path.join(run_config["dir_logs"], filename)
        save_json(data, file_path)
        print(f"Log saved to {file_path}")
        # print(data)
    except IOError:
        print ("Couldn't save the log file:", file_path)


def save_checkpoint(data, checkpoint_type: str, next_id: int, llm_name: str, task_name: str, relation_name: str, run_config: dict):
    checkpoint_data = {
        "llm_name": llm_name,
        "task_name": task_name,
        "relation_name": relation_name,
        "checkpoint_type": checkpoint_type, # 'source_outputs', 'followup_inputs', 'followup_outputs'
        "next_id": next_id,
        "run_config": run_config,
        "data": data,
    }

    try:
        filename = f"checkpoint__{llm_name}__{task_name}__{relation_name}__{checkpoint_type}__{next_id}__{get_timestamp()}.json"
        file_path = os.path.join(run_config["dir_checkpoints"], filename)
        save_json(checkpoint_data, file_path)
        print(f"Checkpoint saved to {file_path}")
        # print(checkpoint_data)
    except IOError:
        print ("Couldn't save the checkpoint:", file_path)
