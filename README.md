# LLMorph: Metamorphic Testing of Large Language Models

LLMorph is a tool to automatically test Large Language Models (LLMs) using Metamorphic Testing (MT), thorough their use on Natural Language Processing (NLP) tasks.
This repository is the artifact for our ICSME'25 paper, Metamorphic Testing of Large Language Models for Natural Language Processing.
In the paper, we extracted 191 metamorphic relations from literature on MT4NLP; this tool currently implements 36.

Video demo: https://youtu.be/sHmqdieCfw4

## Requirements

**Python:** Python 3.10.5

**Dependencies:** Install using

```
pip install -r requirements.txt
```

An OpenAI key is needed in `security/token-key.jwt`.

## Usage

### Running the tool

The tool can be run through either Command Line Interface (CLI) or directly from a script. The latter pulls from a configuration file, giving more control.

#### CLI

To run the tool from command line, use

```
python src/main llm task mr input_data base_dir
```

with:

- `llm`: The name of the LLM to test.
- `task`: The name of the NLP task to test on.
- `mr`: The name of the metamorphic relation to test using.
- `input_data`: The path to the JSON file containing the inputs. Structured as an array of data points.
- `base_dir`: The path to the directory where caches and outputs will be stored. Outputs are found in `{base_dir}/results`.

Names of MRs and tasks can be found in `src/config/list_relations.json` and `src/config/list_tasks.json`, respectively.

To run the example, use

```
python src/main gpt-4o-2024-08-06 question_answering 5 data/data-example/source_inputs/data.json data/data-example
```


#### Advanced Config

To run the tool using the advanced config, use

```
python src/mt_main.py
```

This will run the tool based on the configuration file found at `src/config/run_config.json`.

Default config values can be found in `src/run_config_defaults.json`.

### Results

Results are found in `{base_dir}/results` (with `{base_dir}` specified in the configuration; see above). It includes the LLM name, the task name, the metamorphic relation ID, the source and follow-up inputs, the source and follow-up outputs, and the output satisfactions. Results are saved after every relation tested.

### Changing LLMs

This project uses the `openai` python package to manage LLMs. To change the LLM under test, you can specify the relevant model name in the config value `llm_list` and the API endpoint in `llm_endpoint`. To use a different API, or to use a locally hosted LLM, plese modify `src/llm_runner.py`.

### Example Datasets

Example datasets for each task are currently being pulled from HuggingFace. To download and clean them, you can run 

```
python src/data_with_labels.py
```

Alternatively, set the config value `use_existing_source_inputs` to `false` to automatically download and use the datasets when the tool is run.

### Adding or modifying tasks

Tasks are currently specified via a zero-shot prompting procedure. To add or modify tasks, go to `src/config/template/sut_prompt_templates.json` to implement the prompt, and `src/config/list_tasks.json` to specify the particular task.

### Adding or modifying metamorphic relations

MRs are implemented as either functions or LLM prompts. To add or modify MRs, go to: `src/relations/func_it.py` and `src/relations/func_or.py` for the implementation of the input transformation and output relation, respectively; `src/config/template/it_prompt_templates.json` or `src/config/template/or_prompt_templates.json` if using a prompted LLM for transformation or comparison; and `src/config/list_relations.json` to specify the particular MR.

## Contribution

If you would like to contribute to this project by implementing new MRs or tasks, you may follow the instructions outlined above, then open a pull request. Any and all contributions are apprecated, for the furthering of the utility of this tool.

## Contact

If you have any questions, feel free to contact: steven.cho@aucklanduni.ac.nz

## Citation

```
@inproceedings{cho2025metamorphic,
  author = {Cho, Steven and Ruberto, Stefano and Terragni, Valerio},
  title = {Metamorphic Testing of Large Language Models for Natural Language Processing},
  booktitle = {Proceedings of the IEEE International Conference on Software Maintenance and Evolution (ICSME)},
  year = {2025},
  publisher = {IEEE}
}
```
