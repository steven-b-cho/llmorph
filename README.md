# LLMorph: Metamorphic Testing of Large Language Models

LLMorph is a tool to automatically test Large Language Models (LLMs) using Metamorphic Testing (MT), through their use on Natural Language Processing (NLP) tasks. It leverages the property-based nature of MT to uncover faulty behaviours without the need for expensive labeled data. LLMorph is aimed at researchers and developers who want to evaluate the robustness of LLM-based NLP systems.

This repository is the artifact for our ICSME'25 paper, [Metamorphic Testing of Large Language Models for Natural Language Processing](https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf).
This tool currently utilises Metamorphic Relations (MRs) extracted from academic literature on MT4NLP to test LLMs. 
Currently, LLMorph implements 48 out of the 191 MRs collected from the literature. 36 were introduced by the original authors and presented in the accompanying paper. 
The remaining 12 were added in a later, independent Semester 2 Bachelor project.

A list of all implemented MRs can be found in `implemented_mrs.txt`.

Original video demo: https://youtu.be/sHmqdieCfw4 (Using LLMorph 0.1.0)

## Requirements

**x86_64 System**

LLMorph requires an x86_64 CPU and is not compatible with ARM-based systems (including Apple Silicon Macs). This is due to a required dependency (`intel-openmp`) only being available for x86_64 platforms.

**Python 3.14.5**

LLMorph requires Python 3.14.5 exactly. The Python version and all dependencies are pinned to fixed versions to ensure reproducibility of results.

**OpenAI API Key**

An OpenAI API key is required to access OpenAI-hosted models via the `openai` Python package.

## Installation

Once the requirements are met, install LLMorph using

```
python install_llmorph.py
```
This will also create a Python virtual environment if one does not already exist.

## Usage

### Running the tool

Run `llmorph.bat` or `llmorph.sh` (depending on OS) to activate the venv.

The tool can be run by either using arguments from a configuration file (giving more control), or by providing some of the arguments in the Command Line Interface (CLI).

#### Config (Recommended)

To run the tool using the config, use

```
python src/mt_main.py
```

or simply

```
llmorph
```

This will run the tool based on the configuration file found at `src/config/run_config.json`.

Default config values can be found in `src/config/run_config_defaults.json`.

All available config options can be found in `src/config/run_config_all_settings.json`

#### CLI

To run the tool from the command line, use

```
python src/main.py llm task mr input_data base_dir
```

with:

- `llm`: The name of the LLM to test.
- `task`: The name of the NLP task to test on.
- `mr`: The name of the metamorphic relation to test using.
- `input_data`: The path to the JSON file containing the inputs. Structured as an array of data points.
- `base_dir`: The path to the directory where caches and outputs will be stored. Outputs are found in `{base_dir}/results`.

Names of MRs and tasks can be found in `src/config/list_relations.json` and `src/config/list_tasks.json`, respectively.

### Results

Results are found in `{base_dir}/results` (with `{base_dir}` specified in the configuration; see above). They include the LLM name, the task name, the metamorphic relation ID, the source and follow-up inputs, the source and follow-up outputs, and the output satisfactions. Results are saved after every relation tested.

### Changing LLMs

This project uses the `openai` Python package to manage LLMs. To change the LLM under test, specify the relevant model name in the `llm` parameter in the CLI; or, if using the config file, specify the config value `llm_list` and the API endpoint in `llm_endpoint`. To use a different API, or to use a locally hosted LLM, please modify `src/llm_runner.py`.

### Adding or modifying tasks

Tasks are currently specified via a zero-shot prompting procedure. To add or modify tasks, go to `src/config/template/sut_prompt_templates.json` to implement the prompt, and `src/config/list_tasks.json` to specify the particular task.

### Adding or modifying metamorphic relations

MRs are implemented as either functions or LLM prompts. To add or modify MRs, go to: `src/relations/func_it.py` and `src/relations/func_or.py` for the implementation of the input transformation and output relation, respectively; `src/config/template/it_prompt_templates.json` or `src/config/template/or_prompt_templates.json` if using a prompted LLM for transformation or comparison; and `src/config/list_relations.json` to specify the particular MR.

## Examples

### Basic Example

To test the installation and see a basic example, run:

```
python src/main.py gpt-5.4 question_answering 5 data/data-example/source_inputs/data.json data/data-example
```

This will test the LLM `gpt-5.4` on the `question_answering` task, using the MR with ID `5` (in this case, the "add random spaces" MR), on the single example input value found at `data/data-example/source_inputs/data.json`. Data will be generated in the `data/data-example` directory, with the final output in `data/data-example/results`.

### Datasets

Example datasets for each task are currently being pulled from HuggingFace. To download and clean, you can run 

```
python src/data_with_labels.py
```

Alternatively, set the config value `use_existing_source_inputs` to `false` to automatically download and use the datasets when the tool is run.

### Paper Reproduction

To reproduce the RQ1 data found in our paper, write the following configuration into `src/config/run_config.json`:

Update (April 2026): The gpt-4-1106 snapshot used in this study is no longer publicly available. As a result, exact replication of the reported results may not be possible. The tests can still be run using `gpt-4` (latest gpt-4 snapshot); however, minor differences in outputs may occur due to model updates. This setup is only intended for paper reproduction purposes, as this legacy model is prohibitively expensive to use.

```
{
    "run_all": true,
    "llm_list": [
        "nous-hermes-2-mixtral-8x7b-dpo",
        "llama-3.1-70b-instruct",
        "gpt-4-1106"
    ],
    "llm_for_transformation": "nous-hermes-2-mixtral-8x7b-dpo",
    "use_existing_source_inputs": false,
    "dir_base_default": "data/data-reproduction"
}
```

Then, run

```
python src/mt_main.py
```

The results will be found in `data/data-reproduction/results`.

## Contribution

If you would like to contribute to this project by implementing new MRs or tasks, you may follow the instructions outlined above, then open a pull request. Any and all contributions are appreciated, for the furthering of the utility of this tool.

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

## Acknowledgments

**Original Authors:**   
Steven Cho, Stefano Ruberto, and Valerio Terragni – creators of LLMorph and authors of the ICSME'25 paper [Metamorphic Testing of Large Language Models for Natural Language Processing](https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf).

**Team Contributions:**          
The following team enhanced LLMorph and implemented additional metamorphic relations as part of a S2 Bachelor Semester Project:

Schickes Christophe     
Steichen Laura      
Dias David

We thank the original authors for providing the foundation that enabled these extensions.