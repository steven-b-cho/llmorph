from mt_main import run_using_config
import argparse


def main():
    parser = argparse.ArgumentParser(description="LLMorph: A framework for testing LLMs with metamorphic relations.")
    parser.add_argument("llm", type=str, help="The name of the LLM to test.")
    parser.add_argument("task", type=str, help="The name of the NLP task to test on.")
    parser.add_argument("mr", type=str, help="The name of the metamorphic relation to test using.")
    parser.add_argument("input_data", type=str, help="The path to the JSON file containing the inputs. Structured as an array of data points.")
    parser.add_argument("base_dir", type=str, help="The path to the directory where caches and outputs will be stored.")

    args = parser.parse_args()

    config = {
        "tasks": {args.task: [args.mr]},
        "llm_list": [args.llm],
        "existing_source_inputs": args.input_data,
        "dir_base_default": args.base_dir,
    }

    run_using_config(config)

if __name__ == "__main__":
    main()
