from .func_base import FuncSUT
from llm_handler import run_template_llm
from llm_runner import get_llm_function

class SUTLLM(FuncSUT):
    def __init__(self, prompt_template : str, examples=[]):
        self.prompt_template = prompt_template
        self.examples = examples
        self.run_chosen_llm = None

    def set_llm(self, model_name):
        self.run_chosen_llm = get_llm_function(model_name)

    def run_llm(self, inputs) -> str:
        return run_template_llm(self.run_chosen_llm, inputs, self.prompt_template, self.examples)

    def run_sut(self, inputs):
        return self.run_llm(inputs)
