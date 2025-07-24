from .func_base import FuncDB, FuncIT, FuncOR, FuncSUT, FuncVerify

class Relation():
    def __init__(self, llm_name: str, task_name: str, relation_name: str, func_db: FuncDB, func_it: FuncIT, func_sut: FuncSUT, func_or: FuncOR, func_vsi: FuncVerify = None, func_vso: FuncVerify = None, func_vfi: FuncVerify = None):
        self.llm_name = llm_name
        self.task_name = task_name
        self.relation_name = relation_name

        self.get_dataset = func_db.get_dataset
        self.input_transformation = func_it.input_transformation
        self.run_sut = func_sut.run_sut
        self.output_relation = func_or.output_relation
        if func_vsi: self.verify_source_input = func_vsi.verify
        if func_vso: self.verify_source_output = func_vso.verify
        if func_vfi: self.verify_followup_input = func_vfi.verify

        self.sut = func_sut

        self.dataset = None

        self.set_llm(llm_name)

    def load_dataset(self):
        self.dataset = self.get_dataset()
    
    def set_llm(self, model_name):
        self.sut.set_llm(model_name)
        self.run_sut = self.sut.run_sut

    # default behaviours
        
    def verify_source_input(self, input):
        return True

    def verify_source_output(self, input):
        return True

    def verify_followup_input(self, input):
        return True
