from abc import ABC, abstractmethod

class FuncDB(ABC):
    @abstractmethod
    def get_dataset(self):
        pass

class FuncIT(ABC):
    @abstractmethod
    def input_transformation(self):
        pass

class FuncOR(ABC):
    @abstractmethod
    def output_relation(self):
        pass

class FuncSUT(ABC):
    @abstractmethod
    def run_sut(self):
        pass

class FuncVerify(ABC):
    @abstractmethod
    def verify(self):
        pass