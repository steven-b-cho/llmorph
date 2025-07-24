from .func_base import FuncVerify
import string
from llm_runner import run_template_gpt
from nltk.tokenize import sent_tokenize
import re

class VerifyDiscreteBase():
    def clean_text(self, text: str):
        lowercased = text.lower()
        no_punctuation = ''.join(char for char in lowercased if char not in string.punctuation)
        cleaned = ' '.join(no_punctuation.split())
        return cleaned

class VerifyEquivalence(VerifyDiscreteBase, FuncVerify):
    def __init__(self, comparison_string: str):
        self.comparison_string = comparison_string
    
    def verify(self, input: str):
        return self.clean_text(input) == self.clean_text(self.comparison_string)

class VerifyDifference(VerifyDiscreteBase, FuncVerify):
    def __init__(self, comparison_string: str):
        self.comparison_string = comparison_string
    
    def verify(self, input: str):
        return self.clean_text(input) != self.clean_text(self.comparison_string)    

class VerifyDifferenceMulti(VerifyDiscreteBase, FuncVerify):
    def __init__(self, comparison_string: str, comparison_index: int):
        self.comparison_string = comparison_string
        self.comparison_index = comparison_index
    
    def verify(self, input: list):
        return self.clean_text(input[0][self.comparison_index]) != self.clean_text(self.comparison_string)

class VerifyInclusion(VerifyDiscreteBase, FuncVerify):
    def __init__(self, comparison_set: list):
        self.comparison_set = [self.clean_text(s) for s in comparison_set]
    
    def verify(self, input: str):
        return self.clean_text(input) in self.comparison_set
    
class VerifySetEquivalence(VerifyDiscreteBase, FuncVerify):
    def __init__(self, comparison_string: str):
        self.comparison_string = comparison_string

    def is_similar(self, text: str):
        return self.clean_text(text) == self.clean_text(self.comparison_string)
    
    def is_subset(self, text: str):
        t1_c = self.clean_text(text)
        t2_c = self.clean_text(self.comparison_string)
        return t1_c in t2_c or t2_c in t1_c

    def verify(self, input: str):
        return self.is_similar(input) or self.is_subset(input)

class VerifySetDifference(VerifySetEquivalence):
    def verify(self, input: str):
        return not super().verify(input)


class VerifyNumericBase(FuncVerify):
    def string_to_float(self, s):
        match = re.search("\\d+\\.?\\d*", s)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None
    
    def strings_ok(self, input):
        return self.string_to_float(input) is not None

class VerifyEquivalenceNumeric(VerifyNumericBase):
    def __init__(self, comparison_value: float, error: float = 0):
        self.comparison_value = comparison_value
        self.error = error

    def is_equivalent(self, input):
        return abs(self.string_to_float(input) - self.comparison_value) <= self.error

    def verify(self, input):
        if not self.strings_ok(input):
            return None
        return self.is_equivalent(input)
    
class VerifyDifferenceNumeric(VerifyEquivalenceNumeric):
    def verify(self, input):
        if not self.strings_ok(input):
            return None
        return not self.is_equivalent(input)

class VerifyEquivalenceRange(VerifyNumericBase):
    def __init__(self, range_start: float, range_end: float):
        self.range_start = range_start
        self.range_end = range_end

    def is_equivalent(self, input):
        return self.range_start <= self.string_to_float(input) <= self.range_end

    def verify(self, input):
        if not self.strings_ok(input):
            return None
        return self.is_equivalent(input)

class VerifyDifferenceRange(VerifyEquivalenceRange):
    def verify(self, input):
        if not self.strings_ok(input):
            return None
        return not self.is_equivalent(input)
        


class VerifyMultipleSentences(FuncVerify):
    def __init__(self, min_sentences: int = 2):
        self.min_sentences = min_sentences

    def verify(self, input: str):
        sentence_count = len(sent_tokenize(input))
        return sentence_count >= self.min_sentences

class VerifyGPT(VerifyDiscreteBase, FuncVerify):
    def __init__(self, prompt_template: str, examples: list=[]):
        self.prompt_template = prompt_template
        self.examples = examples

    def run_gpt(self, input):
        if not isinstance(input, list):
            input = [input]
        res = run_template_gpt(input, self.prompt_template, self.examples)
        res_cleaned = self.clean_text(res)
        return res_cleaned == "true" # if not 'true', then regard as false
    
    def verify(self, input):
        return self.run_gpt(input)
    
class VerifyGPTNot(VerifyGPT):
    def verify(self, input):
        return not super().verify(input)

# sets "no relation" as a special case; always symmetric
class VerifySymmetryGPT(VerifyGPT):
    def is_no_relation(self, input):
        if not isinstance(input, list):
            input = [input]
        return self.clean_text(input[0]) == "no relation"
    
    def verify(self, input):
        if self.is_no_relation(input):
            return True
        return super().verify(input)

# needs symmetric-check prompt as an input
class VerifyAsymmetryGPT(VerifySymmetryGPT):
    def verify(self, input):
        return not super().verify(input)