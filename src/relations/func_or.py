from .func_base import FuncOR
import string
from text_similarity import bert_cosine_similarity
from llm_runner import run_template_gpt
import re

class ORDiscreteBase(FuncOR):
    def clean_text(self, text):
        lowercased = text.lower()
        no_punctuation = ''.join(char for char in lowercased if char not in string.punctuation)
        cleaned = ' '.join(no_punctuation.split())
        return cleaned
    
class ORNumericBase(FuncOR):
    def string_to_float(self, s):
        match = re.search("\\d+\\.?\\d*", s)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None
    
    def strings_ok(self, t1, t2):
        return self.string_to_float(t1) is not None and self.string_to_float(t2) is not None
    
    def output_relation(self, t1, t2):
        pass
        


class OREquivalence(ORDiscreteBase):
    def output_relation(self, t1, t2):
        return self.clean_text(t1) == self.clean_text(t2)

class ORDifference(ORDiscreteBase):
    def output_relation(self, t1, t2):
        return self.clean_text(t1) != self.clean_text(t2)

class ORInverse(ORDiscreteBase):
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2

    def output_relation(self, t1, t2):
        ct1 = self.clean_text(t1)
        ct2 = self.clean_text(t2)
        return (ct1 == self.val1 and ct2 == self.val2) or (ct1 == self.val2 and ct2 == self.val1)

class ORNotInverse(ORInverse):
    def output_relation(self, t1, t2):
        return not super().output_relation(t1, t2)

class OREquivalenceNumeric(ORNumericBase):
    def __init__(self, error=0):
        self.error = error

    def is_equivalent(self, t1, t2):
        return abs(self.string_to_float(t1) - self.string_to_float(t2)) <= self.error

    def output_relation(self, t1, t2):
        if not self.strings_ok(t1, t2):
            return None
        try:#TODO check error handling  startegy
            return self.is_equivalent(t1, t2)
            #TypeError: unsupported    operand     type(s)    for -: 'NoneType' and 'float'
        except Exception:
            return None

class ORDifferenceNumeric(OREquivalenceNumeric):
    def output_relation(self, t1, t2):
        return not super().output_relation(t1, t2)
        
class ORStronger(ORNumericBase):
    # the number is closer towards either extreme
    def __init__(self, range_start=0, range_end=1):
        self.range_start = range_start
        self.range_end = range_end

    def is_stronger(self, t1, t2):
        n1 = self.string_to_float(t1)
        n2 = self.string_to_float(t2)
        mid = (self.range_start + self.range_end) / 2
        if n1 == n2 and (n1 == self.range_start or n1 == self.range_end):
            return True # default to True if both at the extreme
        # return n2 < n1 if n1 < mid else n2 > n1 if n1 > mid else True # default to True if n1 == mid
        return n2 <= n1 if n1 < mid else n2 >= n1 if n1 > mid else True # not strictly stronger

    def output_relation(self, t1, t2):
        if not self.strings_ok(t1, t2):
            return None
        return self.is_stronger(t1, t2)

class ORSameStronger(ORStronger, OREquivalenceNumeric):
    def __init__(self, error=0, range_start=0, range_end=1):
        self.error = error
        self.range_start = range_start
        self.range_end = range_end

    def output_relation(self, t1, t2):
        if not self.strings_ok(t1, t2):
            return None
        return self.is_equivalent(t1, t2) or self.is_stronger(t1, t2)


class ORGPT(ORDiscreteBase, FuncOR):
    def __init__(self, prompt_template: str, examples: list=[]):
        self.prompt_template = prompt_template
        self.examples = examples

    def run_gpt(self, input):
        if not isinstance(input, list):
            input = [input]
        res = run_template_gpt(input, self.prompt_template, self.examples)
        res_cleaned = self.clean_text(res)
        return res_cleaned == "true" # if not 'true', then regard as false
    
    def output_relation(self, t1, t2):
        return self.run_gpt([t1, t2])

class ORBERTSimilarityBase(FuncOR):
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def is_similar(self, t1, t2):
        sim = bert_cosine_similarity(t1, t2)
        return sim >= self.threshold

class ORSubsetBase(FuncOR):
    def is_subset(self, t1, t2):
        t1_c = self.clean_text(t1)
        t2_c = self.clean_text(t2)
        return t1_c in t2_c or t2_c in t1_c

class ORBERTSimilarity(ORDiscreteBase, ORBERTSimilarityBase, ORSubsetBase):
    def output_relation(self, t1, t2):
        return self.is_similar(t1, t2) or self.is_subset(t1, t2)

class ORBERTNotSimilarity(ORBERTSimilarity):
    def output_relation(self, t1, t2):
        return not super().output_relation(t1, t2)

class ORBERTSimilarityQA(ORBERTSimilarity):
    
    def generate_question_from_answer(self, answer):
        prompt_template = "What is the question that the following statement answers? Only output the question, nothing else.\nAnswer: {INPUT_0}\nQuestion:"
        examples = [
            [["Nothing happened in 1992; it was in 1993 that the war ended."], "What happened in 1992?"],
            [["There are 235 marbles in the jar."], "How many marbles are in the jar?"],
            [["The text does not state the name of his mother."], "What was the name of George's mother?"],
        ]
        res = run_template_gpt(answer, prompt_template, examples)
        return res
    
    def is_unknown_gpt(self, t):
        # runs only if 'unknown' is not in the answer
        
        prompt_template = "TRUE or FALSE? The following statement answers the question that was asked.\n<statement>{INPUT_0}</statement>"
        examples = [
            [["Nothing happened in 1992; it was in 1993 that the war ended."], "FALSE"],
            [["There are 235 marbles in the jar."], "TRUE"],
            [["The text does not state the name of his mother."], "FALSE"],
            [["I think James did it."], "TRUE"],
        ]
        res = run_template_gpt(t, prompt_template, examples)

        # question = self.generate_question_from_answer(t)
        # prompt_template = "TRUE or FALSE? The responder had the required information to answer the question:\n<question>{INPUT_1}</question>\n<response>{INPUT_0}</response>"
        # examples = [
        #     [["Nothing happened in 1992; it was in 1993 that the war ended.", "What happened in 1992?"], "FALSE"],
        #     [["There are 235 marbles in the jar.", "How many marbles are in the jar?"], "TRUE"],
        #     [["The text does not state the name of his mother.", "What was the name of George's mother?"], "FALSE"],
        # ]
        # res = run_template_gpt([t, question], prompt_template, examples)

        res_cleaned = self.clean_text(res)
        # default to giving original response if the response is not in the format 'true'
        is_unknown = res_cleaned != "true"
        return is_unknown

    def is_unknown(self, t):
        return self.is_subset(t, "unknown") or self.is_unknown_gpt(t)

    def output_relation(self, t1, t2):
        results = []
        for t in [t1, t2]:
            t_is_unknown = self.is_unknown(t)
            inferred_t = "unknown" if t_is_unknown else t
            results.append(inferred_t)

        return super().output_relation(results[0], results[1])
    
class ORBERTNotSimilarityQA(ORBERTSimilarityQA):
    def output_relation(self, t1, t2):
        return not super().output_relation(t1, t2)
