from llm_runner import run_template_gpt
from file_handler import load_json
from .func_base import FuncIT
import random
import string
import math
from nltk.tokenize import sent_tokenize, word_tokenize
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import re

RANDOM_SENTENCES = load_json("./resources/random_sentences.json")
RANDOM_WORDS = load_json("./resources/random_words.json")

# MR list
# 1	Replace characters with random
# 2	Delete characters
# 3	L33t format
# 4	Add characters
# 5	Add spaces
# 6	Swap characters
# 7	Shuffle characters
# 8	Synonym replacement
# 9	Word insertion
# 10	Antonym replacement (output: difference)
# 19	Randomise sentence order
# 25	Replace keywords with random (output: difference)
# 34	Remove keywords (output: difference)
# 49	No change
# 51	Paraphrasing
# 57	Declarative to interrogative sentence
# 77	Generate premise from hypothesis and add to premise
# 78	Generate hypothesis from premise and add to hypothesis
# 79	Generate hypothesis from hypothesis and use as hypothesis (output: not opposite)
# 80	Generate hypothesis from premise and use as premise (output: not opposite)
# 84	Add random sentence
# 102	Capitalisation of all
# 120	Back translate
# 126	Keyboard errors
# 127	Misspell
# 128	OCR errors
# 136	Passive/active voice
# 137	Replace word with another in same category
# 141	Swap symmetric entities
# 142	Swap asymmetric entities (output: opposite relation)
# 149	Singular/plural
# 150	Replace . with !
# 151	Add emphasis words
# 152	Negate (output: difference)
# 154	Capitalise important words
# 155	Tense change

class CleanText():
    def clean_text(self, text):
        lowercased = text.lower()
        no_punctuation = ''.join(char for char in lowercased if char not in string.punctuation)
        cleaned = ' '.join(no_punctuation.split())
        return cleaned

class ITBase(FuncIT):
    def __init__(self, transform_indices=[[0]], multi_input=False):
        self.transform_indices = transform_indices
        self.multi_input = multi_input

    def transform_targeted(self, input: list, n: int, transformation) -> str:
        transform_target = input if self.multi_input else input[n]
        return transformation(transform_target)

    def transform_input(self, input: list, transformation):
        unique_indices = set(index for sublist in self.transform_indices for index in sublist)

        # get transformed values for each of any specified index
        transformed_values = [None] * len(input)
        for i in unique_indices:
            transformed_values[i] = self.transform_targeted(input, i, transformation)
        
        # generate the follow-up inputs from those transformed values
        output_values = []
        for indices in self.transform_indices:
            output_value = input.copy()
            for i in indices:
                output_value[i] = transformed_values[i]
            output_values.append(output_value)

        return output_values

class SingleInputTransformer(ITBase):
    def __init__(self, transform_indices=[[0]]):
        super().__init__(transform_indices, False)

# MR-49
class ITNone(FuncIT):
    def __init__(self, *args, **kwargs):
        pass

    def input_transformation(self, input: list):
        return [input]

class GPTRunner():
    def run_gpt(self, input, prompt_template, examples=[]):
        if not isinstance(input, list):
            input = [input]
        return run_template_gpt(input, prompt_template, examples)

class ITGPT(ITBase):
    def __init__(self, prompt_template: str, examples=[], transform_indices=[[0]], multi_input=False):
        super().__init__(transform_indices, multi_input)
        self.prompt_template = prompt_template
        self.examples = examples

    def run_gpt(self, input):
        if not isinstance(input, list):
            input = [input]
        return run_template_gpt(input, self.prompt_template, self.examples)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.run_gpt)

class ITGPTSentence(ITGPT):
    def sentence_transform(self, input: list):
        input_sentences = sent_tokenize(input)
        output = [self.run_gpt(sentence) for sentence in input_sentences]
        return ' '.join(output)
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.sentence_transform)

class ITGPTConcatInf(ITGPT):
    def input_transformation(self, input: list):
        output = input.copy()
        transform_target = output if self.multi_input else output[self.transform_indices]
        output[self.transform_indices[0][0]] = output[self.transform_indices[0][0]] + " " + self.run_gpt(transform_target) # hardcoded...
        return [output]

class ITConcat(SingleInputTransformer):
    def __init__(self, addition: str, transform_indices=[[0]]):
        super().__init__(transform_indices)
        self.addition = addition

    def concat(self, input_val):
        return input_val + " " + self.addition

    def input_transformation(self, input: list):
        return self.transform_input(input, self.concat)

# MR-84
class ITConcatRandomSentence(SingleInputTransformer):
    def __init__(self, transform_indices=[[0]], rand_seed=42):
        super().__init__(transform_indices)
        self.data = RANDOM_SENTENCES
        self.rand = random.Random(rand_seed)

    def concat_random(self, input_val):
        random_datum = self.rand.choice(self.data)
        return input_val + " " + random_datum
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.concat_random)


# not used
class ITSentiment(CleanText, SingleInputTransformer):
    def get_sentiment(self, inputs): # temporary
        prompt_template = "You are a sentiment analysis tool. Given a sentence, say if it is 'positive' or 'negative' or 'neutral', nothing else.\nWhat is the sentiment of the following sentence?\n\"{INPUT_0}\"\nOnly write a one-word answer."
        response = run_template_gpt([inputs], prompt_template)
        return self.clean_text(response)
 
# not used
class ITGroupBySentiment(ITSentiment):
    def group_sentiments(self, input_val):
        sentences = [sentence.strip() + '.' for sentence in input_val.split('.') if sentence]
        if sentences and not input_val.endswith('.'):
            sentences[-1] = sentences[-1].rstrip('.')
        
        sentiments = {sentence: self.get_sentiment(sentence) for sentence in sentences}
        grouped = {'positive': [], 'negative': [], 'neutral': []}
        for sentence, sentiment in sentiments.items():
            grouped[sentiment].append(sentence)
        return ' '.join([' '.join(grouped[sentiment]) for sentiment in ['positive', 'negative', 'neutral'] if grouped[sentiment]])

    def input_transformation(self, input: list):
        return self.transform_input(input, self.group_sentiments)

# not used
class ITGPTBackTranslate(SingleInputTransformer):
    def back_translate(self, input_val):
        prompt_template_to = "Translate the following into Korean:\n\"{INPUT_0}\"\nOnly output the tranlated text."
        response_to = run_template_gpt([input_val], prompt_template_to)
        prompt_template_from = "Translate the following into English:\n\"{INPUT_0}\"\nOnly output the tranlated text."
        response_from = run_template_gpt([response_to], prompt_template_from)
        return response_from

    def input_transformation(self, input: list):
        return self.transform_input(input, self.back_translate)

class ITPermuteInputs(FuncIT):
    # a permutation list of which element to map to where, e.g. [2,0,1]
    def __init__(self, permute_to: list):
        self.permute_to = permute_to

    def input_transformation(self, input: list):
        output = [''] * len(input)
        for val, n in zip(input, self.permute_to):
            output[n] = val
        return [output]

# MR-102
class ITCapitalisation(SingleInputTransformer):
    def capitalise(self, input_val):
        return input_val.upper()

    def input_transformation(self, input: list):
        return self.transform_input(input, self.capitalise)

# MR-150
class ITReplacePeriodWithExclamation(SingleInputTransformer):
    def replace_period_with_exclamation(self, input_val):
        new_val = input_val.replace('.', '!')
        if not new_val.endswith('!'):
            new_val += '!'
        return new_val

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_period_with_exclamation)



class SingleInputRandomBase(SingleInputTransformer):
    def __init__(self, transform_indices=[[0]], rand_seed=42):
        super().__init__(transform_indices)
        self.rand = random.Random(rand_seed)

# MR-19
class ITRandomiseSentenceOrder(SingleInputRandomBase):
    def randomise_sentences(self, input_val):
        sentences = sent_tokenize(input_val)
        self.rand.shuffle(sentences)
        return ' '.join(sentences)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.randomise_sentences)

# not used
class ITRandomiseWordOrder(SingleInputRandomBase):
    def randomise_words(self, input_val):
        words = word_tokenize(input_val)
        words_without_punct = [word for word in words if word not in string.punctuation]
        self.rand.shuffle(words_without_punct)
        return ' '.join(
            word if word in string.punctuation else words_without_punct.pop(0)
            for word in words
        )

    def input_transformation(self, input: list):
        return self.transform_input(input, self.randomise_words)

# not used
class ITRandomiseWordOrderInSentence(SingleInputRandomBase):
    def shuffle_sentence(self, sentence):
        words = word_tokenize(sentence)
        words_without_punct = [word for word in words if word not in string.punctuation]
        self.rand.shuffle(words_without_punct)
        return ' '.join(
            word if word in string.punctuation else words_without_punct.pop(0)
            for word in words
        )

    def shuffle_text(self, text):
        sentences = sent_tokenize(text)
        return ' '.join(self.shuffle_sentence(sentence) for sentence in sentences)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.shuffle_text)


# base class for character, word and sentence transformations
class ObjectRandomBase(SingleInputRandomBase):
    def __init__(self, transform_indices=[[0]], replace_perc=0.1, rand_seed=42):
        super().__init__(transform_indices, rand_seed)
        self.replace_perc = replace_perc
    
    def transform_function(self, text):
        tokens = self.tokenise(text)
        num_mutated = math.ceil((len(tokens) - 1) * self.replace_perc) # don't do all
        ids = self.rand.sample(range(len(tokens)), num_mutated)
        new_text = self.object_transform(ids, tokens)
        return self.join_tokens(new_text)
    
    def tokenise(self, text):
        pass

    def join_tokens(self, tokens):
        pass
    
    def object_transform(self, ids: list, text):
        pass

    def input_transformation(self, input: list):
        return self.transform_input(input, self.transform_function)


class CharacterRandomBase(ObjectRandomBase):
    def tokenise(self, text):
        return list(text)
    
    def join_tokens(self, tokens):
        return ''.join(tokens)
    
    def transform_function(self, text):
        # only modifies tokens that are letters
        tokens = self.tokenise(text)
        letter_ids = [i for i, token in enumerate(tokens) if token.isalpha()]
        num_mutated = math.ceil(len(letter_ids) * self.replace_perc)
        ids = self.rand.sample(letter_ids, num_mutated)
        new_text = self.object_transform(ids, tokens)

        return self.join_tokens(new_text)


# MR-1
class ITReplaceCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = chr(self.rand.randint(97, 122))
        return text

# MR-2
class ITDeleteCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = ''
        return text

# MR-4
class ITAddCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = text[i] + chr(self.rand.randint(97, 122))
        return text

# MR-3
class ITLeetFormat(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        leet_dict = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 't': '7'}
        for i in ids:
            text[i] = leet_dict.get(text[i], text[i])
        return text

# MR-5
class ITAddSpaces(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            text[i] = text[i] + ' '
        return text

# MR-6
class ITSwapCharacters(CharacterRandomBase):
    def object_transform(self, ids: list, text: list):
        for i in ids:
            if i < len(text) - 1:
                text[i], text[i + 1] = text[i + 1], text[i]
        return text

class WordRandomBase(ObjectRandomBase):
    def tokenise(self, text):
        return word_tokenize(text)
    
    def join_tokens(self, tokens):
        return ' '.join(tokens)

class ITRandomiseCharacterOrderInWord(WordRandomBase):
    def object_transform(self, ids: list, text):
        for i in ids:
            text[i] = ''.join(self.rand.sample(text[i], len(text[i])))
        return text

# MR-7
class ITRandomiseCharacterOrderInWordKeepingEnds(WordRandomBase):
    def transform_function(self, text):
        tokens = self.tokenise(text)
        ids_at_least_4 = [i for i, token in enumerate(tokens) if len(token) >= 4]
        num_mutated = math.ceil(len(ids_at_least_4) * self.replace_perc)
        ids = self.rand.sample(ids_at_least_4, num_mutated)
        new_text = self.object_transform(ids, tokens)
        return self.join_tokens(new_text)
    
    def object_transform(self, ids: list, text):
        for i in ids:
            if len(text[i]) > 3:
                for _ in range(10): # tries to get different word 10 times
                    new_text = text[i][0] + ''.join(self.rand.sample(text[i][1:-1], len(text[i]) - 2)) + text[i][-1]
                    if new_text != text[i]:
                        text[i] = new_text
                        break
        return text

# MR-9
class ITAddRandomWordAfter(WordRandomBase):
    def get_random_word(self):
        return self.rand.choice(RANDOM_WORDS)
    
    def object_transform(self, ids: list, text):
        out_list = []
        for i, t in enumerate(text):
            out_list.append(t)
            if i in ids:
                random_word = self.get_random_word()
                out_list.append(random_word)
        return out_list


class SentenceRandomBase(ObjectRandomBase):
    def tokenise(self, text):
        return sent_tokenize(text)
    
    def join_tokens(self, tokens):
        return ' '.join(tokens)

# not used
class ITDeleteSentences(SentenceRandomBase):
    def object_transform(self, ids: list, text: list):
        return [sentence for i, sentence in enumerate(text) if i not in ids]

# not used
class ITReplaceSentences(SentenceRandomBase):
    def __init__(self, transform_indices=[[0]], replace_perc=0.1, rand_seed=42):
        super().__init__(transform_indices, replace_perc, rand_seed)
        self.data = RANDOM_SENTENCES

    def object_transform(self, ids: list, text: list):
        dummy_sentence = self.rand.choice(self.data)
        for i in ids:
            text[i] = dummy_sentence
        return text


# NLPAUG
# MR 126 (keyboard), 127 (spelling), 128 (ocr), 120 (back_translation)

nlpaug_kwargs = {
    'keyboard': {
        'aug_char_p': 0.1,
        'aug_word_p': 0.1,
    },
    'ocr': {
        'aug_char_p': 0.1,
        'aug_word_p': 0.1,
    },
    'spelling': {
        'aug_p': 0.1,
    },
    'back_translation': {},
}

initialised_augmenter_map = {
    'keyboard': nac.KeyboardAug(**nlpaug_kwargs['keyboard']),
    'ocr': nac.OcrAug(**nlpaug_kwargs['ocr']),
    'spelling': naw.SpellingAug(**nlpaug_kwargs['spelling']),
    'back_translation': naw.BackTranslationAug(**nlpaug_kwargs['back_translation']),
}

class ITNlpaug(SingleInputTransformer):
    def __init__(self, augment_type: str, transform_indices=[[0]], **augmenter_kwargs):
        super().__init__(transform_indices)
        self.augmenter = self._initialize_augmenter(augment_type, **augmenter_kwargs)

    def _initialize_augmenter(self, augmenter_type, **kwargs):
        augmenter_map = {
            'keyboard': nac.KeyboardAug,
            'ocr': nac.OcrAug,
            # 'random_char': nac.RandomCharAug,
            # 'antonym': naw.AntonymAug,
            # 'contextual_word_embs': naw.ContextualWordEmbsAug,
            # 'random_word': naw.RandomWordAug,
            'spelling': naw.SpellingAug,
            # 'split': nac.SplitAug,
            # 'synonym': naw.SynonymAug,
            # 'tfidf': naw.TfIdfAug,
            # 'word_embs': naw.WordEmbsAug,
            'back_translation': naw.BackTranslationAug,
            # 'reserved': naw.ReservedAug,
            # 'contextual_word_embs_sentence': nas.ContextualWordEmbsForSentenceAug,
            # 'abst_summ': nas.AbstSummAug,
            # 'lambada': nas.LambadaAug,
        }

        if augmenter_type not in augmenter_map:
            raise ValueError(f"Unsupported augmenter type: {augmenter_type}")
        
        # return augmenter_map[augmenter_type](**kwargs)
        return initialised_augmenter_map[augmenter_type]

    def nlp_transform(self, input_val):
        augmented_text = self.augmenter.augment(input_val)
        return augmented_text if isinstance(augmented_text, str) else augmented_text[0]
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.nlp_transform)





# KEYWORD-BASED MRS

class GPTKeywordBase(CleanText, GPTRunner, ITBase):
    def get_keywords_gpt(self, input):
        prompt_template = "Identify names, pronouns, country names, occupations, and similar keywords in the following text:\n\"{INPUT_0}\"\nOnly output the list of words, nothing else."
        examples = [
            [["Sarah is an American software engineer. She works for Microsoft."], "Sarah\nAmerican\nsoftware engineer\nMicrosoft"],
            [["My brother will travel to Japan next month to study Japanese."], "brother\nJapan\nJapanese"],
        ]
        return self.run_gpt(input, prompt_template, examples)
    
    def get_keywords(self, input):
        keywords = self.get_keywords_gpt(input)
        keywords_list = re.split(r'[,\n]+\s*', keywords)
        keywords_list_cleaned = [self.clean_text(keyword) for keyword in keywords_list]
        return keywords_list_cleaned
    
    def bind_context_kw(self, context, keywords):
        if isinstance(context, list):
            context = '\n'.join(context)
        return [context, keywords]
    

class ReplaceKeyword(GPTKeywordBase):
    def replace_words(self, text: str, words_from: list[str], words_to: list[str]):
        # return self.replace_words_manual(text, words_from, words_to)
        return self.replace_words_gpt(text, words_from, words_to)

    def replace_words_manual(self, text: str, words_from: list[str], words_to: list[str]):
        for word_from, word_to in zip(words_from, words_to):
            text = re.sub(r'\b' + word_from + r'\b', word_to, text, flags=re.IGNORECASE)
        return text
    
    def replace_words_gpt(self, text: str, words_from: list[str], words_to: list[str]):
        prompt_template = "Look at this text:\n\"{INPUT_0}\"\nReplace words using the following rules:\n{INPUT_1}\nOnly replace any words that are there. Ignore any rules that are not used. Only output the modified text."
        examples = self.get_replace_examples()
        replace_words = '\n'.join(f"{word_from} -> {word_to}" for word_from, word_to in zip(words_from, words_to))
        return self.run_gpt([text, replace_words], prompt_template, examples)
    
    def get_replace_examples(self):
        pass


# 137 - CATEGORY
class ITReplaceKeywordCategory(ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "Sarah is an American software engineer. She works for Microsoft.", 
            "software engineer -> farmer\nSarah -> John\nAmerican -> Swedish\napple -> pear\nMicrosoft -> Nvidea"], 
            "John is a Swedish farmer. He works for Nvidea."],
            [["My brother will travel to Japan next month to study Japanese.", 
            "sweater -> shirt\nJapanese -> Irish\nmonth -> year\nEurope -> Asia\nbrother -> sister\nJapan -> Italy"], 
            "My sister will travel to Italy next year to study Irish."],]

    def get_word_same_category(self, input):
        prompt_template = "Give a word or phrase in the same category as \"{INPUT_0}\"."
        examples = [
            [["Sarah"], "John"],
            [["software engineer"], "farmer"],
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word

    def input_transformation(self, input: list):
        combined_inputs = '\n\n'.join(input)
        keywords_list = self.get_keywords(combined_inputs)
        category_words = [self.get_word_same_category(keyword) for keyword in keywords_list]
        outputs = [self.replace_words(input_val, keywords_list, category_words) for input_val in input]
        return [outputs]

class ITReplaceKeywordCategoryQA(ITReplaceKeywordCategory):
    def input_transformation(self, input: list):
        keywords_list = self.get_keywords(input[1]) # keywords from question
        category_words = [self.get_word_same_category(keyword) for keyword in keywords_list]
        outputs = [self.replace_words(input_val, keywords_list, category_words) for input_val in input]
        return [outputs]

class ITReplaceKeywordCategoryRE(ITReplaceKeywordCategory):
    def input_transformation(self, input: list):
        keywords = [input[1], input[2]] # keywords are entities
        keywords_list = [self.clean_text(keyword) for keyword in keywords]
        category_words = [self.get_word_same_category(keyword) for keyword in keywords_list]
        context_new = self.replace_words(input[0], keywords_list, category_words)
        output = [context_new, category_words[0], category_words[1]]
        return [output]

# 8 - SYNONYM
class ITReplaceKeywordSynonym(SingleInputTransformer, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "Sam walked to the store to buy an apple.", 
            "walked -> travelled\nslowly -> unhurriedly\nbuy -> purchase\nstore -> shop"], 
            "I travelled to the shop to purchase an apple."],
            [["I wonder why clothes are so expensive.", 
            "buy -> purachase\nclothes -> garments\nexpensive -> pricy\nwhy -> for what reason\nfull -> complete"], 
            "I wonder for what reason garments are so pricy."],]

    def get_synonym(self, input):
        prompt_template = "Context:\n\"{INPUT_0}\"\nMaking sense in this context, give a synonym for \"{INPUT_1}\". If the word has no synonym, simply output the word itself."
        examples = [
            [["Sam walked to the store to buy an apple.", "walked"], "travelled"],
            [["I wonder why clothes are so expensive.", "expensive"], "pricey"],
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word
    
    def replace_synonym(self, input):
        keywords = self.get_keywords(input)
        synonym_words = [self.get_synonym(self.bind_context_kw(input, keyword)) for keyword in keywords]
        return self.replace_words(input, keywords, synonym_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_synonym)



class ReplaceKeywordDifferenceRE(ReplaceKeyword):
    def get_keywords_gpt(self, text, input_1, input_2):
        prompt_template = "Here are two input words:\n\"{INPUT_1}\"\n\"{INPUT_2}\"\n\nIdentify names, pronouns, country names, occupations, and similar keywords in the following text that is associated with these words:\n\"{INPUT_0}\"\nOnly output the list of words, nothing else."
        examples = [
            [["Sarah is an American software engineer. She works for Microsoft.", "Sarah", "Microsoft"], "American\nsoftware engineer\nworks for"],
            [["My brother will travel to Japan next month to study Japanese.", "brother", "Japan"], "travel\nnext month\nstudy\nJapanese"],
        ]
        return self.run_gpt([text, input_1, input_2], prompt_template, examples)
    
    def get_keywords(self, text, input_1, input_2):
        keywords = self.get_keywords_gpt(text, input_1, input_2)
        keywords_list = re.split(r'[,\n]+\s*', keywords)
        keywords_list_cleaned = [self.clean_text(keyword) for keyword in keywords_list]
        cleaned_inputs = [self.clean_text(input_1), self.clean_text(input_2)]
        keywords_out = [keyword for keyword in keywords_list_cleaned if keyword not in cleaned_inputs] # remove entities if appear in keywords
        return keywords_out


# 10 - ANTONYM
class ITReplaceKeywordAntonym(SingleInputTransformer, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "She walked to the store to buy an apple.", 
            "walked -> ran\nslowly -> quickly\nbuy -> sell\nstore -> home\nshe -> he"], 
            "He ran to the home to sell an apple."],
            [["In 1993, I broke my arm while cleaning my electric car.", 
            "noisy -> silent\nmy -> your\nfull -> empty\nelectric -> petrol\ncleaning -> dirtying\nbroke -> fixed"], 
            "In 1993, I fixed your arm while dirtying your petrol car."],]

    def get_antonym(self, input):
        # prompt_template = "Context:\n\"{INPUT_0}\"\nMaking sense in this context, give an antonym for \"{INPUT_1}\". If the word has no antonym, simply output the word itself."
        prompt_template = "You are given a context and a word. Produce an antonym of the word. Make sure the antonym makes sense in the context. If the word has no antonym, simply output the word itself. \n<context>{INPUT_0}</context>\n<word>{INPUT_1}</word>"
        examples = [
            [["She walked to the store to buy an apple.", "buy"], "sell"],
            [["In 1993, I broke my arm while cleaning my electric car.", "electric"], "petrol"],
        ]
        new_word = self.run_gpt(input, prompt_template, examples)
        cleaned_new_word = self.clean_text(new_word)
        return cleaned_new_word
    
    def replace_antonym(self, input):
        keywords = self.get_keywords(input)
        antonym_words = [self.get_antonym(self.bind_context_kw(input, keyword)) for keyword in keywords]
        return self.replace_words(input, keywords, antonym_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_antonym)

class ITReplaceKeywordAntonymQA(ITReplaceKeywordAntonym):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[1]) # keywords from question
        antonym_words = [self.get_antonym(self.bind_context_kw(input[1], keyword)) for keyword in keywords]
        output_c = self.replace_words(input[0], keywords, antonym_words)
        output_q = self.replace_words(input[1], keywords, antonym_words)
        return [[output_c, input[1]], [input[0], output_q]] # either, not both

class ITReplaceKeywordAntonymRE(ReplaceKeywordDifferenceRE, ITReplaceKeywordAntonym):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[0], input[1], input[2])
        antonym_words = [self.get_antonym(self.bind_context_kw(input[0], keyword)) for keyword in keywords]
        output = self.replace_words(input[0], keywords, antonym_words)
        return [[output, input[1], input[2]]]
    


# 25 - RANDOM
class ITReplaceKeywordRandom(SingleInputRandomBase, ReplaceKeyword):
    def get_replace_examples(self):
        return [[[
            "He walked to the store to buy an apple.", 
            "walked -> cut\nslowly -> break\nbuy -> run\napple -> play\nstore -> fall"], 
            "He cut to the fall to run an play."],
            [["Sarah is an American software engineer. She works for Microsoft.", 
            "software engineer -> carry\nSarah -> give\nAmerican -> light\nsandwich -> clear\nMicrosoft -> call"], 
            "Give is a light carry. She works for call."],]

    def get_random_words(self, n):
        return list(self.rand.sample(RANDOM_WORDS, n))
    
    def replace_random_word(self, input):
        keywords = self.get_keywords(input)
        random_words = self.get_random_words(len(keywords))
        return self.replace_words(input, keywords, random_words)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.replace_random_word)

class ITReplaceKeywordRandomQA(ITReplaceKeywordRandom):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[1]) # keywords from question
        random_words_c = self.get_random_words(len(keywords))
        random_words_q = self.get_random_words(len(keywords))
        output_c = self.replace_words(input[0], keywords, random_words_c)
        output_q = self.replace_words(input[1], keywords, random_words_q)
        return [[output_c, input[1]], [input[0], output_q], [output_c, output_q]] # all combinations

class ITReplaceKeywordRandomRE(ReplaceKeywordDifferenceRE, ITReplaceKeywordRandom):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[0], input[1], input[2])
        random_words = self.get_random_words(len(keywords))
        output = self.replace_words(input[0], keywords, random_words)
        return [[output, input[1], input[2]]]


# 34 - REMOVE
class ITRemoveKeyword(SingleInputTransformer, GPTKeywordBase):
    def remove_keywords(self, input_val, keywords):
        # return self.remove_keywords_manual(input_val, keywords)
        return self.remove_keywords_gpt(input_val, keywords)
    
    def remove_keywords_manual(self, input_val, keywords):
        new_text = input_val
        for keyword in keywords:
            new_text = re.sub(r'\b' + keyword + r'\b', '', new_text, flags=re.IGNORECASE)
        return new_text
    
    def get_gpt_prompt(self):
        prompt_template = "Look at this text:\n\"{INPUT_0}\"\nRemove the following words:\n{INPUT_1}\nOnly remove the words. Only output the modified text."
        examples = [
            [["Sarah is an American software engineer. She works for Microsoft.", "Sarah\nAmerica\nsoftware engineer\nMicrosoft"], "is an. She works for."],
            [["My brothers will travel to Japan next month to study Japanese.", "brother\nJapan"], "My will travel to next to study."],
        ]
        return prompt_template, examples
    
    def remove_keywords_gpt(self, input_val, keywords):
        prompt_template, examples = self.get_gpt_prompt()
        return self.run_gpt([input_val, '\n'.join(keywords)], prompt_template, examples)

    def get_and_remove_keywords(self, input):
        keywords = self.get_keywords(input)
        return self.remove_keywords(input, keywords)

    def input_transformation(self, input: list):
        return self.transform_input(input, self.get_and_remove_keywords)

class ITRemoveKeywordSentence(ITRemoveKeyword):
    def remove_keywords_gpt(self, input_val, keywords):
        prompt_template, examples = self.get_gpt_prompt()

        input_sentences = sent_tokenize(input_val)
        keyword_string = '\n'.join(keywords)

        output = [self.run_gpt([sentence, keyword_string], prompt_template, examples) for sentence in input_sentences]
        return ' '.join(output)
    
class ITRemoveKeywordQA(ITRemoveKeyword):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[1]) # keywords from question
        output_c = self.remove_keywords(input[0], keywords)
        output_q = self.remove_keywords(input[1], keywords)
        return [[output_c, input[1]], [input[0], output_q], [output_c, output_q]] # all combinations

class ITRemoveKeywordQASentence(ITRemoveKeywordQA, ITRemoveKeywordSentence):
    pass

class ITRemoveKeywordRE(ReplaceKeywordDifferenceRE, ITRemoveKeyword):
    def input_transformation(self, input: list):
        keywords = self.get_keywords(input[0], input[1], input[2])
        output = self.remove_keywords(input[0], keywords)
        return [[output, input[1], input[2]]]
    
class ITRemoveKeywordRESentence(ITRemoveKeywordRE, ITRemoveKeywordSentence):
    pass


# 152 - NEGATE
class ITNegate(GPTRunner, SingleInputTransformer, ITBase):
    def get_prompt(self):
        prompt_template = "Negate the following text with minimal change:\n\"{INPUT_0}\"\nOnly output the changed text, nothing else."
        examples = [
            [["Who was the leader of Norway?"], "Who wasn't the leader of Norway?"],
            [["The President ate a cabbage."], "The President didn't eat a cabbage."]
        ]
        return prompt_template, examples
    
    def get_negated(self, input: list):
        prompt_template, examples = self.get_prompt()
        return self.run_gpt(input, prompt_template, examples)
    
    def input_transformation(self, input: list):
        return self.transform_input(input, self.get_negated)

class ITNegateQA(ITNegate):
    def get_negated_context(self, input: list):
        prompt_template = "Given this question:\n\"{INPUT_1}\"\n\nNegate the following text with minimal change such that the information relavent to the question is the opposite:\n\"{INPUT_0}\"\nOnly output the changed text, nothing else."
        examples = [
            [["The leader of Norway was Jonas Gahr Store, son of a wealthy ship broker.", "Who was the leader of Norway?"], "The leader of Norway wasn't Jonas Gahr Store, son of a wealthy ship broker."],
            [["She went to the shops, ate a cabbage, and returned home with a basketball.", "What did she eat?"], "She went to the shops, didn't eat a cabbage, and returned home with a basketball."]
        ]
        return self.run_gpt(input, prompt_template, examples)
    
    def input_transformation(self, input: list):
        # input: [context, question]
        negated_context = self.get_negated_context(input)
        negated_question = self.get_negated([input[1]])
        return [[negated_context, input[1]], [input[0], negated_question]]

class ITNegateRE(ITNegate):
    def get_negated_re(self, input: list):
        prompt_template = "Negate the following text with minimal change such that the relationship from \"{INPUT_1}\" to \"{INPUT_2}\" is the opposite:\n\"{INPUT_0}\"\nNegate the text so that the relationship from \"{INPUT_1}\" to \"{INPUT_2}\" is the opposite.\nOnly output the changed text, nothing else."
        examples = [
            [["The leader of Norway was Jonas Gahr Store, son of a wealthy ship broker.", "Jonas", "Norway"], "The leader of Norway wasn't Jonas Gahr Store, son of a wealthy ship broker."],
            [["She went to the shops, ate a cabbage, and returned home with a basketball.", "she", "cabbage"], "She went to the shops, didn't eat a cabbage, and returned home with a basketball."]
        ]
        return self.run_gpt(input, prompt_template, examples)
    
    