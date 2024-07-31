import re

def sent_tokenize(text:str, sent_endings:list=['।','?', '!', '\n'])->list:
    """Tokenize the given text into sentence level

    Args:
        text (str): Text to be tokenized
        sent_endings (list, optional): list of sentence endings. defaults=['।','?', '!', '\n']

    Returns:
        list: list of sentences seperated by 
    """
    sent_endings = ''.join(sent_endings)
    sent_pattern = re.compile(f'([^{re.escape(sent_endings)}]+[{re.escape(sent_endings)}]+)')
    sentences = sent_pattern.findall(text)
    return [sentence.strip() for sentence in sentences]


def word_tokenize(text:str, keep_punct=True) -> list:
    """Tokenize the given text into word level

    Args:
        text (str): Text to be tokenized
        keep_punct (bool, optional): Whether to keep the punctuation with the words. Defaults to True.

    Returns:
        list: list of word tokens
    """
    word_pattern_with_punct = re.compile(r'([\s।?!,;:"\'\(\)\[\]\{\}])')
    word_pattern_without_punct = re.compile(r'[\s।?!,;:"\'\(\)\[\]\{\}]+')
    word_pattern = word_pattern_with_punct if keep_punct else word_pattern_without_punct
    tokens = word_pattern.split(text)

    return [token for token in tokens if token.strip()]