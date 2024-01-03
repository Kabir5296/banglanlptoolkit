from langdetect import detect_langs

def detect_lang(line):
    """
    Detect language from a sentence.

    Args:
        line (string): The sentence as a string.

    Returns:
        string: Language
    """
    try:
        lang=detect_langs(line)[0]
    except:
        return 'err'
    return lang.lang