from langdetect import detect_langs

def detect_lang(line):
    try:
        lang=detect_langs(line)[0]
    except:
        return 'err'
    return lang.lang