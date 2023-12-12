from bnunicodenormalizer import Normalizer
from normalizer import normalize
from transformers import pipeline
from .utils import detect_lang
import torch

class BnNLPNormalizer():
    def __init__(self, allow_en=False, translate_en=False, device=None):
        self.uniNorm = Normalizer(allow_english=allow_en)
        self.translate_en = translate_en
        
        if self.translate_en:
            if device is None:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            self.translate_model = pipeline(model="csebuetnlp/banglat5_nmt_en_bn",
                                            use_fast=False,
                                            task='translation',
                                            device=device,
                                            batch_size=12)

    def unicode_normalize(self,sentence):
        return ' '.join([normalized_words['normalized'] for normalized_words in [self.uniNorm(word) for word in sentence.split()] if normalized_words['normalized'] != None])

    def normalize_bn(self, sentences, punct_replacement_token=None):
        normal_sentence = []
        for sentence in sentences:
            language = detect_lang(sentence)
            sentence = normalize(sentence, punct_replacement=punct_replacement_token)

            if self.translate_en and language=='en':
                if self.translate_en:
                    sentence = self.translate_model(sentence)[0]['translation_text']
                sentence = self.unicode_normalize(sentence)
            elif self.translate_en:
                print(f'Language {language} can not be translated. Returning an empty string.')
                sentence = ""
            else:
                sentence = self.unicode_normalize(sentence)
            normal_sentence.append(sentence)
        return normal_sentence