from bnunicodenormalizer import Normalizer
from normalizer import normalize
from transformers import pipeline
from .utils import detect_lang
import torch

class BnNLPNormalizer():
    def __init__(self, allow_en=False, translate_en=False, device=None):
        """
        Normalize Bangla text. Two kinds of normalizers are used: unicode normalization provided by 'bnunicodenormalizer', normalizer provided by 'csebuetnlp'

        Args:
            allow_en (bool, optional): Allow English words existing in a sentence. If true, the unicodenormalizer won't delete english words existing in a sentence. Defaults to False.
            translate_en (bool, optional): Whether to translate english sentences to Bangla. If set to true and allow_en is also set to true, the english sentences/words will be translated to Bangla. Defaults to False.
            device (Any, optional): The device to use for the translator model. If not defined, the code will automatically detect available device and set to GPU if possible. Defaults to None.
        """
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
        """
        Unicode normalization of given Bangla sentence.

        Args:
            sentence (list): List of sentences to be normalized.

        Returns:
            string: Returns a string with unicode normalized sentence.
        """
        return ' '.join([normalized_words['normalized'] for normalized_words in [self.uniNorm(word) for word in sentence.split()] if normalized_words['normalized'] != None])

    def normalize_bn(self, sentences, punct_replacement_token=None):
        """
        Uses both bnunicodenormalizer and csebuetnormalizer to normalize Bangla sentences for NLP application. Also can detect and translate English sentences to Bangla if necessary.

        Args:
            sentences (string): The sentence to normalize as a string.
            punct_replacement_token (Any, optional): The character or string to replace punctuations with. If set to None, the punctuations will not be removed. Defaults to None.

        Returns:
            string: The normalized (and translated if necessary) sentence as a string.
        """
        normal_sentence = []
        for sentence in sentences:
            language = detect_lang(sentence)
            sentence = normalize(sentence, punct_replacement=punct_replacement_token)

            if self.translate_en and language !='bn':
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