from bnunicodenormalizer import Normalizer
from normalizer import normalize
from transformers import pipeline
from .utils import detect_lang
import torch

class BnNLPNormalizer():
    """
    Normalize Bangla text. Two kinds of normalizers are used: unicode normalization provided by 'bnunicodenormalizer', normalizer provided by 'csebuetnlp'

    Arguements:
    -----------
        allow_en (bool, optional): Allow English words existing in a sentence. If true, the unicodenormalizer won't delete english words existing in a sentence. Defaults to False.
        translate_en (bool, optional): Whether to translate english sentences to Bangla. If set to true and allow_en is also set to true, the english sentences/words will be translated to Bangla. Defaults to False.
        device (Any, optional): The device to use for the translator model. If not defined, the code will automatically detect available device and set to GPU if possible. Defaults to None.
    """
    def __init__(self, allow_en: bool = False, translate_en: bool = False, device: any = None):
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
            
    def word_normalize(self, word: str) -> str:
        '''
        Normalize each word using bnunicodenormalizer
        
        Arguements:
        -----------
            word (str): Word that needs to be normalized.
        
        Returns:
        --------
            String: The normalized word.
        '''
        normalized_word = self.uniNorm(word)['normalized']
        if normalized_word is not None:
            return  normalized_word 
        else: 
            return ''

    def unicode_normalize(self, sentence: str ) -> str:
        """
        Unicode normalization of given Bangla sentence.

        Arguements:
        -----------
            sentence (string): Sentence to be normalized.

        Returns:
        --------
            string: Returns a string with unicode normalized sentence.
        """
        return ' '.join([self.word_normalize(word) for word in sentence.split()])

    def normalize_bn(self, sentences: list, punct_replacement_token: any = None) -> list:
        """
        Uses both bnunicodenormalizer and csebuetnormalizer to normalize Bangla sentences for NLP application. Also can detect and translate English sentences to Bangla if necessary.

        Arguements:
        -----------
            sentences (list): List of sentences to be normalized.
            punct_replacement_token (Any, optional): The character or string to replace punctuations with. If set to None, the punctuations will not be removed. Defaults to None.
        Returns:
        --------
            The normalized (and translated if necessary) sentence as a list of strings.
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