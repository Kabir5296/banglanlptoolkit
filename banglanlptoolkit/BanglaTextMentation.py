from transformers import pipeline
import torch
import random
from .BnNLPNormalizer import BnNLPNormalizer
from normalizer import normalize
MASK_TOKEN = '[MASK]'

class Compose():
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def forward(self, text: str):
        for aug in self.augmentations:
            text = aug.forward(text)   
            print(text)         
        return text

class Normalize():
    def __init__(self, allow_en = False, punct_replacement_token = None, device='cpu'):
        self.allow_en = allow_en
        self.device = device
        self.punct_replacement_token = punct_replacement_token
        self.normalize_unicode = BnNLPNormalizer(allow_en=self.allow_en, device=self.device)
                
    def forward(self, text):
        text = self.normalize_unicode.unicode_normalize(text)
        return normalize(text)

class TokenReplacement():
    def __init__(self, probability, device = 'cpu'):
        self.device = device
        self.pipeline = pipeline('fill-mask',
                                model='csebuetnlp/banglishbert_generator',
                                tokenizer='csebuetnlp/banglishbert_generator',
                                top_k = 1,
                                device= self.device,
                                )
        self.probability = probability
        
    def forward(self, text):
        if torch.rand(1) >= self.probability:
            return text
        tokens = text.split()
        if len(tokens) <= 2:
            print('Not enough words in the sentence for token replacement, returning original text.')
            return text
        token = tokens[random.choice(range(len(tokens)))]
        rep_text = text.replace(token, MASK_TOKEN)
        aug_text = self.pipeline(rep_text)
        if len(aug_text) > 1:
            return rep_text.replace(MASK_TOKEN, '')
        return aug_text[0]['sequence']
        
class BackTranslation():
    def __init__(self, probability, device = 'cpu'):
        self.device = device
        self.probability = probability
        self.bn2en_pipeline = pipeline(model="csebuetnlp/banglat5_nmt_bn_en",
                                    use_fast=False,
                                    task='translation',
                                    device=self.device,
                                    max_length=512,
                                    )
        
        self.en2bn_pipeline = pipeline(model="csebuetnlp/banglat5_nmt_en_bn",
                                    use_fast=False,
                                    task='translation',
                                    device=self.device,
                                    max_length=512,
                                    )
        
    def forward(self,text):
        if torch.rand(1) >= self.probability:
            return text
        en = self.bn2en_pipeline(text)[0]['translation_text']
        bn = self.en2bn_pipeline(en)[0]['translation_text']
        return bn
    
class ParaPhrase():
    def __init__(self, probability, device = 'cpu'):
        self.device = device
        self.probability = probability
        self.paraphrase_pipeline = pipeline(model="csebuetnlp/banglat5_banglaparaphrase",
                                    task="text2text-generation",
                                    use_fast=False,
                                    device=self.device,
                                    )
        
    def forward(self,text):
        if torch.rand(1) >= self.probability:
            return text
        para = self.paraphrase_pipeline(text)
        return para[0]['generated_text']