from transformers import pipeline
import torch
import random
from .bnnlpnormalizer import BnNLPNormalizer
from normalizer import normalize
MASK_TOKEN = '[MASK]'

class Compose():
    '''
    Stack several transformation techniques and apply one by one.
    Arguements:
    -----------
        augmentations (list): List of augmentations. You can use prebuilt augmentations techniques in this library or you can use your own custom augmentation class. To build your custom augmentation, build a class and a 'forward' method.
    '''
    def __init__(self, augmentations: list) -> None:
        self.augmentations = augmentations
    
    def forward(self, text: str) -> str:
        '''
        Apply transformation to given text.
        Arguements:
        -----------
            text (str): String input which needs to be transformed.
        Returns:
        --------
            String: Transformed text.
        '''
        for aug in self.augmentations:
            text = aug.forward(text)        
        return text

class Normalize():
    '''
    Normalize Bengali text. The original text is first normalized with bnunicodenormalizer and then again normalized by csebuetnlp/normalizer.
    Arguements:
    -----------
        allow_en (bool, Optional): Whether to allow English words in the normalized sentences or not. Defaults to False.as_integer_ratio
        punct_replacement_token (any, Optional): Token to replace punctuations with. Defaults to None, punctuations are not replaced.
    '''
    def __init__(self, allow_en: bool = False, punct_replacement_token: any = None, device: str ='cpu') -> None:
        self.allow_en = allow_en
        self.device = device
        self.punct_replacement_token = punct_replacement_token
        self.normalize_unicode = BnNLPNormalizer(allow_en=self.allow_en, device=self.device)
                
    def forward(self, text: str) -> str:
        '''
        Apply normalization to given text.
        Arugements:
        -----------
            text (str): String input which needs to be normalized
        Returns:
        --------
            String: Normalized text.
        '''
        text = self.normalize_unicode.unicode_normalize(text)
        return normalize(text, punct_replacement=self.punct_replacement_token)

class TokenReplacement():
    '''
    Replace a random word in a sentence with [MASK] and fill with BERT models to create augmented sentence.
    Arguements:
    -----------
        probability (float, Optional): Probability by which the augmentation is applied.
        device (str, Optional): Device to run the processing. Defaults to CPU. 
    '''
    def __init__(self, probability: float = 0.5, device: str = 'cpu') -> None:
        self.device = device
        self.pipeline = pipeline('fill-mask',
                                model='csebuetnlp/banglishbert_generator',
                                tokenizer='csebuetnlp/banglishbert_generator',
                                top_k = 1,
                                device= self.device,
                                )
        self.probability = probability
        
    def forward(self, text: str) -> str:
        '''
        Replace a random word with [MASK] and let Deep Learning models fill it.
        Arguements:
        -----------
            text (str): Sentence that needs to be augmented.
        Returns:
        --------
            String: Token replaced sentence. 
        '''
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
    '''
    Translate text to English and then back to Bengali to get an augmented version of the sentence.
    Arguements:
    -----------
        probability (float, Optional): Probability by which the augmentation is applied.
        device (str, Optional): Device to run the processing. Defaults to CPU. 
    '''
    def __init__(self, probability: float = 0.5, device: str = 'cpu') -> None:
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
        
    def forward(self,text: str) -> str:
        '''
        Apply back-translation.
        Arguements:
        -----------
            text (str): Sentence that needs to be augmented.
        Returns:
        --------
            String: Augmented sentence. 
        '''
        if torch.rand(1) >= self.probability:
            return text
        en = self.bn2en_pipeline(text)[0]['translation_text']
        bn = self.en2bn_pipeline(en)[0]['translation_text']
        return bn
    
class ParaPhrase():
    '''
    Paraphrase to get an augmented version of a text.
    Arguements:
    -----------
        probability (float, Optional): Probability by which the augmentation is applied.
        device (str, Optional): Device to run the processing. Defaults to CPU.  
    '''
    def __init__(self, probability: float = 0.5, device: str = 'cpu') -> None:
        self.device = device
        self.probability = probability
        self.paraphrase_pipeline = pipeline(model="csebuetnlp/banglat5_banglaparaphrase",
                                    task="text2text-generation",
                                    use_fast=False,
                                    device=self.device,
                                    )
        
    def forward(self,text: str) -> str:
        '''
        Apply paraphrasing.
        Arguements:
        -----------
            text (str): Sentence that needs to be augmented.
        Returns:
        --------
            String: Augmented sentence. 
        '''
        if torch.rand(1) >= self.probability:
            return text
        para = self.paraphrase_pipeline(text)
        return para[0]['generated_text']