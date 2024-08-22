from transformers import pipeline
from .bnnlpnormalizer import BnNLPNormalizer
import torch

class BanglaPunctuation():
    '''
    Initialize Bangla punctuation class. Currently only 3 punctuations are used, '।', ',' and '?'.
    
    Arguements:
    -----------
        device (str, Optional): Device for loading the model to. Defaults to cuda if available, or cpu.
    '''
    def __init__(self, device: str ='cuda:0' if torch.cuda.is_available() else 'cpu') -> None:
        self.device = device
        self.bnormalize = BnNLPNormalizer()
        
        self.label2punc = {'LABEL_0': '', 'LABEL_1': '।', 'LABEL_2': ',', 'LABEL_3': '?'}
        
        self.punctuation_pipeline = pipeline(
            task = 'ner',
            model='kabir5297/BanglaPunctuationModel',
            device=self.device,
        )
        
    def add_punctuation(self,raw_text: str,chunk_size : int = 512) -> str:
        '''
        Infer Bengali text and add punctuations.
        
        Arguements:
        -----------
            raw_text (str): Text to add punctuations to.
            chunk_size (int, Optional): Tokenizer max length to consider while chunking texts.
        
        Returns:
        --------
            String of original text with punctuations. 
        '''
        text = ''
        raw_text = self.bnormalize.unicode_normalize(raw_text)
        
        tokenized = self.punctuation_pipeline.tokenizer.encode(raw_text)
        chunks = [tokenized[i:i + chunk_size] for i in range(0, len(tokenized), chunk_size)]
        results = []
        
        for chunk in chunks:
            results += self.punctuation_pipeline(self.punctuation_pipeline.tokenizer.decode(chunk, skip_special_tokens=True))
        
        for data in results:
            if data['word'][:2] == '##':
                text += data['word'][2:]+ self.label2punc[data['entity']]
            else:
                text += ' ' + data['word']+ self.label2punc[data['entity']]

        return self.bnormalize.unicode_normalize(text)
    
if __name__ == '__main__':
    punct_agent = BanglaPunctuation()
    print(punct_agent.add_punctuation(raw_text = 'আমার নাম কবির আপনাকে ধন্যবাদ আমার প্যাকেজ ব্যবহার করার জন্য'))