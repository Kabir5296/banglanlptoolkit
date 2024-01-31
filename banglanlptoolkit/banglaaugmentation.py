import pandas as pd
from normalizer import normalize
from transformers import pipeline
import warnings, glob, random, torch
from tqdm import tqdm
import torch
tqdm.pandas()

MASK_TOKEN = "[MASK]"

class AugmentationBangla:
    def __init__(self, device='cuda:0', batch_size=12, top_k=1, max_length=512):
        self.device = device
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_length = max_length

        # Unmasking MODEL
        print('\n'+'='*10+'Please wait while the models are being loaded'+'='*10+'\n')
        self.unmasker = pipeline('fill-mask', 
                                 model="csebuetnlp/banglishbert_generator", 
                                 tokenizer="csebuetnlp/banglishbert_generator",
                                 top_k=self.top_k, 
                                 device=self.device,
                                 batch_size=self.batch_size)
        
        ## Back Translation MODEL
        self.bn2en = pipeline(model="csebuetnlp/banglat5_nmt_bn_en",
                              use_fast=False,
                              task='translation',
                              device=self.device,
                              max_length=512,
                              batch_size=self.batch_size)
        
        self.en2bn = pipeline(model="csebuetnlp/banglat5_nmt_en_bn",
                              use_fast=False,
                              task='translation',
                              device=self.device,
                              max_length=512,
                              batch_size=self.batch_size)
        
        ## Text Generation MODEL
        self.TG = pipeline(model="csebuetnlp/banglat5_banglaparaphrase",
                           task="text2text-generation",
                           use_fast=False,
                           device=self.device,
                           batch_size=self.batch_size)
        print('\n'+'='*25+'MODEL ARE LOADED'+'='*25)
        
    def Unmasking(self, data, iters=1):
        sentences = []
        for i in range(iters):
            for idx in range(len(data)):
                try:
                    text=data[idx]
                    tokens = text.split()
                    token = tokens[random.choice(range(len(tokens)))]
                    rep_text = text.replace(token, MASK_TOKEN)
                    aug_text = self.unmasker(rep_text)

                    sentences.append(aug_text[0]['sequence'])
                except:
                    sentences.append(text)
        return {'sentence' : sentences}
    
    def BackTranslation(self, data, iters=1):
        sentences = []

        for i in tqdm(range(iters)):
            en_list=[d['translation_text'] for d in self.bn2en(data)]
            bn_list=[line['translation_text'] for line in self.en2bn(en_list)]
            sentences = sentences + (bn_list)
        return {'sentence' : sentences}
    
    def ParaPhrase(self, data, iters=1):
        sentences = []

        for i in tqdm(range(iters)):
            para=self.TG(data)
            for lines in para:
                sentences.append(lines['generated_text'])
        return {'sentence' : sentences}
    
    def test():
        test_data=['পাশে অবস্থিত একটি সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।',
            'সড়কটি অপর অঙ্গরাজ্য সড়ক ৭৯ হতে উদ্ভুত হয়ে বাক-আই হ্রদের সমান্তরালে থেকে পুনরায় একই সড়কে মিশেছে।',
            'এসআর ৩৬০ সড়কের বেশিরভাগ অংশই ফেয়ারফিল্ড কাউন্টিতে, পাশাপাশি লিকিং কাউন্টিতেও এর কিছু অংশ রয়েছে।',
            'এটি বাকআই হ্রদের উত্তর তীরের একটি অংশের সাথে সমান্তরালে']

        warnings.filterwarnings('ignore')
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Running on {device}")
        augs = AugmentationBangla(device=device)

        print("\nTesting Unmasking:")
        print(augs.Unmasking(test_data,iters=3))
        print("Unmasking Done Well.")

        print("\nTesting Back Translation:")
        print(augs.BackTranslation(test_data,iters=3))
        print("Back Translation Done Well.")
        
        print("\nTesting Paraphrase:")
        print(augs.ParaPhrase(test_data,iters=3))
        print("Paraphrase Done Well.")
    
if __name__=="__main__":
    AugmentationBangla.test()