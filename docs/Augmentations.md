## Bangla Text Augmentation
The package uses three kind of text augmentation techniques. 
- Bangla Token Replacement
- Back Translation
- Bangla Paraphrasing

The token replacement method uses fill-mask method to replace random tokens from a sentence and then replace them. The package uses BanglishBERT Generator model by CSEBUETNLP for this task. The model can be found in <a href='https://huggingface.co/csebuetnlp/banglishbert_generator'> here</a>.

The back translation method translates the sentences from Bangla to English and then to Bangla again. The package uses bn-en and en-bn models of BanglaT5 by CSEBUETNLP for this task. The models can be found here: <a href='https://huggingface.co/csebuetnlp/banglat5_nmt_bn_en'>bn2en</a>, <a href='https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn'>en2bn</a>.

The paraphrasing toolkit uses Bangla paraphrase model of BanglaT5 by CSEBUETNLP. The model can be found in <a href='https://huggingface.co/csebuetnlp/banglat5_banglaparaphrase'>here</a>.

## Online Augmentation
First, initialize the Compose class with augmentation or transformation techniques. You can access the forward method of the class compose with following code as well.

````
import banglanlptoolkit.BanglaTextMentation as BTM

augmentations = BMT.Compose([
                   BMT.Normalize(allow_en=False, punct_replacement_token=''),
                   BMT.BackTranslation(probability=0.5),
                   BMT.TokenReplacement(probability=0.5),
                   BMT.ParaPhrase(probability=0.5),
                ])

augmentations.forward(text)
````
Define your custom PyTorch dataset and initialize it with the initialized Compose class.
````
from torch.utils.data import Dataset
import pandas as pd

class dataset(Dataset):
    def __init__(self, df, augs):
        self.df = df
        self.augs = augs
        
    def __len__(self):
        return len(self.df.text)
    
    def __getitem__(self, index):
        text = self.df.text[index]
        return augs.forward(text)
    
dummy_df = pd.DataFrame.from_records({'text'
            :['পাশে অবস্থিত একটি সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।',
            'সড়কটি অপর অঙ্গরাজ্য সড়ক ৭৯ হতে উদ্ভুত হয়ে বাক-আই হ্রদের সমান্তরালে থেকে পুনরায় একই সড়কে মিশেছে।',
            'এসআর ৩৬০ সড়কের বেশিরভাগ অংশই ফেয়ারফিল্ড কাউন্টিতে, পাশাপাশি লিকিং কাউন্টিতেও এর কিছু অংশ রয়েছে।',
            'এটি বাকআই হ্রদের উত্তর তীরের একটি অংশের সাথে সমান্তরালে']})

dummy_dataset = dataset(dummy_df,augmentations)
````
You can use DataLoader as well.

````
from torch.utils.data import DataLoader

dummy_loader = DataLoader(dummy_dataset, batch_size=10, num_workers=4, pin_memory=False)
````
The online augmentation technique utilizes several open-source deep learning models for generating augmented text data. This arises the need for loading the models in a computation device (CPU or GPU). By default, I've used CPU in the backend of the package, since, one might need the GPU memory for training. However, the models are still loaded in RAM and you might run out of it during training. I recommend using <b>pin_memory = False</b> in the DataLoader to make it easier during training. Meanwhile, I'll try my best to come up with a more simple solution. Thank you.

## Offline Augmentation
````
from banglanlptoolkit.BanglaAugmentation import AugmentationBangla
augmentations = AugmentationBangla()

test_data=['পাশে অবস্থিত একটি সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।',
            'সড়কটি অপর অঙ্গরাজ্য সড়ক ৭৯ হতে উদ্ভুত হয়ে বাক-আই হ্রদের সমান্তরালে থেকে পুনরায় একই সড়কে মিশেছে।',
            'এসআর ৩৬০ সড়কের বেশিরভাগ অংশই ফেয়ারফিল্ড কাউন্টিতে, পাশাপাশি লিকিং কাউন্টিতেও এর কিছু অংশ রয়েছে।',
            'এটি বাকআই হ্রদের উত্তর তীরের একটি অংশের সাথে সমান্তরালে']

augmentations.Unmasking(test_data)
augmentations.BackTranslation(test_data)
augmentations.ParaPhrase(test_data)
````

## Bangla Sequence Classification and Sequence to Sequence Data Augmentation
By using the methods mentioned and explained above, both sequence classification and sequence to sequence augmentation toolkit takes a dataframe as input and returns a dictionary of augmented data.

#### Use:
````
from banglanlptoolkit import SequenceClassificationAug
seq2seq = SequenceClassificationAug(allow_en=True, translate_en=False, punct_replacement_token=None)
seq2seq = Seq2SeqAug(allow_en=True,translate_en=False,punct_replacement_token=None)
````

The attributes allow_en and translate_en are used during normalization and punct_replacement allows the user to replace punctuations to any character of his choice. If set to None, the punctuations will not be replaced at all.

For sequence classification augmentation use like this.
````
import pandas as pd

test_data=pd.DataFrame({
    'sentence':['পাশে অবস্থিত একটি সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।',
                'সড়কটি অপর অঙ্গরাজ্য সড়ক ৭৯ হতে উদ্ভুত হয়ে বাক-আই হ্রদের সমান্তরালে থেকে পুনরায় একই সড়কে মিশেছে।',
                'এসআর ৩৬০ সড়কের বেশিরভাগ অংশই ফেয়ারফিল্ড কাউন্টিতে, পাশাপাশি লিকিং কাউন্টিতেও এর কিছু অংশ রয়েছে।',
                'এটি বাকআই হ্রদের উত্তর তীরের একটি অংশের সাথে সমান্তরালে'],
    'label':[0,1,2,3]})


seq2seq.BnAugSeqClassification(df=test_data,iters=1)
````
For sequence to sequence augmentation use like this.
````
test_data=pd.DataFrame({
    'sentence1':['পাশে অবস্থিত একটি সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।',
                'সড়কটি অপর অঙ্গরাজ্য সড়ক ৭৯ হতে উদ্ভুত হয়ে বাক-আই হ্রদের সমান্তরালে থেকে পুনরায় একই সড়কে মিশেছে।',
                'এসআর ৩৬০ সড়কের বেশিরভাগ অংশই ফেয়ারফিল্ড কাউন্টিতে, পাশাপাশি লিকিং কাউন্টিতেও এর কিছু অংশ রয়েছে।',
                'এটি বাকআই হ্রদের উত্তর তীরের একটি অংশের সাথে সমান্তরালে'],
            
    'sentence2':['পাশে অবস্থিত একটি সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।',
                'সড়কটি অপর অঙ্গরাজ্য সড়ক ৭৯ হতে উদ্ভুত হয়ে বাক-আই হ্রদের সমান্তরালে থেকে পুনরায় একই সড়কে মিশেছে।',
                'এসআর ৩৬০ সড়কের বেশিরভাগ অংশই ফেয়ারফিল্ড কাউন্টিতে, পাশাপাশি লিকিং কাউন্টিতেও এর কিছু অংশ রয়েছে।',
                'এটি বাকআই হ্রদের উত্তর তীরের একটি অংশের সাথে সমান্তরালে']
                })

seq2seq.BnAugSeq2Seq(df=test_data,iters=1)
````

<b> If you use this module, please mention the following papers: </b>
````
@article{bhattacharjee2022banglanlg,
  author    = {Abhik Bhattacharjee and Tahmid Hasan and Wasi Uddin Ahmad and Rifat Shahriyar},
  title     = {BanglaNLG: Benchmarks and Resources for Evaluating Low-Resource Natural Language Generation in Bangla},
  journal   = {CoRR},
  volume    = {abs/2205.11081},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.11081},
  eprinttype = {arXiv},
  eprint    = {2205.11081}
}
````
````
@inproceedings{bhattacharjee-etal-2022-banglabert,
    title     = {BanglaBERT: Lagnuage Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla},
    author = "Bhattacharjee, Abhik  and
      Hasan, Tahmid  and
      Mubasshir, Kazi  and
      Islam, Md. Saiful  and
      Uddin, Wasi Ahmad  and
      Iqbal, Anindya  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
      booktitle = "Findings of the North American Chapter of the Association for Computational Linguistics: NAACL 2022",
      month = july,
    year      = {2022},
    url       = {https://arxiv.org/abs/2101.00204},
    eprinttype = {arXiv},
    eprint    = {2101.00204}
}
````
````
@article{akil2022banglaparaphrase,
  title={BanglaParaphrase: A High-Quality Bangla Paraphrase Dataset},
  author={Akil, Ajwad and Sultana, Najrin and Bhattacharjee, Abhik and Shahriyar, Rifat},
  journal={arXiv preprint arXiv:2210.05109},
  year={2022}
}
````