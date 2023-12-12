## Bangla NLP Toolkit
Created by <b>A F M Mahfuzul Kabir</b> \
<a href='mahfuzulkabir.com'>mahfuzulkabir.com</a> \
https://www.linkedin.com/in/mahfuzulkabir \

## Installation

install the package with

````
pip install banglanlptoolkit
````

## Introduction
This package contains several toolkits for Bangla NLP text processing and augmentation. The available tools are listed below.

- Bangla Text Normalizer
- Bangla Text Augmentation

## Bangla Text Normalizer
The package uses two normalization toolkits for Bangla text processing. The unicode normalizer is used from <a href='https://github.com/mnansary/bnUnicodeNormalizer'> here</a>. The other normalizer is specifically used for BanglaT5 translation module and taken from <a href='https://github.com/csebuetnlp/normalizer'> here</a>.

#### Use:
````
from banglanlptoolkit import BnNLPNormalizer
normalizer = BnNLPNormalizer()

normalizer.normalize_bn(['পাশে অবস্থিত সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।'])
````

You can also use only the unicode normalizer
````
from banglanlptoolkit import BnNLPNormalizer
normalizer = BnNLPNormalizer()

normalizer.unicode_normalize(['পাশে অবস্থিত সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।'])
````

To allow English, change the code as below. By default, the normalizer module deletes any English words or pronunciations present. You can also set the module to translate English words to Bengali by changing translate_en attribute to True.

````
normalizer = BnNLPNormalizer(allow_en=True, translate_en=True)
````

## Bangla Text Augmentation
The package uses three kind of text augmentation techniques. 
- Bangla Token Replacement
- Back Translation
- Bangla Paraphrasing

The token replacement method uses fill-mask method to replace random tokens from a sentence and then replace them. The package uses BanglishBERT Generator model by CSEBUETNLP for this task. The model can be found in <a href='https://huggingface.co/csebuetnlp/banglishbert_generator'> here</a>.

The back translation method translates the sentences from Bangla to English and then to Bangla again. The package uses bn-en and en-bn models of BanglaT5 by CSEBUETNLP for this task. The models can be found here: <a href='https://huggingface.co/csebuetnlp/banglat5_nmt_bn_en'>bn2en</a>, <a href='https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn'>en2bn</a>.

The paraphrasing toolkit uses Bangla paraphrase model of BanglaT5 by CSEBUETNLP. The model can be found in <a href='https://huggingface.co/csebuetnlp/banglat5_banglaparaphrase'>here</a>.

#### Use:
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

## Inspired from
- <a href='https://amitness.com/2020/05/data-augmentation-for-nlp/'>A Visual Survey of Data Augmentation in NLP</a>
- <a href='https://huggingface.co/csebuetnlp'>CSE BUET NLP</a>
- <a href='https://github.com/mnansary/bnUnicodeNormalizer'>Bangla Unicode Normalizer by Bengali Ai</a>
- <a href='https://github.com/sagorbrur/bnaug'>Bangla Text Augmentation </a>

<b>If you use this package, please don't forget to cite the links and papers mentioned.</b> 
