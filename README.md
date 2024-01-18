## Bangla NLP Toolkit
Created by <b>A F M Mahfuzul Kabir</b> \
<a href='https://mahfuzulkabir.com'>mahfuzulkabir.com</a> \
https://www.linkedin.com/in/mahfuzulkabir 

## Installation
Install the requirements first with:
````
pip install -r requirements.txt
````

install the package with

````
pip install banglanlptoolkit
````
## Introduction
This package contains several toolkits for Bangla NLP text processing and augmentation. The available tools are listed below.

- Bangla Text Normalizer
- Bangla Punctuation Generator
- Bangla Text Augmentation

## Documentations:
- For detailed use of Bangla Text Normalizer, follow [this documentation](https://github.com/Kabir5296/banglanlptoolkit/blob/main/docs/Normalization.md).
- For detailed use of Bangla Punctuation Generation, follow [this documentation](https://github.com/Kabir5296/banglanlptoolkit/blob/main/docs/Punctuations.md).
- For detailed use of Bangla Text Augmentation (both online and offline), follow [this documentation](https://github.com/Kabir5296/banglanlptoolkit/blob/main/docs/Augmentations.md).

Thank you very much for using my package. I handle this package all on my own, so if there's any issue with it, I might not always be available to fix it. But if you do encounter such event, feel free to let me know and I'll fix them as soon as I can.

## Bangla Text Normalizer
Bangla text normalization is a known problem in language processing for normalizing Bangla text data in computer readable format. The unicode normalization normalizes all characters of a text string in the same unicode format and removes unwanted characters present. The csebuetnlp normalizer is used for models such as BanglaBERT, BanglaT5 etc.

The package uses two normalization toolkits for Bangla text processing. The unicode normalizer is used from <a href='https://github.com/mnansary/bnUnicodeNormalizer'> here</a>. The other normalizer is specifically used for BanglaT5 translation module and taken from <a href='https://github.com/csebuetnlp/normalizer'> here</a>.

## Bangla Punctuation Generator
The scarcity of good punctuation generator model for Bangla language was very dominant even a few months ago. However, with development of Bangla AI models, we now have very good punctuation generation models for our language as well. 

The package uses an open-source punctuation generation model from <a href='https://www.kaggle.com/datasets/tugstugi/bengali-ai-asr-submission/data'> this</a> Kaggle dataset. I currently have this model in my huggingface for ease of use without any token. You can replace with any model of your like if you want.

## Bangla Text Augmentation
The package uses three kind of text augmentation techniques. 
- Bangla Token Replacement
- Back Translation
- Bangla Paraphrasing

The token replacement method uses fill-mask method to replace random tokens from a sentence and then replace them. The package uses BanglishBERT Generator model by CSEBUETNLP for this task. The model can be found in <a href='https://huggingface.co/csebuetnlp/banglishbert_generator'> here</a>.

The back translation method translates the sentences from Bangla to English and then to Bangla again. The package uses bn-en and en-bn models of BanglaT5 by CSEBUETNLP for this task. The models can be found here: <a href='https://huggingface.co/csebuetnlp/banglat5_nmt_bn_en'>bn2en</a>, <a href='https://huggingface.co/csebuetnlp/banglat5_nmt_en_bn'>en2bn</a>.

The paraphrasing toolkit uses Bangla paraphrase model of BanglaT5 by CSEBUETNLP. The model can be found in <a href='https://huggingface.co/csebuetnlp/banglat5_banglaparaphrase'>here</a>.

The package supports both online and offline augmentations. Offline augmentation can be used to generate new dataframe of augmented texts from original dataframe. This can be saved in a variable or to a file for later use. While offline augmentation can be faster for utilizing processing power (GPU parallelism), it can get a bit annoying because of saving the augmented data every once in a while. People also love to use online augmentation, meaning, augmenting the data 'on the fly' in predefined custom dataset class. This improves performance by augmentation of sentences during training or inference, with no hassle of saving the data separately.

From <b>version 1.1.5</b>, I'm happy to introduce online augmentation techniques in this package. This technique was inspired from the exact same technique of <b>torchvision.transpose</b>, meaning, you can stack several augmentation techniques with a <b>compose</b> class. You can also write your own custom class of augmentation or transform techniques and use them with <b>compose</b>.

## Inspired from
- <a href='https://amitness.com/2020/05/data-augmentation-for-nlp/'>A Visual Survey of Data Augmentation in NLP</a>
- <a href='https://huggingface.co/csebuetnlp'>CSE BUET NLP</a>
- <a href='https://github.com/mnansary/bnUnicodeNormalizer'>Bangla Unicode Normalizer by Bengali Ai</a>
- <a href='https://github.com/sagorbrur/bnaug'>Bangla Text Augmentation </a>

<b>If you use this package, please don't forget to cite the links and papers mentioned.</b> 
