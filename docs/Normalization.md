## Bangla Text Normalizer
The package uses two normalization toolkits for Bangla text processing. The unicode normalizer is used from <a href='https://github.com/mnansary/bnUnicodeNormalizer'> here</a>. The other normalizer is specifically used for BanglaT5 translation module and taken from <a href='https://github.com/csebuetnlp/normalizer'> here</a>.

#### Use:
The BnNLPNormalizer module was developed for normalization of multiple texts at once. The module takes a list of sentences as input and returns the normalized sentences as a list of strings as well.

````
from banglanlptoolkit import BnNLPNormalizer
normalizer = BnNLPNormalizer()

normalizer.normalize_bn(['পাশে অবস্থিত সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।'])
````

To allow English, change the code as below. By default, the normalizer module deletes any English words or pronunciations present. You can also set the module to translate English words to Bengali by changing translate_en attribute to True.

````
normalizer = BnNLPNormalizer(allow_en=True, translate_en=True)
````

You can also use a separate Normalize class for a single text string, if convenient. The class was originally introduced to be used in augmentations but since it's already here, why not use it?

````
import banglanlptoolkit.BanglaTextMentation as BTM

normalizer = BTM.Normalize(allow_en = False, punct_replacement_token = '')

normalize.forward('পাশে অবস্থিত সংক্ষিপ্ত পূর্ব-পশ্চিম অভিমুখি অনিয়মিত অর্ধবৃত্তাকার সড়ক।')
````

<b> If you use this module, please mention the following papers: </b>
````
@inproceedings{alam2021large,
  title={A large multi-target dataset of common bengali handwritten graphemes},
  author={Alam, Samiul and Reasat, Tahsin and Sushmit, Asif Shahriyar and Siddique, Sadi Mohammad and Rahman, Fuad and Hasan, Mahady and Humayun, Ahmed Imtiaz},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={383--398},
  year={2021},
  organization={Springer}
}
````

````
@inproceedings{hasan-etal-2020-low,
    title = "Not Low-Resource Anymore: Aligner Ensembling, Batch Filtering, and New Datasets for {B}engali-{E}nglish Machine Translation",
    author = "Hasan, Tahmid  and
      Bhattacharjee, Abhik  and
      Samin, Kazi  and
      Hasan, Masum  and
      Basak, Madhusudan  and
      Rahman, M. Sohel  and
      Shahriyar, Rifat",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.207",
    doi = "10.18653/v1/2020.emnlp-main.207",
    pages = "2612--2623"
}
````