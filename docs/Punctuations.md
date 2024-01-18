## Bangla Punctuation Generator
The package has one punctuation generation model for Bangla. The model was used from <a href='https://www.kaggle.com/datasets/tugstugi/bengali-ai-asr-submission/data'> this</a> notebook. I currently have this model in my huggingface for ease of use without any token. You can replace with any model of your like if you want.

#### Use:

````
from banglanlptoolkit import BanglaPunctuation

punct_agent = BanglaPunctuation()
print(punct_agent.add_punctuation(raw_text = 'আমার নাম কবির আপনাকে ধন্যবাদ আমার প্যাকেজ ব্যবহার করার জন্য'))
````