from banglanlptoolkit import BnNLPNormalizer
from tqdm import tqdm
from pandarallel import pandarallel
from pqdm.processes import pqdm
import pandas as pd
import json, os, gc
from time import time
pandarallel.initialize(progress_bar=True)

class BnNLPNormalizerPlus():
    def __init__(self,allow_en = False):
        # Initialize normalizer
        self.bnorm = BnNLPNormalizer(allow_en=allow_en)
        
    def total_num_lines(self, file_path):
        # Calculate total number of lines present in the corpus
        with open(file_path,'rbU') as f:
            num_lines = sum(1 for _ in tqdm(f))
        print(f'The total number of lines in "{file_path}" is: {num_lines}')
        return num_lines
    
    def word_freq_dict(self, file_path, num_lines):
        # Create a dictionary of all unique words with their respective frequency in the corpus
        word_dict = {}
        with open(file_path,'r') as f:
            for _, line in tqdm(enumerate(f),total = num_lines):
                for word in line.split():
                    if word in word_dict:
                        word_dict[word] +=1
                    else:
                        word_dict[word] = 1
        print(f'Total number of unique words in the corpus: {len(word_dict)}')
        return word_dict
    
    def normalized_words_dict_fn(self, unique_words_dict):
        # Create a dictionary of normalized dictionary. It contains keys of unnormalized words and values as the normalized versions of them.
        print('Creating new unique word normalized dictionary.')
        
        word_df = pd.DataFrame(pd.Series(unique_words_dict)).reset_index(drop=False).rename(columns={'index':'words',0:'freq'})
        normalized_words = word_df.words.parallel_apply(lambda word: self.bnorm.word_normalize(word)).tolist()
        normalized_word_dict = {}
        
        for word, normalized_word in zip(unique_words_dict, normalized_words):
            normalized_word_dict[word] = normalized_word if normalized_word is not None else ''
        
        print(f'Normalized word dictionary length: {normalized_word_dict}')
        return normalized_word_dict
        
    def __call__(self, file_path, normalized_word_dict = None):
        # Get some necessary variables
        start_time = time()
        root_folder = os.path.split(file_path)[:-1][0]
        file_name = os.path.split(file_path)[-1][:-4]
        # Calculate total number of lines present in the corpus
        num_lines = self.total_num_lines(file_path)
        # Get the word frequency dictionary
        unique_words_dict = self.word_freq_dict(file_path, num_lines)        
        
        # If not using a predefined normalization dictionary (can be loaded manually), then create a normalization dictionary by itself from the unique word frequency dictionary.
        
        if normalized_word_dict is None:
            normalized_word_dict = self.normalized_words_dict_fn(unique_words_dict)
        elif os.path.isfile(normalized_word_dict):
            with open(normalized_word_dict,'r') as f:
                normalized_word_dict = json.load(f)
        elif isinstance(normalized_word_dict, dict):
            print(f'Length of passed dictionary is {len(normalized_word_dict)}')
        else:
            raise TypeError('Please pass a valid normalized dictionary or None.')
            
        
        save_file = os.path.join(root_folder,(file_name+'normalized.txt'))
        
        # Write a new text file with normalized texts of the original corpus.
        with open(save_file,'w') as f1:
            with open(file_path,'r') as f2:
                for _, line in tqdm(enumerate(f2), total = num_lines):
                    for word in line.split():
                        # For each word, check if the normalized dictionary contains the word or not. 
                        # If it does, simply replace the original word with normalized one. If not, the word will be normalized manually.
                        if word in normalized_word_dict and normalized_word_dict[word] is not None:
                            f1.write(normalized_word_dict[word]+' ')
                        elif word not in normalized_word_dict:
                            f1.write(self.bnorm.word_normalize(word) + ' ')
                    f1.write('\n')
        end_time = time()
        print(f'File saved at {save_file}')
        print(f"The whole process took {int(end_time - start_time)/60} minutes. It'd probably have taken you a lot longer for typical normalization.")