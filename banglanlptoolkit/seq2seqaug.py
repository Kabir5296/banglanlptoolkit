from .banglaaugmentation import AugmentationBangla
import torch, pandas
from .bnnlpnormalizer import BnNLPNormalizer

class Seq2SeqAug():
    """
    Augmentation for Bangla NLP sequence to sequence generation task. Only can be used for offline augmentation. Online augmentation feature will be added soon.

    Arguements:
    -----------
        allow_en (bool, optional): Allow English words existing in a sentence. If true, the unicodenormalizer won't delete english words existing in a sentence. Defaults to False.
        translate_en (bool, optional): Whether to translate english sentences to Bangla. If set to true and allow_en is also set to true, the english sentences/words will be translated to Bangla. Defaults to False.
        punct_replacement_token (_type_, optional):The character or string to replace punctuations with. If set to None, the punctuations will not be removed. Defaults to None.
        device (Any, optional): The device to use for the deep learning models used in this library. If not defined, the code will automatically detect available device and set to GPU if possible. Defaults to None.
    """
    def __init__(self, allow_en: bool = False, translate_en: bool = False, punct_replacement_token: bool = None, device: any = None):
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.augs = AugmentationBangla(device=self.device)
        self.normalizer = BnNLPNormalizer(allow_en=allow_en,
                                          translate_en=translate_en,
                                          device=self.device)
        self.punct_replacement_token=punct_replacement_token

    def BnAugSeq2Seq(self, df: any, iters: int = 1, do_unmask: bool = True, do_BackTrans: bool = True, do_Para: bool = True) -> dict:
        """
        Return a dictionary of augmented sentences for Bangla sequence to sequence generation task.
        
        Unmasking:
            Selects a random token from the given sentence and replaces it with a mask. A DL model is then used to predict the hidden word.
        Backtranslation:
            Translates a sentence from Bangla to English and then back to Bangla to generate different versions of the same sentence.
        Paraphrasing:
            Paraphrases the given sentence to generate augmented sentences.

        Arguements:
        -----------
            df (pandas dataframe): The original dataframe to use for augmentation. The dataframe should have 'sentence1' column for input sentences and 'sentence2' column for target.
            iters (int, optional): How many iterations to do while doing augmentation. Defaults to 1.
            do_unmask (bool, optional): Whether to do unmasking augmentation or not. Defaults to True.
            do_BackTrans (bool, optional): Whether to do back translation augmentation or not. Defaults to True.
            do_Para (bool, optional): Whether to do paraphrasing augmentation or not. Defaults to True.

        Returns:
        --------
            dict: A dictionary of augmented sentences.
        """
        sentences1 = []
        sentences2 = []

        sentence_truth1 = self.normalizer.normalize_bn(df['sentence1'].tolist(),punct_replacement_token=self.punct_replacement_token)
        sentence_truth2 = self.normalizer.normalize_bn(df['sentence2'].tolist(),punct_replacement_token=self.punct_replacement_token)
        
        if do_unmask:
            sentences1 = sentences1 + self.augs.Unmasking(data=sentence_truth1, iters=iters)['sentence']
            sentences2 = sentences2 + self.augs.Unmasking(data=sentence_truth2, iters=iters)['sentence']
        if do_BackTrans:
            sentences1 = sentences1 + self.augs.BackTranslation(data=sentence_truth1,iters=iters)['sentence']
            sentences2 = sentences2 + self.augs.BackTranslation(data=sentence_truth2,iters=iters)['sentence']
        if do_Para:
            sentences1 = sentences1 + self.augs.ParaPhrase(data=sentence_truth1,iters=iters)['sentence']
            sentences2 = sentences2 + self.augs.ParaPhrase(data=sentence_truth2,iters=iters)['sentence']

        sentences1 = self.normalizer.normalize_bn(sentences1,punct_replacement_token=self.punct_replacement_token)
        sentences2 = self.normalizer.normalize_bn(sentences2,punct_replacement_token=self.punct_replacement_token)

        return {'sentence1' : sentences1,'sentence2' : sentences2}