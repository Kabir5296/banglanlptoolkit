from .BanglaAugmentation import AugmentationBangla
import torch
from .BnNLPNormalizer import BnNLPNormalizer

class Seq2SeqAug():
    def __init__(self,allow_en=False,translate_en=False, punct_replacement_token=None,device=None):
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.augs = AugmentationBangla(device=self.device)
        self.normalizer = BnNLPNormalizer(allow_en=allow_en,
                                          translate_en=translate_en,
                                          device=self.device)
        self.punct_replacement_token=punct_replacement_token

    def BnAugSeq2Seq(self, df, iters=1, do_unmask=True, do_BackTrans=True, do_Para=True):
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