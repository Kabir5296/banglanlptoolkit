from .BanglaAugmentation import AugmentationBangla
import torch
from .BnNLPNormalizer import BnNLPNormalizer

class SequenceClassificationAug():
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

    def BnAugSeqClassification(self, df, iters=1, do_unmask=True, do_BackTrans=True, do_Para=True):
        sentences = []
        labels = []

        sentence_truth = self.normalizer.normalize_bn(df['sentence'].tolist(),punct_replacement_token=self.punct_replacement_token)
        label_truth = df['label'].tolist()
        
        if do_unmask:
            sentences = sentences + self.augs.Unmasking(data=sentence_truth, iters=iters)['sentence']
            for i in range(iters):
                labels = labels + label_truth
        if do_BackTrans:
            sentences = sentences + self.augs.BackTranslation(data=sentence_truth,iters=iters)['sentence']
            for i in range(iters):
                labels = labels + label_truth
        if do_Para:
            sentences = sentences + self.augs.ParaPhrase(data=sentence_truth,iters=iters)['sentence']
            for i in range(iters):
                labels = labels + label_truth

        sentences = self.normalizer.normalize_bn(sentences,punct_replacement_token=self.punct_replacement_token)

        return {'sentence' : sentences, 'label' : labels}