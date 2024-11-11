import torch
import numpy as np
from .evaluator import Evaluator


class MACE:
    def __init__(self,
                 CLAP_model="MS-CLAP",
                 echecker_model="echecker_clotho_audiocaps_base",
                 penalty=0.3) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluator = Evaluator(device=device, CLAP_model=CLAP_model,
            echecker_model=echecker_model, penalty=penalty)
        
    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        keys = list(gts.keys())
        list_cand = [res[key][0] for key in keys]
        list_refs = [gts[key] for key in keys]
        scores = self.evaluator.corpus_score(list_cand, list_refs, agg_score="none")
        average_score = np.mean(np.array(scores))
        return average_score, np.array(scores)

    def method(self):
        return "MACE"
