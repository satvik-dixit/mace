import os
import re
import json
import numpy as np
import torch
from tqdm import trange
from .model import BERTFlatClassifier
from .data import infer_preprocess
from functools import lru_cache
# from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers import logging as trf_logging
import transformers
from msclap import CLAP

PRETRAIN_ECHECKERS = {
    'echecker_clotho_audiocaps_base': ("https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt", "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa"),
    'echecker_clotho_audiocaps_tiny': ("https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt", "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673"),
    "none": (None, None)
}

def _use_new_echecker_loading() -> bool:
    version = transformers.__version__
    major, minor, _patch = map(int, version.split("."))
    return major > 4 or (major == 4 and minor >= 31)

def load_pretrain_echecker(echecker_model, device='cuda', use_proxy=False, proxies=None):
    from .download_utils import RemoteFileMetadata, check_download_resource
    trf_logging.set_verbosity_error()  # suppress loading warnings
    url, checksum = PRETRAIN_ECHECKERS[echecker_model]
    remote = RemoteFileMetadata(
        filename=f'{echecker_model}.ckpt',
        url=url,
        checksum=checksum)
    file_path = check_download_resource(remote, use_proxy, proxies)
    model_states = torch.load(file_path)

    state_dict = model_states['state_dict']
    if _use_new_echecker_loading():
        state_dict.pop("encoder.embeddings.position_ids")

    clf = BERTFlatClassifier(model_type=model_states['model_type'], num_classes=model_states['num_classes'])
    clf.load_state_dict(model_states['state_dict'])
    clf.eval()
    clf.to(device)
    return clf


class Evaluator:
    def __init__(self, batch_size=32, device='cuda', clap_model="MS-CLAP", echecker_model="echecker_clotho_audiocaps_base", error_threshold=0.97, penalty=0.3, use_proxy=False, proxies=None):
        # assert sbert_model in {'paraphrase-MiniLM-L6-v2', 'paraphrase-TinyBERT-L6-v2', 'paraphrase-mpnet-base-v2'}
        assert clap_model=="MS-CLAP"
        clap_model = CLAP(version='2023', use_cuda=True if device=='cuda' else 'cpu') 

        assert echecker_model in PRETRAIN_ECHECKERS
        self.batch_size = batch_size
        self.device = device
        self.clap_model = clap_model
        self.echecker_model = echecker_model
        self.error_threshold = error_threshold
        self.penalty = penalty

        if echecker_model != "none":
            self.echecker = load_pretrain_echecker(echecker_model, device, use_proxy, proxies)
            self.echecker_tokenizer = AutoTokenizer.from_pretrained(self.echecker.model_type)
            self.echecker.to(device)
            self.echecker.eval()

    def encode_sents_clap(self, sents, batch_size=32):
        # return self.sbert.encode(sents, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
        # return torch.stack([self.clap_model.get_text_embeddings([sent]).to('cuda') for sent in sents], dim=0).squeeze()
        return self.clap_model.get_text_embeddings(sents).to('cuda') 

    def encode_audios_clap(self, audio_paths):
        audio_embs = torch.stack([self.clap_model.get_audio_embeddings([audio_path]).to('cuda') for audio_path in audio_paths], dim=0)
        return audio_embs
    
    @lru_cache(maxsize=32)   # reuse cache if encode the same sentence
    def encode_sent_clap(self, sent):
        # return self.sbert.encode(sent, convert_to_tensor=True, normalize_embeddings=True)
        return self.clap_model.get_text_embeddings(sent).to('cuda') 
    
    def encode_audio_clap(self, audio_path):
        # audio_embs = torch.stack([self.clap_model.get_audio_embeddings([audio_file]).to('cuda') for audio_file in audio_files], dim=0)
        return self.clap_model.get_audio_embeddings([audio_path]).to('cuda')


    def detect_error_sents(self, sents, batch_size=32):
        if len(sents) <= batch_size:
            batch = infer_preprocess(self.echecker_tokenizer, sents, max_len=64)
            for k, v in batch.items():
                batch[k] = v.to(self.device)
            with torch.no_grad():
                logits = self.echecker(**batch)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
        else:
            probs = []
            for i in trange(0, len(sents), batch_size):
                batch = infer_preprocess(self.echecker_tokenizer, sents[i:i+batch_size], max_len=64)
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                with torch.no_grad():
                    batch_logits = self.echecker(**batch)
                    batch_probs = torch.sigmoid(batch_logits).detach().cpu().numpy()[:, -1]
                probs.append(batch_probs)
            probs = np.concatenate(probs)
        return (probs > self.error_threshold).astype(float)

    @lru_cache(maxsize=32)   # reuse cache if infer with the same sentence
    def detect_error_sent(self, sent, return_error_prob=False):
        batch = infer_preprocess(self.echecker_tokenizer, [sent], max_len=64)
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        with torch.no_grad():
            logits = self.echecker(**batch)
            # probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs = logits.sigmoid().transpose(0, 1).cpu().numpy()
        has_error = probs[0, -1] > self.error_threshold
        if return_error_prob:
            return has_error, probs[0, -1]
        else:
            return has_error 

    def corpus_score(self, cands, list_refs=None, audio_paths=None, config='text-text', agg_score='mean'):
        assert config in ['combined', 'audio-text', 'text-text']
        assert agg_score in {'none', 'mean', 'max'}

        if config=='text-text':
            assert len(cands)==len(list_refs)
            rng_ids = [0]
            all_refs = []
            for lst in list_refs:
                rng_ids.append(rng_ids[-1]+len(lst))
                all_refs.extend(lst)
            emb_cands = self.encode_sents_clap(cands, self.batch_size)
            emb_refs = self.encode_sents_clap(all_refs, self.batch_size)
            sim_scores = [(emb_cands[i] @ emb_refs[rng_ids[i]:rng_ids[i+1]].T).mean().detach().cpu().item() for i in range(len(cands))]
        elif config=='audio-text':
            assert len(cands)==len(audio_paths)
            emb_cands = self.encode_sents_clap(cands, self.batch_size)
            emb_audios = self.encode_audios_clap(audio_paths)
            sim_scores = [(emb_cands[i] @ emb_audios[i].T).mean().detach().cpu().item() for i in range(len(cands))]
        elif config=='combined':
            assert len(cands)==len(list_refs)
            assert len(cands)==len(audio_paths)
            rng_ids = [0]
            all_refs = []
            for lst in list_refs:
                rng_ids.append(rng_ids[-1]+len(lst))
                all_refs.extend(lst)
            emb_cands = self.encode_sents_clap(cands, self.batch_size)
            emb_refs = self.encode_sents_clap(all_refs, self.batch_size)
            text_sim_scores = [(emb_cands[i] @ emb_refs[rng_ids[i]:rng_ids[i+1]].T).mean().detach().cpu().item() for i in range(len(cands))]

            emb_audios = self.encode_audios_clap(audio_paths)
            audio_sim_scores = [(emb_cands[i] @ emb_audios[i].T).mean().detach().cpu().item() for i in range(len(cands))]

            sim_scores = text_sim_scores + audio_sim_scores

        if self.echecker_model == "none":
            if agg_score == 'mean':
                return np.mean(sim_scores)
            elif agg_score == 'max':
                return np.max(sim_scores)
            else:
                return sim_scores
        else:
            sim_scores = np.array(sim_scores)
            print("Performing error detection")
            has_error = self.detect_error_sents(cands, self.batch_size)
            penalized_scores = sim_scores * (1-self.penalty*has_error)
            if agg_score == 'mean':
                return np.mean(penalized_scores)
            elif agg_score == 'max':
                return np.max(penalized_scores)
            else:
                return penalized_scores

    def sentence_score(self, cand, refs=None, audio_path=None, config='text-text', return_error_prob=False):
        assert config in ['combined', 'audio-text', 'text-text']
        
        emb_cand = self.encode_sent_clap(cand)
        if config=='audio-text':
            emb_refs_audio = self.encode_audio_clap(audio_path)
            scores = emb_cand @ emb_refs_audio.T

        elif config=='text-text':
            emb_refs_text = self.encode_sents_clap(refs, self.batch_size)
            scores = emb_cand @ emb_refs_text.T

        elif config=='combined':
            emb_refs_audio = self.encode_audio_clap(audio_path)
            emb_refs_text = self.encode_sents_clap(refs, self.batch_size)
            scores = emb_cand @ emb_refs_audio.T + emb_cand @ emb_refs_text.T
        
        if self.echecker_model == "none":
            return scores.mean().detach().cpu().item()
        else:
            score = scores.mean().detach().cpu().item()
            if not return_error_prob:
                has_error = self.detect_error_sent(cand)
                penalized_score = (1-self.penalty)*score if has_error else score
                return penalized_score
            else:
                has_error, error_prob = self.detect_error_sent(cand, return_error_prob)
                penalized_score = (1-self.penalty)*score if has_error else score
                return score, error_prob, penalized_score

if __name__ == "__main__":
    evaluator = Evaluator(device='cuda', clap_model='MS-CLAP', echecker_model='echecker_clotho_audiocaps_tiny')
