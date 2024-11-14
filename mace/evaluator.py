
import numpy as np
import torch
import os
from msclap import CLAP
from torchaudio import load, save
import torch.nn.functional as F
import tempfile
from tqdm import trange
from .model import BERTFlatClassifier
from .data import infer_preprocess
from transformers import AutoTokenizer
from transformers import logging as trf_logging
import transformers

clap_model = CLAP(version='2023', use_cuda=True)  

PRETRAIN_ECHECKERS = {
    'echecker_clotho_audiocaps_base': ("https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt", "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa"),
    'echecker_clotho_audiocaps_tiny': ("https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt", "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673"),
    "none": (None, None)
}

def cosine_similarity(input, target):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(input, target).item()

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
        clap_model = CLAP(version='2023', use_cuda=True) 
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

    # @lru_cache(maxsize=32)   # reuse cache if infer with the same sentence
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
    

    def get_chunked_audio_embedding(self, audio_path, chunk_duration=7.0, sample_rate=16000):
        waveform, sr = load(audio_path)  # Load the audio file
        sample_rate = sr

        chunk_samples = int(chunk_duration * sample_rate)  # Number of samples in each 7-second chunk
        num_chunks = (waveform.size(1) + chunk_samples - 1) // chunk_samples  # Calculate total number of chunks

        # Calculate the embedding for each chunk
        chunk_embeddings = []
        chunk_durations = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, waveform.size(1))
            chunk = waveform[:, start:end]
            chunk_duration_seconds = chunk.size(1) / sample_rate  # Calculate the actual duration of this chunk
            chunk_durations.append(chunk_duration_seconds)
            
            # Create a temporary file for the chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                save(temp_path, chunk, sample_rate=sample_rate)  
                chunk_embedding = clap_model.get_audio_embeddings([temp_path]).to('cuda')
                chunk_embeddings.append(chunk_embedding)
                os.remove(temp_path)

        # Stack embeddings and calculate weighted mean
        chunk_embeddings = torch.stack(chunk_embeddings, dim=0).squeeze()
        chunk_durations = torch.tensor(chunk_durations, device=chunk_embeddings.device)  # Convert durations to tensor
        total_duration = chunk_durations.sum()
        weights = (chunk_durations / total_duration).view(-1, 1)  # Normalize durations to get weights, shape [num_chunks, 1]

        # Compute the weighted mean of embeddings
        weighted_mean_embedding = torch.sum(chunk_embeddings * weights, dim=0).unsqueeze(0) 
        print('weighted_mean_embedding shape:', weighted_mean_embedding.shape)
        return weighted_mean_embedding

    def get_similarity_score(self, all_preds_text, all_refs_text=None, audio_files=None, method='text-text', average=True):

        N = len(all_preds_text)
        all_preds_text = np.array(all_preds_text, dtype=str)
        print('all_preds_text shape:', len(all_preds_text))

        if method=='text-text':
            K = len(all_refs_text[0]) if all_refs_text is not None else 1
            all_refs_text = np.array(all_refs_text, dtype=str)
            print('all_refs_text shape:', len(all_refs_text))
            print('all_refs_text shape:', len(all_refs_text[0]))
        elif method=='audio-text':
            K = 1

        score = torch.zeros((N, K))

        # For text-text
        if method == 'text-text':
            preds_clap = torch.stack([clap_model.get_text_embeddings([pred]).to('cuda') for pred in all_preds_text], dim=0).squeeze()
            print('preds_clap shape:', preds_clap.shape)
            refs_clap = torch.stack([clap_model.get_text_embeddings(refs).to('cuda') for refs in all_refs_text], dim=0)
            print('refs_clap shape:', refs_clap.shape)
            for i in range(K):
                score[:, i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_clap, refs_clap[:, i])])
            print('score:', score)

        # For audio-text
        elif method == 'audio-text':
            if audio_files is None:
                raise ValueError("Audio files must be provided for ms_clap_audio_caption.")
            preds_clap = torch.stack([clap_model.get_text_embeddings([pred]).to('cuda') for pred in all_preds_text], dim=0).squeeze()
            print('preds_clap shape:', preds_clap.shape)
            print('audio_files:', audio_files)
            audio_embs = torch.stack([self.get_chunked_audio_embedding(audio_file) for audio_file in audio_files], dim=0)
            print('audio_embs shape:', audio_embs.shape)
            for i in range(1):
                score[:, i] = torch.Tensor([cosine_similarity(input, target) for input, target in zip(preds_clap, audio_embs[:, i])])
            print('score:', score)
            print('score shape:', score.shape)

        elif method=='combined':
            text_score = self.get_similarity_score(all_preds_text, all_refs_text, method='text-text', average=True, audio_files=None)
            audio_score = self.get_similarity_score(all_preds_text, all_refs_text, method='audio-text', average=True, audio_files=audio_files)
            score = (text_score + audio_score) / 2
            return score

        # Calculate average or max score
        score = score.mean(dim=1) if average else score.max(dim=1)[0]
        print('avg_score:', score)
        print('avg_score shape:', score.shape)
        return score    


    def corpus_score(self, cands, list_refs=None, audio_paths=None, config='text-text', agg_score='mean'):
        assert config in ['combined', 'audio-text', 'text-text']
        assert agg_score in {'none', 'mean', 'max'}

        sim_scores = self.get_similarity_score(cands, list_refs, audio_paths, config, average=True)

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

if __name__ == "__main__":
    evaluator = Evaluator(device='cuda', clap_model='MS-CLAP', echecker_model='echecker_clotho_audiocaps_tiny')
