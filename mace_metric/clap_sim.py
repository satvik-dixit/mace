#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import numpy as np
import torch

from msclap import CLAP
from torch import Tensor

from utils.checks import check_metric_inputs
from utils.globals import _get_device


pylog = logging.getLogger(__name__)
DEFAULT_CLAP_SIM_MODEL = "MS-CLAP-2023"

def clap_sim(
    method: str,
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: int = 32,
    reset_state: bool = True,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Cosine-similarity of the CLAP embeddings.
    :param method: The method used to encode the sentences. Can be "text" or "audio".
    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param clap_model: The CLAP model used to extract sentence embeddings for cosine-similarity. defaults to "2023".
    :param device: The PyTorch device used to run MACE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the CLAP models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    # Init models
    clap_model = _load_clap(clap_model, device, reset_state)

    # Encode sents
    rng_ids = [0]
    for refs in mult_references:
        rng_ids.append(rng_ids[-1] + len(refs))

    flat_references = [ref for refs in mult_references for ref in refs]

    cands_embs = _encode_sents_clap(clap_model, candidates, batch_size, verbose)
    if method == 'text':
        mrefs_embs = _encode_sents_clap(clap_model, flat_references, batch_size, verbose)
    elif method == 'audio':
        mrefs_embs = _encode_audios_clap(clap_model, flat_references, batch_size, verbose)
    else:
        raise ValueError(f"Invalid method: {method}")

    # Compute CLAP similarities
    clap_sim_scores = [(cands_embs[i] @ mrefs_embs[rng_ids[i] : rng_ids[i + 1]].T).mean().item()for i in range(len(cands_embs))]
    clap_sim_scores = np.array(clap_sim_scores)

    # Aggregate and return
    clap_sim_score = clap_sim_scores.mean()

    clap_sim_score = torch.as_tensor(clap_sim_score)
    clap_sim_score = torch.from_numpy(clap_sim_score)

    if return_all_scores:
        clap_sim_outs_corpus = {
            "clap_sim": clap_sim_score,
        }
        clap_sim_outs_sents = {
            "clap_sim": clap_sim_scores,
        }

        return clap_sim_outs_corpus, clap_sim_outs_sents
    else:
        return clap_sim_score


def _load_clap(
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
) -> CLAP:
    state = torch.random.get_rng_state()

    device = _get_device(device)
    if isinstance(clap_model, str):
        clap_model = CLAP(version='2023', use_cuda=(device=='cuda'))

    if reset_state:
        torch.random.set_rng_state(state)
    return clap_model


@torch.no_grad()
def _encode_sents_clap(
    clap_model: CLAP,
    sents: list[str],
    batch_size: int = 32,
    verbose: int = 0,
) -> Tensor:
    return clap_model.get_text_embeddings(
        sents,
        # convert_to_tensor=True,
        # normalize_embeddings=True,
        # batch_size=batch_size,
        # show_progress_bar=verbose >= 2,
    )  # type: ignore

@torch.no_grad()
def _encode_audios_clap(
    clap_model: CLAP,
    audio_paths: list[str],
    batch_size: int = 32,
    verbose: int = 0,
) -> Tensor:
    return clap_model.get_audio_embeddings(
        audio_paths,
        # convert_to_tensor=True,
        # normalize_embeddings=True,
        # batch_size=batch_size,
        # show_progress_bar=verbose >= 2,
    )  # type: ignore
