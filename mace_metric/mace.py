#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Optional, Union
from transformers.models.auto.tokenization_auto import AutoTokenizer
from msclap import CLAP
import torch
from torch import Tensor

from fer import (
    fer,
    _load_echecker_and_tokenizer,
    BERTFlatClassifier,
    DEFAULT_FER_MODEL,
)
from clap_sim import (
    clap_sim,
    _load_clap,
    DEFAULT_CLAP_SIM_MODEL,
)
from utils.checks import check_metric_inputs


pylog = logging.getLogger(__name__)


def fense(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    # CLAP args
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    # FluencyError args
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: int = 32,
    reset_state: bool = True,
    return_probs: bool = False,
    # Other args
    penalty: float = 0.3,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """
    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param clap_model: The CLAP model used to extract CLAP embeddings for cosine-similarity. defaults to "MS-CLAP-2023".
    :param echecker: The echecker model used to detect fluency errors.
        Can be "echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny", "none" or None.
        defaults to "echecker_clotho_audiocaps_base".
    :param echecker_tokenizer: The tokenizer of the echecker model.
        If None and echecker is not None, this value will be inferred with `echecker.model_type`.
        defaults to None.
    :param error_threshold: The threshold used to detect fluency errors for echecker model. defaults to 0.9.
    :param penalty: The penalty coefficient applied. Higher value means to lower the cos-sim scores when an error is detected. defaults to 0.9.
    :param device: The PyTorch device used to run FENSE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    # Init models
    clap_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
        clap_model=clap_model,
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )
    clap_sim_outs: tuple[dict[str, Tensor], dict[str, Tensor]] = clap_sim(  # type: ignore
        candidates=candidates,
        mult_references=mult_references,
        return_all_scores=True,
        sbert_model=clap_model,
        device=device,
        batch_size=batch_size,
        reset_state=reset_state,
        verbose=verbose,
    )
    fer_outs: tuple[dict[str, Tensor], dict[str, Tensor]] = fer(  # type: ignore
        candidates=candidates,
        return_all_scores=True,
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        error_threshold=error_threshold,
        device=device,
        batch_size=batch_size,
        reset_state=reset_state,
        return_probs=return_probs,
        verbose=verbose,
    )
    mace_outs = _mace_from_outputs(clap_sim_outs, fer_outs, penalty)

    if return_all_scores:
        return mace_outs
    else:
        return mace_outs[0]["mace"]


def _mace_from_outputs(
    clap_sim_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    fer_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    penalty: float = 0.3,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Combines CLAP and FER outputs.

    Based on https://github.com/blmoistawinde/fense/blob/main/fense/evaluator.py#L121
    """
    clap_sim_outs_corpus, clap_sim_outs_sents = clap_sim_outs
    fer_outs_corpus, fer_outs_sents = fer_outs

    clap_sims_scores = clap_sim_outs_sents["clap_sim"]
    fer_scores = fer_outs_sents["fer"]
    mace_scores = clap_sims_scores * (1.0 - penalty * fer_scores)
    mace_score = torch.as_tensor(
        mace_scores.cpu()
        .numpy()
        .mean(),  # note: use numpy mean to keep the same values than the original mace
        device=mace_scores.device,
    )

    mace_outs_corpus = clap_sim_outs_corpus | fer_outs_corpus | {"mace": mace_score}
    mace_outs_sents = clap_sim_outs_sents | fer_outs_sents | {"mace": mace_score}
    mace_outs = mace_outs_corpus, mace_outs_sents

    return mace_outs


def _load_models_and_tokenizer(
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
    verbose: int = 0,
) -> tuple[CLAP, BERTFlatClassifier, AutoTokenizer]:
    clap_model = _load_clap(
        clap_model=clap_model,
        device=device,
        reset_state=reset_state,
    )
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )
    return clap_model, echecker, echecker_tokenizer
