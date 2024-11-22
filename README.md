# MACE: Multimodal Audio-Text Evaluation for Automated Audio Captioning
[[`Paper`](https://arxiv.org/abs/2411.00321)]

MACE (Multimodal Audio-Caption Evaluation) is a metric designed for evaluating automated audio captioning (AAC) systems. Unlike metrics that compare machine-generated captions solely to human references, MACE uses both audio and text to improve evaluation. By integrating both audio and text, it produces assessments that align better with human judgments.

MACE achieves a 3.28% and 4.36% relative accuracy improvement over the FENSE metric on the AudioCaps-Eval and Clotho-Eval datasets respectively. Moreover, it significantly outperforms all the previous metrics on the audio captioning evaluation task.

## Installation
```
git clone https://github.com/satvik-dixit/mace.git
pip install -q msclap
cd mace
pip install -r requirements.txt
```

## Using MACE to evaluate captions
```
cd mace_metric
from mace import mace

candidates: list[str] = ["a man is speaking", "rain falls"]
mult_references: list[list[str]] = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]
audio_paths: list[str] = ["/content/mace/assets/woman_singing.wav", "/content/mace/assets/rain.wav"]

# MACE_text
corpus_scores = mace(method='text', candidates=candidates, mult_references=mult_references)
print('corpus_scores (text):', corpus_scores)

# MACE_audio
corpus_scores = mace(method='audio', candidates=candidates, audio_paths=audio_paths)
print('corpus_scores (audio):', corpus_scores)

# MACE
corpus_scores = mace(method='combined', candidates=candidates, mult_references=mult_references, audio_paths=audio_paths)
print('corpus_scores (combined):', corpus_scores)

```


## Citation
```
@article{dixit2024mace,
  title={MACE: Leveraging Audio for Evaluating Audio Captioning Systems},
  author={Dixit, Satvik and Deshmukh, Soham and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2411.00321},
  year={2024}
}
```

## References
- [AAC-Metrics](https://github.com/Labbeti/aac-metrics/tree/main)
- [FENSE](https://github.com/blmoistawinde/fense)
- [MS COCO Caption Evaluation](https://github.com/tylin/coco-caption)
