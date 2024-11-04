# MACE: Multimodal Audio-Text Evaluation for Automated Audio Captioning
[[`Paper`](https://arxiv.org/abs/2411.00321)]

MACE (Multimodal Audio-Caption Evaluation) is a metric designed for evaluating automated audio captioning (AAC) systems. Unlike metrics that compare machine-generated captions solely to human references, MACE uses both audio and text to improve evaluation. By integrating both audio and text, it produces assessments that align better with human judgments.

MACE achieves a 3.28% and 4.36% relative accuracy improvement over the FENSE metric on the AudioCaps-Eval and Clotho-Eval datasets respectively. Moreover, it significantly outperforms all the previous metrics on the audio captioning evaluation task.

## Installation
```
git clone https://github.com/satvik-dixit/mace.git
pip install -r requirements.txt
cd mace
```

## Using MACE to evaluate captions
```
```


## Citation
```
@misc{dixit2024maceleveragingaudioevaluating,
      title={MACE: Leveraging Audio for Evaluating Audio Captioning Systems}, 
      author={Satvik Dixit and Soham Deshmukh and Bhiksha Raj},
      year={2024},
      eprint={2411.00321},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2411.00321}, 
}
```

## References
- [FENSE](https://github.com/blmoistawinde/fense)
- [MS COCO Caption Evaluation](https://github.com/tylin/coco-caption)
