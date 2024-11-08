# MACE: Multimodal Audio-Text Evaluation for Automated Audio Captioning
[[`Paper`](https://arxiv.org/abs/2411.00321v1)]

MACE (Multimodal Audio-Caption Evaluation) is a metric designed for evaluating automated audio captioning (AAC) systems. Unlike metrics that compare machine-generated captions solely to human references, MACE uses both audio and text to improve evaluation. By integrating both audio and text, it produces assessments that align better with human judgments.

MACE achieves a 3.28% and 4.36% relative accuracy improvement over the FENSE metric on the AudioCaps-Eval and Clotho-Eval datasets respectively. Moreover, it significantly outperforms all the previous metrics on the audio captioning evaluation task.

## Runnning Evaluation on AudioCaps-Eval and Clotho-Eval Benchmark

Follow these steps to replicate the results of our paper:

Install dependencies for MACE
```
git clone -b experiments https://github.com/satvik-dixit/mace.git
pip install msclap
pip install -r requirements.txt
```

Install dependencies for some traditional AAC metrics
```
chmod 755 mace/caption-evaluation-tools/coco_caption/get_stanford_models.sh
mace/caption-evaluation-tools/coco_caption/get_stanford_models.sh
```

Run evaluation of popular AAC metrics on AudioCaps-Eval and Clotho-Eval 
```
cd mace/experiments
python main.py
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
