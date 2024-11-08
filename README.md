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

## Results on AudioCaps-Eval and Clotho-Eval Benchmarks

| **Metric**         | **Clotho-Eval ↑**     |                        |                          |                        |           | **AudioCaps-Eval ↑**  |                        |                          |                        |           |
|--------------------|-----------------------|------------------------|--------------------------|------------------------|-----------|-----------------------|------------------------|--------------------------|------------------------|-----------|
|                    | HC                    | HI                     | HM                       | MM                     | All       | HC                    | HI                     | HM                       | MM                     | All       |
| **BLEU@1**         | 51.0                  | 90.6                   | 65.5                     | 50.3                   | 59.0      | 58.6                  | 90.3                   | 77.4                     | 50.3                   | 62.4      |
| **BLEU@4**         | 52.9                  | 88.9                   | 65.1                     | 53.2                   | 60.5      | 54.7                  | 85.8                   | 78.7                     | 50.6                   | 61.6      |
| **METEOR**         | 54.8                  | 93.0                   | 74.6                     | 57.8                   | 65.4      | 66.0                  | 96.4                   | 90.0                     | 60.1                   | 71.7      |
| **ROUGE-L**        | 56.2                  | 90.6                   | 69.4                     | 50.7                   | 60.5      | 61.1                  | 91.5                   | 82.8                     | 52.1                   | 64.9      |
| **CIDEr**          | 51.4                  | 91.8                   | 70.3                     | 56.0                   | 63.2      | 56.2                  | 96.0                   | 90.4                     | 61.2                   | 71.0      |
| **SPICE**          | 44.3                  | 84.4                   | 65.5                     | 48.9                   | 56.3      | 50.2                  | 83.8                   | 77.8                     | 49.1                   | 59.7      |
| **SPICE+**         | 46.7                  | 88.1                   | 70.3                     | 48.7                   | 57.8      | 59.1                  | 85.4                   | 83.7                     | 49.0                   | 62.0      |
| **ACES**           | 56.7                  | 95.5                   | 82.8                     | 69.9                   | 74.0      | 64.5                  | 95.1                   | 89.5                     | 82.0                   | 83.0      |
| **SPIDEr**         | 53.3                  | 93.4                   | 70.3                     | 57.0                   | 64.2      | 56.7                  | 93.4                   | 70.3                     | 57.0                   | 64.2      |
| **FENSE**          | 60.5                  | 94.7                   | 80.2                     | 72.8                   | 75.7      | 64.5                  | 98.4                   | 91.6                     | 84.6                   | 85.3      |
| **CLAIR<sub>A</sub> (+ Gemini-v1.5)** | 59.0 | 95.9 | 83.2 | 75.1 | 77.4 | 70.4 | 99.2 | 93.7 | 81.5 | 84.9 |
| **CLAIR<sub>A</sub> (+ GPT-4o)**      | 62.4 | 97.1 | **83.6** | **77.9** | **79.7** | 70.9 | **99.2** | **93.3** | 84.6 | 86.6 |
| **MACE**            | **63.3**              | **98.0**               | 80.6                     | 77.0                   | 79.0      | **74.4**              | **99.2**               | **94.6**                 | **86.3**               | **88.1** |


## Citation
```
@article{dixit2024mace,
  title={MACE: Leveraging Audio for Evaluating Audio Captioning Systems},
  author={Dixit, Satvik and Deshmukh, Soham and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2411.00321},
  year={2024}
}
```
