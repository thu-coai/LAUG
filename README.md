# LAUG
**LAUG** is an open-source toolkit for Language understanding AUGmentation. It is an automatic method to approximate the natural perturbations to existing data. Augmented data could be used to conduct black-box robustness testing or enhancing training. [[paper]](https://arxiv.org/abs/2012.15262)


- [LAUG](#laug)
  - [Installation](#installation)
  - [Augmentation Methods](#augmentation-methods)
  - [Supported Datasets](#supported-datasets)
  - [NLU Models](#nlu-models)
  - [Citing](#citing)
  - [License](#license)
  


## Installation

Require python 3.6.

Clone this repository:
```bash
git clone https://github.com/thu-coai/LAUG.git
```

Install via pip:

```bash
cd LAUG
pip install -e .
```

Download data and models:

The data used in our paper and model parameters pre-trained by us are available at [Link](http://115.182.62.174:9876/). Please download and place them into corresponding dir. For model parameters released by others, please refer to `README.md` under dirs of each augmentation method such as `LAUG/aug/Speech_Recognition/README.md`.

## Augmentation Methods

Here are the 4 augmentation methods described in our paper. They are placed under `LAUG/aug` dir.
- Word Perturbation (WP), at `Word_Perturbation/` dir.
- Text Paraphrasing (TP), at `Text_Paraphrasing/`dir.
- Speech Recognition (SR), at `Speech_Recognition/`dir.
- Speech Disfluency (SD), at `Speech_Disfluency/`dir.

Please see our paper and README.md in each augmentation method for detailed information.

See `demo.py` for the usage of these augmentation methods.
```bash 
python demo.py
```


Noting that our augmentation methods contains several neural models, pre-trained parameters need to be downloaded before use. Parameters pre-trained by us are available at [Link](http://115.182.62.174:9876/). For parameters which released by others, please follow the instructions of each method.





## Supported Datasets

The data used in our paper is available at [Link](http://115.182.62.174:9876/) . Please download it and place it  `data/` dir.

Our data contains 2 datasets: MultiWOZ and Frames, along with their augmented copies.


- MultiWOZ
  - Original data
    - We use MultiWOZ 2.3 as the original data. We place it at `data/multiwoz/` dir.
    - Train/val/test size: 8434/999/1000 dialogs.
    - LICENSE:
  - Augmented data
    - We have 4 augmented testsets : 
      - WP (Word Perturbation), size: 1000, placed at `data/multiwoz/WP`.
      - TP (Text Paraphrasing), size: 1000, placed at `data/multiwoz/TP`.
      - SR (Speech Perturbation), size: 1000, placed at `data/multiwoz/SR`.
      - SD (Speech Disfluency), size: 1000, placed at `data/multiwoz/SD`.
    - We have 1 augmented training set :
      - Size : 16868 , Contains : 50%Original+(12.5%WP+12.5%TP+12.5%SR+12.5%SD) , placed at `data/multiwoz/Enhanced`.
  - Real user evaluation data :
    - We collected 240 utterance from real users for our real user evaluation.
    - We place it at `data/multiwoz/Real` dir.
    - Please see our paper for detailed information about the statistics and collection of the real data.

- Frames
  - Original data
    - We proccess Frames into the same format as MultiWOZ and place it at `data/Frames/` dir.
    - Train/val/test size: 1095/137/137 dialogs.
    - LICENSE:
  - Augmented data
    - We have 4 augmented testsets : 
      - WP (Word Perturbation), size: 137, placed at `data/Frames/WP`.
      - TP (Text Paraphrasing), size: 137, placed at `data/Frames/TP`.
      - SR (Speech Perturbation), size: 137, placed at `data/Frames/SR`.
      - SD (Speech Disfluency), size: 137, placed at `data/Frames/SD`.
    - We have 1 augmented training set :
      - Size : 2190 , Contains : 50%Original+(12.5%WP+12.5%TP+12.5%SR+12.5%SD) , placed at `data/Frames/Enhanced`.


## NLU Models

We provide four base NLU models which are described in our paper:
- MILU
- BERT
- CopyNet
- GPT-2

These models are adapted from [ConvLab-2](https://github.com/thu-coai/ConvLab-2). For more details, You can refer to `README.md` under `LUAG/nlu/$model/$dataset` dir such as `LAUG/nlu/gpt/multiwoz/README.md`.

## Related Work that uses LAUG

* [Know Thy Strengths: Comprehensive Dialogue State Tracking Diagnostics (EMNLP2022 Findings)](https://arxiv.org/abs/2112.08321): Provides a comprehensive dialogue state tracking diagnostics toolkit _CheckDST_ that facilitates robustness testing and failure mode analytics.  


## Citing

If you use LAUG in your research, please cite:



```
@inproceedings{liu2021robustness,
    title={Robustness Testing of Language Understanding in Task-Oriented Dialog},
    author={Liu, Jiexi and Takanobu, Ryuichi and Wen, Jiaxin and Wan, Dazhen and Li, Hongguang and Nie, Weiran and Li, Cheng and Peng, Wei and Huang, Minlie},
    year={2021},
    booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
}

```

## License

Apache License 2.0
