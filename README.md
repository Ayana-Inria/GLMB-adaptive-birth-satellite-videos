# Python GLMB filter with History-based Birth

Python adaptation of our paper "Adaptive Birth for the GLMB Filter for object tracking in satellite videos",

|Traditional GLMB | AB-GLMB |
|:--:| :--:|
| <img src="glmb.gif"> | <img src="adaptive_birth.gif"> |
### GLMB-with adaptive birth

Please consider citing:

    @inproceedings{aguilar_glmb_adaptive_b,
        author = {Aguilar, Camilo and Ortner, Mathias and Zerubia, Josiane},
        title = {Adaptive Birth for the GLMB Filter for object tracking in satellite videos},
        booktitle = {2022 IEEE 32nd International Workshop on Machine Learning for Signal Processing (MLSP)},
        pages={1-6},
        doi={},
        Year = {2022}
    }

### License

    GLMB with adaptive birth is released under the GNUv3 License (refer to the LICENSE file for details).

### Contents
1. [Installation](#installation-sufficient-for-the-demo)
2. [Usage](#usage)
3. [Sample Dataset](#dataset)


### Installation

1. Clone the repository
  ```Shell
  git clone --recursive https://github.com/Ayana-Inria/GLMB-adaptive-birth-satellite-videos
  ```

2. Install the required libraries
```Bash
# install required libraries
pip install requirements.txt
```


### Usage

To run the demo
```Bash
python demo.py --inputs measurements.txt
```

Test outputs are saved under:

```
filter_outputs/object_states.csv
```
