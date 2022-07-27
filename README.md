# Python GLMB filter with History-based Birth

This code was used to produce the results shown in:

> C. Aguilar, M. Ortner and J. Zerubia, "Adaptive Birth for the GLMB Filter for object tracking in satellite videos," 2022 IEEE 32st International Workshop on Machine Learning for Signal Processing (MLSP), 2022, pp. 1-6, doi: "

If you use this code, we strongly suggest you cite:

    @inproceedings{aguilar_glmb_adaptive_b,
        author = {Aguilar, Camilo and Ortner, Mathias and Zerubia, Josiane},
        title = {Adaptive Birth for the GLMB Filter for object tracking in satellite videos},
        booktitle = {2022 IEEE 32nd International Workshop on Machine Learning for Signal Processing (MLSP)},
        pages={1-6},
        doi={},
        Year = {2022}
    }

|Traditional GLMB | AB-GLMB |
|:--:| :--:|
| <img src="images/glmb.gif"> | <img src="images/adaptive_birth.gif"> |
### GLMB-with adaptive birth

### Contents
1. [Installation](#installation-sufficient-for-the-demo)
2. [Usage](#usage)
3. [Dataset Format](#dataset)


### Installation

1. Clone the repository
  ```Shell
  git clone --recursive https://github.com/Ayana-Inria/GLMB-adaptive-birth-satellite-videos
  ```

2.
To install required dependencies run:
```
$ pip install -r requirements.txt
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


#### Using AB-GLMB in your own project

You can use our code by adding the following lines of code:

```python
# python 3.6
from glmb import glmb_ab

#create instance of SORT
mot_tracker = glmb_ab()

# get detections
...

# update SORT
track_bbs_ids = glmb_ab.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
...
```

#### Parameter Choice

The GLMB filter relies on numerous parameters, but the most important ones are:

```python
a=1
b=2
c=3
```

### Aknowledgment
    Thanks to BPI France (LiChiE contract) for funding this research work, and to the OPAL infrastructure from Université Côte d'Azur for providing computational resources and support.

### License
    GLMB with adaptive birth is released under the GNUv3 License (refer to the LICENSE file for details).
