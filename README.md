# Python GLMB filter with History-based Birth

This code was used to produce the results shown in:

> C. Aguilar, M. Ortner and J. Zerubia, "Adaptive Birth for the GLMB Filter for object tracking in satellite videos," 2022 IEEE 32st International Workshop on Machine Learning for Signal Processing (MLSP), 2022, pp. 1-6, doi: "

If you use this code, we strongly suggest you cite:

    @inproceedings{aguilar2022,
        author = {Aguilar, Camilo and Ortner, Mathias and Zerubia, Josiane},
        title = {Adaptive Birth for the GLMB Filter for object tracking in satellite videos},
        booktitle = {2022 IEEE 32nd International Workshop on Machine Learning for Signal Processing (MLSP)},
        pages={1-6},
        doi={},
        Year = {2022}
    }

|Tracked Objects | Adaptive Birth Field |
|:--:| :--:|
| <img src="images/adaptive_birth.gif"> | <img src="images/birth_field.gif"> |

### GLMB-with adaptive birth

### Contents
1. [Installation](#installation-sufficient-for-the-demo)
2. [Usage](#usage)
3. [Results Replication](#dataset)


### Installation

1. Clone the repository
  ```Shell
  git clone --recursive https://github.com/Ayana-Inria/GLMB-adaptive-birth-satellite-videos
  ```

2.
To install required dependencies run:
```Shell
$ pip install -r requirements.txt
```



### Usage

To run the demo
```Bash
python demo.py
```

Demo outputs are saved under:

```
[root directory]/dataset/WPAFB_2009/AOI_02/FILTER_OUTPUT/
```


### Data Structure
To replicate
```
[root for demo.py]
    └── WPAFB_2009/
        └── AOI_02/
            ├── INPUT_DATA/
            |   ├── img01.png
            |   ├── img02.png
            |   └── ...
            ├── GT/
            |   ├── stabilized_oject_states.csv
            |   └── labels/
            |         ├── labels_as_points_01.png
            |         ├── labels_as_points_02.png
            |         └── ...
            └── FILTER_OUTPUT/
                ├── birth_field/
                |     ├── birth_field_01.png
                |     ├── birth_field_02.png
                |     └── ...
                ├── objects/
                |     ├── tracked_objects_01.png
                |     ├── tracked_objects_02.png
                |     └── ...
                ├── labels/
                |     ├── labels_as_points_01.png
                |     ├── labels_as_points_02.png
                |     └── ...
                └── object_states.csv



```

#### Using AB-GLMB in your own project

You can use our code by adding the following lines of code:

```python
# python 3.6
import src.filter.glmb as GLMB_FILTER_BASE
import src.filter.adaptive_birth_glmb as ADAPTIVE_GLMB



# Initialize Filter Parameters
model = GLMB_FILTER_BASE.Model(parameters)
filter_parameters = GLMB_FILTER_BASE.Filter(model)

# Create instance of GLMB
glmb_update = GLMB_FILTER_BASE.glmb_instance()
glmb_update.w = torch.tensor([1])
glmb_update.n = [0]
glmb_update.cdn = [1]


# get detections
...

# update GLMB
glmb_update = ADAPTIVE_GLMB.jointpredictupdate_a_birth(glmb_update,
                                                               model,
                                                               filter_parameters,
                                                               zk,
                                                               birth_field,
                                                               k)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
...
```

#### Parameter Choice

The GLMB filter relies on numerous parameters, but the most important ones are:

```python
# parameters.py
parameters['tau'] = 1 # Sampling Frequency
parameters['Q'] = 10    
parameters['R'] = 1
parameters['Po'] = 5 # birth covariance
parameters["gating_distance"] = 30
```

### Aknowledgment
    Thanks to BPI France (LiChiE contract) for funding this research work, and to the OPAL infrastructure from Université Côte d'Azur for providing computational resources and support.

### License
    GLMB with adaptive birth is released under the GNUv3 License (refer to the LICENSE file for details).
