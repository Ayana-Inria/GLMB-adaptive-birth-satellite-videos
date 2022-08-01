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
| <img src="adaptive_birth.gif"> | <img src="birth_field.gif"> |

### GLMB-with adaptive birth

### Contents
1. [Installation](#installation-sufficient-for-the-demo)
2. [Usage](#usage)
3. [Dataset Registration](#dataset)


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


#### Parameter Choice

The GLMB filter relies on numerous parameters, but the most important ones are:

```python
# parameters.py
parameters['tau'] = 1   # Sampling period
parameters['Q'] = 10    # Motion covariance  
parameters['R'] = 1     # Measurement covariance
parameters['Po'] = 5    # Birth covariance
```


#### Using AB-GLMB in your own project

All the essential functions for our filter are located at the src/filter/**glmb.py** file:

```python
# python 3.6
import src.filter.glmb as GLMB_FILTER

# Fix filter parameters
parameters = {}
parameters['tau'] = 1   # Sampling period
parameters['Q'] = 10    # Motion covariance  
parameters['R'] = 1     # Measurement covariance
parameters['Po'] = 5    # Birth covariance

# Initialize Filter Model
model = GLMB_FILTER.Model(parameters)

# Create instance of GLMB
glmb_update = GLMB_FILTER.glmb_instance()

# Create birth field
birth_field = torch.zeros((data_rows, data_cols, 3))

# Create state list
X = []

# Iterate over each frame
    # get frame detections
    zk = get_frame_detections(frame_number) # torch tensor with shape (n_objects_at_frame_k, n_dims). n_dims=4: [px, py, w, h]

    # Update GLMB
    glmb_update = GLMB_FILTER.jointpredictupdate_a_birth(glmb_update, model,  zk, birth_field, frame_number)

    # State Estimation
    Xk = glmb_update.extract_estimates() # Xk is a dictionary of the form Xk[obj_label] = [px, py, vx, vy, w, h]

    # Add frame state to state list
    X.append(Xk)

    # Update birth field
    birth_field = update_birth_field(birth_field, X, frame_number)


```

### Dataset
#### Video Regisration
1. Download the WPAFB 2009 dataset from the [AFRL's Sensor Data Management System (SDMS) website](https://www.sdms.afrl.af.mil/index.php?collection=wpafb2009)
2. Convert the .ntf files to .png (we used Matlab's _nitfread_ function)
3. Use our [video stabilization repository](https://github.com/Ayana-Inria/satellite-video-stabilization) to stabilize the sequence and format the labeling


#### Data Structure
To replicate the results shown in the paper, the DATASET needs to be formatted in the following way:
```
[root for demo.py]
└──dataset
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

### Aknowledgment
    Thanks to BPI France (LiChiE contract) for funding this research work, and to the OPAL infrastructure from Université Côte d'Azur for providing computational resources and support.

### License
    GLMB with adaptive birth is released under the GNUv3 License (refer to the LICENSE file for details).
