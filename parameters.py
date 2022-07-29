# Filters Functions

def get_dataset_parameters(AOI_number=0, filter_type=0):
    parameters = {}
    parameters["name"] = "WPAFB_2009"

    # Naming Parameters
    number = str(AOI_number).zfill(2)
    parameters['AOI_number'] = AOI_number
    parameters["AOI"] = "AOI_" + number

    # Directory Parameters
    path = 'dataset/WPAFB_2009/AOI_' + number
    parameters['path'] = path
    parameters["data_path"] = path + "/INPUT_DATA/stabilized_data"
    parameters["gt_measurement_path"] = path + "/GT/labels"
    parameters['gt_csv_path'] = path + "/GT/transformed_object_states.csv"


    # Filter Parameters
    parameters['tau'] = 1
    parameters['R'] = 1
    parameters['Q'] = 10
    parameters['Po'] = 5 # birth covariance


    return parameters
