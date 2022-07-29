from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import csv
from PIL import Image, ImageOps
import cv2
import glob
import motmetrics as mm

# from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
def get_number_string(number):
    '''
        Convert Sequence Number to String for Saving Purposes
    '''
    if(number > 999):
        number_string = str(number)
    elif(number > 99):
        number_string = '0' + str(number)
    elif(number > 9):
        number_string = '00' + str(number)
    else:
        number_string = '000' + str(number)

    return number_string


def read_image_files_2gray(input_sequence_folder_name='valencia_sequence', input_sequence_name='valencia_sequence'):
    images = []
    for i in range(1, 581):
        input_image = Image.open(input_sequence_folder_name + '/' + input_sequence_name + '_' + get_number_string(i) + '.png')
        input_image = ImageOps.grayscale(input_image)
        im_og = np.array(input_image, dtype=np.uint8)
        images.append(im_og)

    images = np.array(images)
    images = np.moveaxis(images, 0, -1)
    return images


def save_labels_as_colors(image, directory="images", k=0):
    np.random.seed(seed=0)
    if not os.path.exists(directory):
        os.makedirs(directory)

    number_of_objects = image.max()
    color_array = (255 * np.random.rand(number_of_objects, 3)).astype(np.uint8)
    new_im = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for label in np.unique(image):
        if(label == 0):
            continue
        for color in range(3):
            temp_im = np.zeros(image.shape)
            temp_im[np.where(image == label)] = color_array[label - 1, color]
            new_im[:, :, color] += temp_im.astype(np.uint8)
    Image.fromarray(new_im).save(directory + '/sequence_' + get_number_string(k + 1) + '.png')


def read_directory_lazy(directory):
    '''
        returns image sequene in the form [w, h, time_steps]
    '''
    list_of_files = []
    file_names = sorted([f for f in listdir(directory) if isfile(join(directory, f))])
    for name in file_names:
        path = directory + "/" +  name
        im = np.array(Image.open(path), dtype=np.uint16)
        list_of_files.append(im)

    # Sequence is of the form [t, w, h] so swap axes
    sequence = np.array(list_of_files)

    sequence = np.swapaxes(sequence, 0, 1)
    sequence = np.swapaxes(sequence, 1, 2)
    return sequence


def get_number_of_images(directory):
    '''
        returns number_of_images in a directory
    '''
    file_names = [f for f in listdir(directory) if isfile(join(directory, f))]
    num_images = len(file_names)
    return num_images


def read_appearances_at_specific_index(root_dir, index, num_channels=64):
    list_of_files = []
    for i in range(num_channels):
        directory = root_dir + "/{}".format(i)
        file_names = sorted([f for f in listdir(directory) if isfile(join(directory, f))])
        name = file_names[index]
        path = directory + "/" + name
        im = np.array(Image.open(path), dtype=np.uint16)
        list_of_files.append(im)

    # Sequence is of the form [t, w, h] so swap axes
    sequence = np.array(list_of_files)
    sequence = np.swapaxes(sequence, 0, 1)
    sequence = np.swapaxes(sequence, 1, 2)
    return sequence

def read_directory_lazy_specific_index(directory, index):
    '''
        returns image sequene in the form [w, h, 1]
    '''
    list_of_files = []
    file_names = sorted([f for f in listdir(directory) if isfile(join(directory, f))])
    name = file_names[index]
    path = directory + "/" +  name
    im = np.array(Image.open(path), dtype=np.uint16)
    list_of_files.append(im)
    # Sequence is of the form [t, w, h] so swap axes
    sequence = np.array(list_of_files)
    sequence = np.swapaxes(sequence, 0, 1)
    sequence = np.swapaxes(sequence, 1, 2)
    return sequence


def merge_sequences_side_to_side(directory0, directory1, directory2, directory3, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    sequence0 = read_directory_lazy(directory0)
    sequence0 = np.stack((sequence0, sequence0, sequence0), axis=3)
    num_s1 = sequence0.shape[2]
    sequence1 = read_directory_lazy(directory1)
    num_s1 = min(num_s1, sequence1.shape[2])
    sequence2 = read_directory_lazy(directory2)
    num_s1 = min(num_s1, sequence2.shape[2])
    sequence3 = read_directory_lazy(directory3)
    num_s1 = min(num_s1, sequence3.shape[2])


    num_dims = len(sequence2.shape)
    if(num_dims == 3):
        sequence2[:, 0, :] = 255
        sequence2[:, 1, :] = 255
    else:
        sequence2[:, 0, :, :] = 255
        sequence2[:, 1, :, :] = 255

    merged1 = np.concatenate((sequence0[:, :, 0:num_s1, ...], sequence1[:, :, 0:num_s1, ...]), axis=1).astype(np.uint8)
    merged2 = np.concatenate((sequence2[:, :, 0:num_s1, ...], sequence3[:, :, 0:num_s1, ...]), axis=1).astype(np.uint8)

    merged = np.concatenate((merged1, merged2), axis=0)
    print("Saving Side to Side")
    print(output_directory)
    for k in range(merged.shape[2]):
        if num_dims == 3:
            save_labels_as_colors(merged[:, :, k], output_directory, k)
        else:
            save_array_as_image(merged[:, :, k, :], output_directory, k + 1)


    return merged


def read_csv(directory):
    '''
        Returns a np array of size [N_frames x N_objects * 6], each object has (px, py, vx, vy, ax, ay) components
    '''

    objects_list = []
    with open(directory) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        frame_counter = 0
        for row in spamreader:
            frame_counter += 1
            if(frame_counter == 1):
                continue
            objects_list.append(row)
    objects_array = np.array(objects_list)
    return objects_array.astype(np.double)


def intermediates(p1, p2, nb_points=20):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)] 
            for i in range(1, nb_points+1)]


def save_detection_with_trajecories(csv_directory, data, output_dir):
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    print(data.shape)
    objects = read_csv(csv_directory)
    num_objects = objects.shape[1] // 6
    n_frames = objects.shape[0]
    color_array = (255.0 * np.random.random( (num_objects, 3))).astype(np.uint8)
    rows, cols, _, _ = data.shape

    for k in range(n_frames):
        image = data[:, :, k, :]
        # image = np.stack((image,) * 3, axis=-1)
        for obj_label in range(num_objects):
            obj_i = int(objects[k, obj_label * 6 + 1])
            obj_j = int(objects[k, obj_label * 6 + 2])
            if(obj_i > 0):
                for xx in range(-4, 5):
                    for yy in range(-4, 5):
                        if(abs(xx) < 3 and abs(yy) < 3):
                            continue
                        oi = min(max(obj_i + xx, 0), rows - 1)
                        oj = min(max(obj_j + yy, 0), cols - 1)
                        image[oi, oj, :] = color_array[obj_label, :]
                t_prev = k
                oi_prev = oi
                oj_prev = oj

                t = k - 1
                while(t >= 0):
                    oi = min(max(int(objects[t, obj_label * 6 + 1]), 0), rows - 1)
                    oj = min(max(int(objects[t, obj_label * 6 + 2]), 0), cols - 1)
                    if(oi > 0):
                        points = intermediates([oi_prev, oj_prev], [oi, oj])
                        for oii, ojj in points:
                            image[oii, ojj, :] = color_array[obj_label, :]
                        oi_prev = oi
                        oj_prev = oj
                    t -= 1

        frame_number = int(objects[k, 0])
        Image.fromarray((image).astype(np.uint8)).save(output_dir + '/sequence_' + get_number_string(frame_number) + '.png')


def save_array_as_image(image, directory="images", k=0):
    if not os.path.exists(directory):
        os.makedirs(directory)
    Image.fromarray(image).save(directory + '/sequence_' + get_number_string(k) + '.png')


def check_details_single_approach(dataset, approach='3_frame'):
    output_dir = dataset + "/FILTER_OUTPUT/GT_FILTER_DEBUG"
    # merge_sequences_side_to_side(directory1, directory2, output_dir)
    directory0 = dataset + "/GT/stabilized_gt"
    directory1 = dataset + "/INPUT_DATA/stabilized"
    directory2 = dataset + "/DETECTOR/superimposed"
    directory3 = dataset + "/3_frame/measurements/colors"

    sequence2 = read_directory_lazy(directory2)[:, :, ::10, :]
    sequence3 = read_directory_lazy(directory3)[:, :, ::10, :]

    sequence1 = read_directory_lazy(directory1)
    sequence0 = read_directory_lazy(directory0)
    # sequence3 = sequence3[:, :, 1:-1]
    sequence1 = np.stack((sequence1,) * 3, axis=-1)[:, :, ::10, :]
    sequence3 = (sequence1 // 4 + sequence3 // 2)

    merged = np.concatenate((sequence0, sequence2[:, :, 1:], sequence3[:, :, 1:]), axis=1).astype(np.uint8)

    print(output_dir)
    for k in range(merged.shape[2]):
        save_array_as_image(merged[:, :, k, :], output_dir, k + 1)

    return merged




def merge_sequences_side_to_gt_chinese(likelihood_directory, measurement_directory, inference_directory, gt_directory, output_directory, subsample=1):
    likelihood = read_directory_lazy(likelihood_directory)
    if(len(likelihood.shape) < 4):
        likelihood = np.stack((likelihood,) * 3, axis=-1)
    likelihood = likelihood[:, :, 10::10, :]

    measurement = read_directory_lazy(measurement_directory)
    if(len(measurement.shape) < 4):
        measurement = np.stack((measurement,) * 3, axis=-1)
    measurement = measurement[:, :, 10::10, :]

    inference = read_directory_lazy(inference_directory)
    # inference = inference[:, :, 2::10]
    if(len(inference.shape) < 4):
        inference = np.stack((inference,) * 3, axis=-1)

    inference = inference[:, :, 2::2, :]
    inference = inference[:, :, :-1]

    gt_sequence = read_directory_lazy(gt_directory)
    if(gt_sequence.shape[-1] < 3):
        gt_sequence = np.stack((gt_sequence,) * 3, axis=-1)

    print(likelihood.shape)
    print(measurement.shape)
    print(inference.shape)
    print(gt_sequence.shape)

    merged_u = np.concatenate((likelihood, measurement), axis=1).astype(np.uint8)
    merged_d = np.concatenate((gt_sequence, inference), axis=1).astype(np.uint8)
    merged = np.concatenate((merged_u, merged_d), axis=0).astype(np.uint8)
    print("Saving Side to Side")
    print(output_directory)
    for k in range(merged.shape[2]):
        save_array_as_image(merged[:, :, k, :], output_directory, k + 1)

    return merged

# output_directory + "/object_states.csv", stabilized_data_color[:, :, :, 0:3], output_directory + "/filter_trajectories"

def save_detection_with_trajecories_smart(parameters, GT_trajectories=False, trajectory_length=3):
    data = read_directory_lazy(parameters["data_path"])
    if(GT_trajectories):
        csv_directory = parameters["gt_csv_path"]
        output_dir = parameters["path"] + '/GT/gt_trajectories'
    else:
        csv_directory = parameters["path"] + '/FILTER_OUTPUT/object_states.csv'
        output_dir = parameters["path"] + '/FILTER_OUTPUT//filter_trajectories'

    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    objects = read_csv(csv_directory)
    num_objects = objects.shape[1] // 6
    n_frames = objects.shape[0]
    color_array = (255.0 * np.random.random( (num_objects, 3))).astype(np.uint8)
    
    data_colors = len(data.shape) > 3
    
    if(data_colors):
        rows, cols, _, _ = data.shape
    else:
        rows, cols, _ = data.shape

    for k in range(n_frames):
        print('Trajectory {}'.format(k + 1))
        frame_number = int(objects[k, 0])
        if("WPAFB_2009" in parameters["name"] and GT_trajectories==True):
            frame_number -= 99
        if(data_colors):
            image = data[:, :, frame_number - 1, :]
        else:
            image = data[:, :, frame_number - 1]
            image = np.dstack((image, image, image))

        # image = np.stack((image,) * 3, axis=-1)
        for obj_label in range(num_objects):
            obj_i = int(objects[k, obj_label * 6 + 1])
            obj_j = int(objects[k, obj_label * 6 + 2])

            obj_h = int(objects[k, obj_label * 6 + 5])
            obj_w = int(objects[k, obj_label * 6 + 6])
            if(obj_h <= 0):
                obj_h = 8

            if(obj_w <= 0):
                obj_w = 8

            obj_h = 2
            obj_w = 2
            if(obj_i > 0):
                for xx in range(-obj_h // 2, obj_h // 2 + 1):
                    for yy in range(-obj_w // 2, obj_w // 2 + 1):
                        if(abs(xx) < (obj_h // 2 - 1) and abs(yy) < (obj_w // 2 - 1)):
                            continue
                        oi = min(max(obj_i + xx, 0), rows - 1)
                        oj = min(max(obj_j + yy, 0), cols - 1)
                        image[oi, oj, 0:3] = color_array[obj_label, :]

                # Draw Previous Trajectory
                t_prev = k
                oi_prev = obj_i
                oj_prev = obj_j

                t = k - 1
                counter_prev_steps = 0
                while(t >= 0 and counter_prev_steps < trajectory_length):
                    counter_prev_steps += 1
                    oi = min(max(int(objects[t, obj_label * 6 + 1]), 0), rows - 1)
                    oj = min(max(int(objects[t, obj_label * 6 + 2]), 0), cols - 1)
                    if(oi > 0):
                        points = intermediates([oi_prev, oj_prev], [oi, oj])
                        for oii, ojj in points:
                            oii = min(max(oii, 0), rows - 1)
                            ojj = min(max(ojj, 0), cols - 1) 
                            image[oii, ojj, 0:3] = color_array[obj_label, :]
                        oi_prev = oi
                        oj_prev = oj
                    t -= 1

        frame_number = int(objects[k, 0])
        Image.fromarray((image).astype(np.uint8)).save(output_dir + '/sequence_' + get_number_string(frame_number) + '.png')


def save_detection_with_trajecories_smart_w_ids_and_selection(parameters, GT_trajectories=False, trajectory_length=3,
                                                              specific_ids=None,
                                                              specific_dir=None,
                                                              name_modifier=None):
    if specific_dir is None:
        specific_dir = parameters["data_path"]
    data = read_directory_lazy(specific_dir)
    if(GT_trajectories):
        csv_directory = parameters["gt_csv_path"]
        if specific_ids is None:
            output_dir = parameters["path"] + '/GT/gt_trajectories'
        else:
            output_dir = parameters["path"] + '/GT/ids=' + str(specific_ids)
    else:
        csv_directory = parameters["path"] + '/FILTER_OUTPUT/' + parameters['filter_name'] + '/' + parameters["measurement_name"] + '/object_states.csv'
        if specific_ids is None:
            output_dir = parameters["path"] + '/FILTER_OUTPUT/' + parameters['filter_name'] + '/' + parameters["measurement_name"] + '/filter_trajectories'
        else:
            output_dir = parameters["path"] + '/FILTER_OUTPUT/' + parameters['filter_name'] + '/' + parameters["measurement_name"] + '/filter_trajectories_specific_ids=' + str(specific_ids)

    if name_modifier is not None:
        output_dir = output_dir + "_" + name_modifier

    print("Saving at {}".format(output_dir))
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    objects = read_csv(csv_directory)
    num_objects = objects.shape[1] // 6
    n_frames = objects.shape[0]
    color_array = (255.0 * np.random.random( (num_objects, 3))).astype(np.uint8)

    data_colors = len(data.shape) > 3

    if(data_colors):
        rows, cols, _, _ = data.shape
    else:
        rows, cols, _ = data.shape

    for k in range(n_frames):
        print('Trajectory {}'.format(k + 1))
        frame_number = int(objects[k, 0])
        if("AFRL" in parameters["name"] and GT_trajectories==True):
            frame_number -= 99
        if(data_colors):
            image = data[:, :, frame_number - 1, :]
        else:
            image = data[:, :, frame_number - 1]
            image = np.dstack((image, image, image))

        # image = np.stack((image,) * 3, axis=-1)
        for obj_label in range(num_objects):

            # in case of plotting specific ids
            if specific_ids is not None and obj_label not in specific_ids:
                continue
            obj_i = int(objects[k, obj_label * 6 + 1])
            obj_j = int(objects[k, obj_label * 6 + 2])

            obj_h = int(objects[k, obj_label * 6 + 5])
            obj_w = int(objects[k, obj_label * 6 + 6])
            if(obj_h <= 0):
                obj_h = 8

            if(obj_w <= 0):
                obj_w = 8

            obj_h = 12
            obj_w = 12
            if(obj_i > 0):
                for xx in range(-obj_h // 2, obj_h // 2 + 1):
                    for yy in range(-obj_w // 2, obj_w // 2 + 1):
                        if(abs(xx) < (obj_h // 2 - 1) and abs(yy) < (obj_w // 2 - 1)):
                            continue
                        oi = min(max(obj_i + xx, 0), rows - 1)
                        oj = min(max(obj_j + yy, 0), cols - 1)
                        image[oi, oj, 0:3] = color_array[obj_label, :]

                letter_color = (int(color_array[obj_label, 0]), int(color_array[obj_label, 1]), int(color_array[obj_label, 2]))
                image = cv2.putText(image, str(obj_label), org=(int(oj), int(oi)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=letter_color)


                # Draw Previous Trajectory
                t_prev = k
                oi_prev = obj_i
                oj_prev = obj_j

                t = k - 1
                counter_prev_steps = 0
                while(t >= 0 and counter_prev_steps < trajectory_length):
                    counter_prev_steps += 1
                    oi = min(max(int(objects[t, obj_label * 6 + 1]), 0), rows - 1)
                    oj = min(max(int(objects[t, obj_label * 6 + 2]), 0), cols - 1)
                    if(oi > 0):
                        points = intermediates([oi_prev, oj_prev], [oi, oj])
                        for oii, ojj in points:
                            oii = min(max(oii, 0), rows - 1)
                            ojj = min(max(ojj, 0), cols - 1)
                            image[oii, ojj, 0:3] = color_array[obj_label, :]
                        oi_prev = oi
                        oj_prev = oj
                    t -= 1

        frame_number = int(objects[k, 0])
        Image.fromarray((image).astype(np.uint8)).save(output_dir + '/sequence_' + get_number_string(frame_number) + '.png')


def crop_for_display(parameters, number, xy_start=[0, 0], window_size=150):
    xy_end = [xy_start[0] + window_size, xy_start[1] + window_size]
    # print("Cropping for Display: {}".format(parameters['measurement_name']))
    folder_name = str(number) + "_start_" + str(xy_start) + "_end_" + str(xy_end)
    directory = parameters['path'] + '/DISPLAY_IMAGES/' + folder_name 
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory2 = parameters['path'] + '/DISPLAY_IMAGES/COMPARISON'
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    # FOR GT AND OG IMAGE
    if(parameters['measurement_name'] == 'GT_Measurements'):
        im = read_directory_lazy_specific_index(parameters["data_path"], number - 1)
        im = im[xy_start[0]:xy_end[0], xy_start[1]:xy_end[1], 0, :].astype(np.uint8)
        method_name = "og"
        Image.fromarray(im).save(directory + '/' + method_name + '.png')

        im = read_directory_lazy_specific_index(parameters["path"] + "/DETECTOR/3_frame_measurements/likelyhood", number - 1)
        im = im[xy_start[0]:xy_end[0], xy_start[1]:xy_end[1], 0].astype(np.uint8)
        method_name = "3_frame_l"
        Image.fromarray(im).save(directory + '/' + method_name + '.png')

        # To save GT
        number = int(round(float(number) / 10.0))

    # Save Image
    im = read_directory_lazy_specific_index(parameters["display_path"], number - 1)
    im = im[xy_start[0]:xy_end[0], xy_start[1]:xy_end[1], 0, :].astype(np.uint8)
    method_name = parameters['measurement_name']
    Image.fromarray(im).save(directory + '/' + method_name + '.png')
    return im, method_name


def crop_for_display_filter(parameters, number, xy_start=[0, 0], window_size=150):
    xy_end = [xy_start[0] + window_size, xy_start[1] + window_size]
    # print("Cropping for Display: {}".format(parameters['measurement_name']))
    folder_name = str(number) + "_start_" + str(xy_start) + "_end_" + str(xy_end)
    directory = parameters['path'] + '/DISPLAY_IMAGES/' + folder_name 
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory2 = parameters['path'] + '/DISPLAY_IMAGES/COMPARISON_FILTER'
    if not os.path.exists(directory2):
        os.makedirs(directory2)

    # Save Image
    print(parameters["path"] + '/FILTER_OUTPUT/' + parameters['filter_name'] + '/' +  parameters["measurement_name"])
    im = read_directory_lazy_specific_index(parameters["path"] + '/FILTER_OUTPUT/' + parameters['filter_name'] + '/' +  parameters["measurement_name"] + "/filter_trajectories", number - 1)
    im = im[xy_start[0]:xy_end[0], xy_start[1]:xy_end[1], 0, :].astype(np.uint8)
    method_name = parameters['measurement_name']
    return im, method_name


def crop_images(directory_list, output_directory, reference, ks=[1], names=None):
    # function to display the coordinates of
    # of the points clicked on the image
    x = 0
    y = 0
    def click_and_crop(event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished
            refPt.append((x, y))
            cropping = False
            # draw a rectangle around the region of interest
            cv2.rectangle(img, refPt[0], refPt[1], (0, 0, 255), 2)
            cv2.imshow("image", img)


    def draw_circle(event, x, y, flags, param):
        global refCircles
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                refCircles.append((x, y))
            except:
                refCircles = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            refCircles.append((x, y))

    # reading the image
    img0_dir = directory_list[0]
    number_images = get_number_of_images(img0_dir)
    if ks is None:
        ks = np.arange(number_images)
    img = read_directory_lazy_specific_index(img0_dir, ks[0]).astype(np.uint8)
    if(len(img.shape) > 3):
        img = img[:, :, 0, :]

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_and_crop)

    # wait for a key to be pressed to exit
    circles = False
    key = cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

    if not os.path.exists(output_directory + "/" + reference):
        os.makedirs(output_directory + "/" + reference)
        os.makedirs(output_directory + "/" + reference + "/large_images")
        os.makedirs(output_directory + "/" + reference + "/marked_images")

    super_large_image = None
    large_image = None

    save_indxs = [0, len(ks) // 2, len(ks) - 1]
    # list_of_frames_for_gif = len(directory_list) * [None]
    list_of_list_of_frames = []

    dir_names_dict = {}
    directory_counter = 0
    for im_dir in directory_list:
        list_of_frames = []
        dir_names = os.path.normpath(im_dir).split(os.sep)
        if names is None:
            title = dir_names[-1]
        else:
            title = names[directory_counter]
            directory_counter += 1

        dir_name = dir_names[-3] + "_" + dir_names[-2] + "_" + dir_names[-1]
        if dir_name in dir_names_dict.keys():
            dir_names_dict[dir_name] += 1
        else:
            dir_names_dict[dir_name] = 1

        if(dir_names_dict[dir_name] > 1):
            dir_name = dir_name + str(dir_names_dict[dir_name])
        print(dir_name)
        if not os.path.exists(output_directory + "/" + reference + "/" + dir_name):
            os.makedirs(output_directory + "/" + reference + "/" + dir_name)
        number = 0
        for k in ks:
            temp_image = read_directory_lazy_specific_index(im_dir, k).astype(np.uint8)
            if len(temp_image.shape) > 3:
                temp_image = temp_image[:, :, 0, :]
            cropped_im = temp_image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0], ...]

            if(cropped_im.shape[-1] == 1):
                cropped_im = np.concatenate((cropped_im, cropped_im, cropped_im), axis=2)

            white_strip = 255 * np.ones((cropped_im.shape[0], 10, 3)).astype((np.uint8))

            textSize = cv2.getTextSize(title, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, thickness=1)
            text_width, text_height = textSize[0]
            offset_text = cropped_im.shape[1] // 2 - text_width // 2 - 1
            cv2.rectangle(cropped_im, (offset_text, 10 - text_height // 2 - 3), (cropped_im.shape[1] // 2 + text_width // 2 + 1, 10 + text_height // 2 + 3), color=(0, 0, 0), thickness=-1)
            cv2.putText(cropped_im, title, org=(offset_text, 20), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=(255, 255, 255))

            if(number in save_indxs):
                if(number == 0):
                    large_image = np.concatenate((cropped_im, white_strip), axis=1)
                else:
                    large_image = np.concatenate((large_image, cropped_im, white_strip), axis=1)

            PIL_image = cv2.cvtColor(cropped_im, cv2.COLOR_BGR2RGB)
            list_of_frames.append(Image.fromarray(PIL_image))
            cv2.imwrite(output_directory + "/" + reference + "/" + dir_name + "/im_" + str(number).zfill(2) + ".png", cropped_im)
            number += 1

        # Create large image that contains all plots at 3 times
        if super_large_image is None:
            super_large_image = large_image
            white_h_strip = 255 * np.ones((15, super_large_image.shape[1], 3)).astype((np.uint8))
            super_large_image = np.concatenate((super_large_image, white_h_strip), axis=0)
        else:
            super_large_image = np.concatenate((super_large_image, large_image, white_h_strip), axis=0)

        # create large image for gif
        # if super_large_image_for is None:

        # Write large image and GIF
        cv2.imwrite(output_directory + "/" + reference + "/large_images/" + dir_name + ".png", large_image)
        make_gif(list_of_frames, output_directory + "/" + reference, dir_name)

        # Append GIF to the list of frames
        list_of_list_of_frames.append(list_of_frames)

    white_h_strip = 255 * np.ones((30, super_large_image.shape[1], 3)).astype((np.uint8))
    super_large_image = np.concatenate((white_h_strip, super_large_image), axis=0)

    offset_x = cropped_im.shape[1] // 2
    # cv2.putText(super_large_image, "t:", org=(10, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 0, 0))

    for k_idx in save_indxs:
        k = ks[k_idx]
        cv2.putText(super_large_image, "k={}".format(k), org=(offset_x, 20), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 0, 0))
        offset_x += cropped_im.shape[1]
    cv2.imwrite(output_directory + "/" + reference + "/SUPER_LARGE_IMAGE.png", super_large_image)

    num_frames = len(list_of_frames)
    i = 0
    print("q: left. e: right. c: draw_circles_and_save")
    while True:
        counter = 0
        for list_of_images in list_of_list_of_frames:
            cv2.imshow('image{}'.format(counter), np.array(list_of_images[i]))
            cv2.setMouseCallback('image{}'.format(counter), draw_circle)
            counter += 1

        key = cv2.waitKey(100)
        # wait for a key to be pressed to exit
        if(key&0xFF == 27):
            break
        elif(key == ord('q')):
            i = i - 1
            if(i < 0):
                i = num_frames - 1
        elif(key == ord('e')):
            i += 1
            if(i == num_frames):
                i = 0
        elif key == ord('c'):
            cv2.destroyAllWindows()
            counter = 0
            for list_of_images in list_of_list_of_frames:
                temp_image = np.array(list_of_images[i])
                for c_i in range(0, len(refCircles), 2):
                    circle_diameter = max(refCircles[c_i + 1][0] - refCircles[c_i][0], refCircles[c_i + 1][1] - refCircles[c_i][1])
                    coords = (refCircles[c_i][0] + circle_diameter // 2, refCircles[c_i][1] + circle_diameter // 2)
                    temp_image = cv2.circle(temp_image, coords, circle_diameter // 2, (0, 0, 255), 2)

                if names is None:
                    title = os.path.normpath(directory_list[counter]).split(os.sep)[-1]
                else:
                    title = names[counter]

                cv2.imwrite(output_directory + "/" + reference + "/marked_images/" + title + "_{:02}.png".format(i), temp_image)
                list_of_images[i] = Image.fromarray(temp_image)
                counter += 1
        # elif key == ord('z'):
        #    refCircles = []
    # close the window
    cv2.destroyAllWindows()

    print(f"rows: {refPt[0][1]} , {refPt[1][1]}")
    print(f"cols: {refPt[0][0]} , {refPt[1][0]}")
    file_name = os.path.join(output_directory, reference, "coords.txt")
    with open(file_name, mode='w') as f:
        f.write("coords")
        f.write("ROWS")
        f.write("x1: {}".format(refPt[0][1]))
        f.write("x2: {}".format(refPt[1][1]))
        f.write("COLS")
        f.write("y1: {}".format(refPt[0][0]))
        f.write("y2: {}".format(refPt[1][0]))


def make_gif(frames, output_directory, file_name):
    frame_one = frames[-1]
    frame_one.save(output_directory + "/" + file_name + ".gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=0)




def calculate_metrics(parameters, ellapsed_time=0, c=30, save_plots=True):
    gt_directory = parameters['gt_csv_path']
    inference_directory = parameters["path"] + '/FILTER_OUTPUT//object_states.csv'
    output_directory = parameters["path"] + "/FILTER_OUTPUT/METRICS"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    gt_objects = read_csv(gt_directory)
    inference_objects = read_csv(inference_directory)
    # inference_objects = inference_objects[2::2,:]
    # inference_objects = inference_objects[:-1, :]
    gt_frames = gt_objects.shape[0]
    inf_frames = inference_objects.shape[0]

    gt_counter = 0
    inf_counter = 0

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)
    # exit()
    while(gt_counter < gt_frames and inf_counter < inf_frames):
        inf_frame_number = inference_objects[inf_counter, 0]
        gt_frame_number = gt_objects[gt_counter, 0]

        if(parameters["name"] == "WPAFB_2009"):
            gt_frame_number -= 99

        # Make sure both gt and inf are about the same frame
        if(inf_frame_number < gt_frame_number):
            while(inf_frame_number < gt_frame_number and inf_counter < inf_frames):
                inf_counter += 1
                inf_frame_number = inference_objects[inf_counter, 0]
        elif(gt_frame_number < inf_frame_number):
            while(gt_frame_number < inf_frame_number and gt_counter < gt_frames):
                gt_counter += 1
                gt_frame_number = gt_objects[gt_counter, 0]
                if(parameters["name"] == "WPAFB_2009"):
                    gt_frame_number -= 99
        else:
            # case they are the same
            pass

        if(gt_frame_number != inf_frame_number):
            print("Error with metrics calculation. Gt and Inference data does not have overlapping frames")
            return -9999, -9999, -9999
        else:
            # print("Calculating metric for frame: {}".format(gt_frame_number))
            gt_counter += 1
            inf_counter += 1

        p_gt, v_gt, a_gt, labels_gt = get_x_clean_row(gt_objects[gt_counter - 1, :])
        p_inf, v_inf, a_inf, labels_inf = get_x_clean_row(inference_objects[inf_counter - 1, :])
        N_gt = len(p_gt)
        N_inf = len(p_inf)

        # print(inf_frame_number, gt_frame_number)
        # print(N_inf, N_gt)

        if(N_gt == 0 or N_inf == 0):
            continue

        # Distance Matrix
        dist_m_p = distance_matrix(p_gt, p_inf)
        rows, cols = np.where(dist_m_p > c)
        dist_m_p[rows, cols] = np.nan
        acc.update(
        labels_gt,                     # Ground truth objects in this frame
        labels_inf,                  # Detector hypotheses in this frame
        dist_m_p,     # Distances from object 1 to hypotheses 1, 2, 3
        )
        # print(acc.events)
    mh = mm.metrics.create()
    summary = mh.compute_many(
    [acc, acc.events.loc[0:1]],
    metrics=['num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_detections', 'num_false_positives', 'num_misses', 'num_switches', 'recall', 'precision', 'mota', 'motp'],
    names=['full', 'part'],
    generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )

    '''
    event_history = str(acc.mot_events)
    f = open('test_event_history.txt', 'w')
    f.write(event_history)
    f.close()
    '''

    precision = float(int(summary["precision"]['OVERALL'] * 1000)) / 1000
    recall = float(int(summary["recall"]['OVERALL'] * 1000)) / 1000
    f1 = (2 * precision * recall ) / (precision + recall)

    print(strsummary)
    print(f1)
    # file = open(output_directory + "/test.txt", "w")
    file = open(output_directory + "/metrics.txt", "w")
    file.writelines("\n\n")
    file.writelines(strsummary)
    file.writelines("\nf1: {}".format(f1))
    file.close()



def get_x_clean_row(object_array):
    '''
        Gets object row, removes empty data separates each property into a 2D tuple
        returns 3 np 2D arrays of size: [N_objects x 2] and a list of labels
    '''
    list_of_ps = []
    list_of_vs = []
    list_of_as = []
    list_of_labels = []
    for i in range(1, object_array.shape[0], 6):
        label = i // 6
        px, py = [object_array[i], object_array[i + 1]]
        vx, vy = [object_array[i + 2], object_array[i + 3]]
        ax, ay = [object_array[i + 4], object_array[i + 5]]
        if(px > 0):
            list_of_ps.append([px, py])
            list_of_vs.append([vx, vy])
            list_of_as.append([ax, ay])
            list_of_labels.append(label)
    ps = np.array(list_of_ps)
    vs = np.array(list_of_vs)
    acs = np.array(list_of_as)
    return ps, vs, acs, list_of_labels

