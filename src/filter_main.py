import numpy as np
import time
import os

# Data reader
import src.data_reader.data_reader as data_reader
import src.filter.glmb as AB_GLMB_FILTER

import torch
import time

from PIL import Image
from skimage.draw import circle, circle_perimeter, ellipse_perimeter, rectangle_perimeter
import cv2

import csv

def perform_adaptive_birth_glmb_filter(parameters):

    # Read Measurements
    data = data_reader.read_directory_lazy(parameters["gt_measurement_path"])  # [w, h, time_steps]
    data_to_plot = data_reader.read_directory_lazy(parameters["data_path"])

    output_directory = parameters["path"] + '/FILTER_OUTPUT/'

    # Output stuff
    output_image = np.zeros(data.shape, dtype=np.uint16)
    plotter = GLMB_Plotter([data.shape[0], data.shape[1]])

    # Filter parameters
    model = AB_GLMB_FILTER.Model(parameters)

    # Initial Posterior
    glmb_update = AB_GLMB_FILTER.glmb_instance()

    X = []

    k = 0
    end_frame = data.shape[-1]

    # CREATE VECTOR FIELD
    birth_field = torch.zeros((data.shape[0], data.shape[1], 3))
    while k < end_frame:

        # Measure Data Comming In
        zk = get_Zk(data[:, :, k])

        # Prediction & Update
        glmb_update = AB_GLMB_FILTER.jointpredictupdate_a_birth(glmb_update, model, zk, birth_field, k)

        # State Estimation
        # D_updated = glmb_update.extract_estimates()
        Xk = glmb_update.extract_estimates()

        # Inference
        # output_image, Xk = plotter.inference_w_ellipsis(D_updated)
        Xk[-999] = k + 1
        X.append(Xk)

        # Update and plot birth field
        birth_field = update_birth_field(birth_field, X, k)
        plotter.plot_birth_field(birth_field, data_to_plot[:, :, k], output_directory, k=k + 1)

        # Save Plot
        # inference_image = plotter.save_inference_w_super_position(output_image, data_to_plot[:, :, k], output_directory, k=k + 1)

        if k % 20 == 0:
            plotter.save_object_states(X, output_directory)

        print('Finished time step {}'.format(k + 1))
        k = k + 1

    # total_time_seconds = time.time() - start_seconds
    print("Saving States")
    plotter.save_object_states(X, output_directory)



def get_Zk(data_at_time_k, specific_labels=None):
    '''
        Returns np array of shape [num_measurements x 2] for center coords of each measurement
    '''
    Zk = []
    for label in np.unique(data_at_time_k):
        if label == 0:
            continue
        if specific_labels is not None and label not in specific_labels:
            continue
        coords = np.where(data_at_time_k == label)
        Zk.append([coords[0].mean(), coords[1].mean(), 5, 5])
    Zk = np.array(Zk)
    return Zk



def update_birth_field(vector_field, X, k):
        Xk = X[k]
        for label in Xk.keys():
            if(label < 0):
                continue
            px, py, vx, vy, w, h = torch.from_numpy(Xk[label]).to(vector_field.device)
            # vx = torch.from_numpy(vx).to(vector_field.device)
            # vy = torch.from_numpy(vy).to(vector_field.device)
            for ii in range(-2, 2):
                for jj in range(-2, 2):
                    ci = min(max(int(ii + px), 0), vector_field.shape[0] - 1)
                    cj = min(max(int(jj + py), 0), vector_field.shape[1] - 1)
                    if(vector_field[ci, cj, 0] > 0):
                        vector_field[ci, cj, 0] = (vx + vector_field[ci, cj, 0]) / 2.0
                    else:
                        vector_field[ci, cj, 0] = vx

                    if(vector_field[ci, cj, 1] > 0):
                        vector_field[ci, cj, 1] = (vy + vector_field[ci, cj, 1]) / 2.0
                        # vector_field[ci, cj, 2] = (vy + vector_field[ci, cj, 2]) / 2.0
                    else:
                        vector_field[ci, cj, 1] = vy
                        # vector_field[ci, cj, 2] = vy

            if( k > 0 and label in X[k - 1].keys()):
                px_prev, py_prev, vx_prev, vy_prev, w, h = torch.from_numpy(X[k-1][label]).to(vector_field.device)
                points = intermediates([px_prev, py_prev], [px, py])
                velocities = intermediates([vx_prev, vy_prev], [vx, vy])
                v_counter = 0
                for pii, pjj in points:
                    for ii in range(-2, 2):
                        for jj in range(-2, 2):
                            ci = min(max(int(ii + pii), 0), vector_field.shape[0] - 1)
                            cj = min(max(int(jj + pjj), 0), vector_field.shape[1] - 1)
                            if(vector_field[ci, cj, 0] > 0):
                                vector_field[ci, cj, 0] = (vx + vector_field[ci, cj, 0]) / 2.0
                            else:
                                vector_field[ci, cj, 0] = vx

                            if(vector_field[ci, cj, 1] > 0):
                                vector_field[ci, cj, 1] = (vy + vector_field[ci, cj, 0]) / 2.0
                            else:
                                vector_field[ci, cj, 1] = vy

        return vector_field





def intermediates(p1, p2, nb_points=5):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)] 
            for i in range(1, nb_points+1)]



class GLMB_Plotter():
    def __init__(self, canvas_shape):
        self.canvas_shape = canvas_shape
        X = np.linspace(0, canvas_shape[0], canvas_shape[0])
        Y = np.linspace(0, canvas_shape[1], canvas_shape[1])
        self.X, self.Y = np.meshgrid(X, Y)
        pos = np.empty(self.X.shape + (2,))
        pos[:, :, 0] = self.X
        pos[:, :, 1] = self.Y
        self.pos = pos


    def plot_birth_field(self, birth_field, image, directory=".", k=0):
        birth_mins = birth_field.min() 
        birth_maxs = birth_field.max()
        idxs = np.where(birth_field.sum(axis=2) == 0)
        birth_field = ((birth_field - birth_mins) / (birth_maxs - birth_mins) * 255).numpy().astype(np.uint8)
        birth_field[idxs[0], idxs[1], 0] = 0
        birth_field[idxs[0], idxs[1], 1] = 0
        birth_field[idxs[0], idxs[1], 2] = 0

        if not os.path.exists(directory + '/birth_field'):
            os.makedirs(directory + '/birth_field')

        image_color = np.stack((image, image, image), axis=2).astype(np.uint8)
        im_to_plot = cv2.addWeighted(birth_field, 0.8, image_color, 0.7, 0)

        Image.fromarray(im_to_plot).save(directory + '/birth_field/sequence_' + str(k).zfill(4) + '.png')

    def save_inference_w_super_position(self, image, raw_image, directory="images", k=0):
        if(len(raw_image.shape) == 2):
            raw_image = np.stack((raw_image, raw_image, raw_image), axis=2)
        np.random.seed(seed=0)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(directory + "/labels"):
            os.makedirs(directory + "/labels")


        if not os.path.exists(directory + "/objects"):
            os.makedirs(directory + "/objects")

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

        # if(raw_image.shape[-1]):
        # raw_im_color = np.stack((raw_image,) * 3, axis=-1)
        super_imposed = np.maximum(new_im, raw_image)

        # Image.fromarray(new_im).save(directory + '/colors/sequence_' + str(k).zfill(4) + '.png')
        Image.fromarray((image).astype(np.uint16)).save(directory + '/labels/sequence_' + str(k).zfill(4) + '.png')
        Image.fromarray((super_imposed).astype(np.uint8)).save(directory + '/objects/sequence_' + str(k).zfill(4) + '.png')


    def inference_w_ellipsis(self, D_updated):
        canvas = np.zeros(self.canvas_shape, dtype=np.uint16)
        if('Hs' not in D_updated.keys()):
            return canvas, {}

        ws = D_updated['ws']
        estimate_number_of_objects = int(round(sum(ws)))
        index_sorted = np.argsort(-1 * np.array(ws))
        # for i in range(len(ws)):
        #    print("{} ws:{} mu:{} {}".format(D_updated['Ls'][i], ws[i], D_updated['mus'][i][0], D_updated['mus'][i][1]))

        labels = []
        mus = []
        objec_states = {}

        objs_alive = 0
        obj_index = 0
        total_number_components = len(index_sorted)

        while(objs_alive < estimate_number_of_objects and obj_index < total_number_components):
            idx = index_sorted[obj_index]
            label = D_updated['Ls'][idx]
            if(label not in labels and ws[idx] > 0.0001):
                labels.append(label)
                mus.append(D_updated['mus'][idx])
                objs_alive += 1
                objec_states[label] = D_updated['mus'][idx]
            obj_index += 1

        theta = np.arange(0, 2 * np.pi, 2 * np.pi / 30)
        a = 2
        b = 1

        xpos = a * np.cos(theta)
        ypos = b * np.sin(theta)

        for i in range(objs_alive):
            xi = mus[i]
            pix = xi[0].round().astype(np.int)
            piy = xi[1].round().astype(np.int)

            v = xi[2:4]
            phi = np.arctan2(v[0], v[1]) + np.pi / 2.0

            new_xpos = xpos * np.cos(phi) + ypos * np.sin(phi)
            new_ypos = -xpos * np.sin(phi) + ypos * np.cos(phi)

            for m, n in zip(new_xpos, new_ypos):
                ii = int(min(max(pix + m, 0), canvas.shape[0] - 1))
                jj = int(min(max(piy + n, 0), canvas.shape[1] - 1))
                canvas[ii, jj] = labels[i]
        return canvas, objec_states

    def save_object_states(self, object_states, directory, FPS=None, partial_k=None):
        '''
            INPUTS:
                object_states: list of dictionaries
                                each dictionary key: object label
                                                  value: np array [1, 6] for object state
            OUTPUTS:
                csv file with this information
                values of -9999999 are added where the object does not exists

        '''
        if not os.path.exists(directory):
            os.makedirs(directory)

        number_of_frames = len(object_states)
        # Find number of objects. Could be optimized
        map_of_labels = {}
        label_n = 1
        for frame in object_states:
            for k, v in frame.items():
                if(k not in map_of_labels.keys() and k != -999):
                    map_of_labels[k] = label_n
                    label_n += 1

        number_of_objects = label_n - 1

        # Create GT array. 6 values per  object (state vector). Allow + 1 col for frame number
        gt_values = np.zeros((number_of_frames, 6 * number_of_objects + 1)) - 9999999

        # Populate gt_values
        frame_counter = 1
        for frame in object_states:
            # Row number
            gt_values[frame_counter - 1, 0] = frame[-999]
            for label, x_v in frame.items():
                if(label != -999):
                    # Store the values at each time step. Store 6 values for each object
                    gt_values[frame_counter - 1, (map_of_labels[label] - 1) * 6 + 1: (map_of_labels[label] - 1) * 6 + 1 + 6] = x_v

            frame_counter += 1

        # Start creating CSV
        if(partial_k):
            name = 'object_states_{}.csv'.format(partial_k)
        else:
            name = 'object_states.csv'
        with open(directory + '/' + name, mode='w', newline='') as state_file:
            csv_writer = csv.writer(state_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            # Create CSV file header
            header_content = []
            if(FPS is not None):
                header_content.append('FPS {:.2f}'.format(FPS))
            else:
                header_content.append('Frame Number')
            for object_n in map_of_labels.keys():
                header_content.append('obj_{}_px'.format(object_n))
                header_content.append('obj_{}_py'.format(object_n))
                header_content.append('obj_{}_vx'.format(object_n))
                header_content.append('obj_{}_vy'.format(object_n))
                header_content.append('obj_{}_ax'.format(object_n))
                header_content.append('obj_{}_ay'.format(object_n))
            csv_writer.writerow(header_content)

            for frame_n in range(gt_values.shape[0]):
                csv_writer.writerow(gt_values[frame_n, :])

