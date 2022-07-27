import numpy as np
import time
import os

# Data reader
import src.data_reader.data_reader as data_reader
import src.filter.glmb as GLMB_FILTER_BASE
import src.filter.adaptive_birth_glmb as ADAPTIVE_GLMB

import torch
import time

from PIL import Image
from skimage.draw import circle, circle_perimeter, ellipse_perimeter, rectangle_perimeter
import cv2


def perform_adaptive_birth_glmb_filter(parameters):

    # Read Measurements
    data = data_reader.read_directory_lazy(parameters["gt_measurement_path"])  # [w, h, time_steps]
    data_to_plot = data_reader.read_directory_lazy(parameters["data_path"])

    output_directory = parameters["path"] + '/FILTER_OUTPUT/'

    # Output stuff
    output_image = np.zeros(data.shape, dtype=np.uint16)
    pl = GM_2D_Plotter([data.shape[0], data.shape[1]])

    # Filter parameters
    model = GLMB_FILTER_BASE.Model(parameters)
    filter_parameters = GLMB_FILTER_BASE.Filter(model)

    # Initial Posterior
    glmb_update = GLMB_FILTER_BASE.glmb_instance()
    glmb_update.w = torch.tensor([1])
    glmb_update.n = [0]
    glmb_update.cdn = [1]


    X = []

    average_FPS = 0.0
    start_time = time.time() # start time of the loop
    Zs_plot = {}

    k = 0
    end_frame = data.shape[-1]

    # CREATE VECTOR FIELD
    birth_field = torch.zeros((data.shape[0], data.shape[1], 3))
    while k < end_frame:

        # Measure Data Comming In
        zk = get_Zk(data[:, :, k])

        # Prediction & Update
        glmb_update = ADAPTIVE_GLMB.jointpredictupdate_a_birth(glmb_update,
                                                                       model,
                                                                       filter_parameters,
                                                                       zk,
                                                                       birth_field,
                                                                       k)

        # State Estimation
        D_updated = glmb_update.extract_estimates()

        # Inference
        output_image, Xk = pl.inference_w_ellipsis(D_updated)
        Xk[-999] = k + 1
        X.append(Xk)

        birth_field = pl.update_vector_field(birth_field, X, k)
        # Save Plot
        inference_image = pl.save_inference_w_super_position(output_image, data_to_plot[:, :, k], output_directory, k=k + 1)

        # DEBUG
        D_updated_glmb = D_updated # GLMB_FILTER.extract_all_estimates(glmb_update)
        distribution_to_plot_glmb = pl.generate_mixture_fast_w_measurements(D_updated_glmb, x_offset=0, Zs=zk)
        birth_mins = birth_field.min() 
        birth_maxs = birth_field.max() 
        idxs = (birth_field == np.array([0, 0, 0]))
        birth_field_to_plot = ((birth_field - birth_mins) / (birth_maxs - birth_mins) * 255).numpy().astype(np.uint8)
        birth_field_to_plot[idxs] = 0
        pl.plot_double_distributions_overalap(distribution_to_plot_glmb, birth_field_to_plot, data_to_plot[:, :, k], output_directory, k=k + 1, plot_stuff=False)
        k = k + 1

        num_objects = len(Xk) - 1
        seconds = (time.time() - start_time)
        if num_objects > 0:
            average_FPS = float(k + 1) / seconds
        print('Finished time step {} Objects: {} FPS is {:.2f}'.format(k, num_objects, average_FPS))
    
        if "test_iters" in parameters.keys() and k % parameters["test_iters"] == 0:
            print("Saving Partial States")
            pl.save_object_states(X, output_directory, average_FPS, partial_k=k - 1)

    # total_time_seconds = time.time() - start_seconds
    print("Saving States")
    pl.save_object_states(X, output_directory, average_FPS)





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


class GM_2D_Plotter():
    def __init__(self, canvas_shape):
        self.canvas_shape = canvas_shape
        X = np.linspace(0, canvas_shape[0], canvas_shape[0])
        Y = np.linspace(0, canvas_shape[1], canvas_shape[1])
        self.X, self.Y = np.meshgrid(X, Y)
        pos = np.empty(self.X.shape + (2,))
        pos[:, :, 0] = self.X
        pos[:, :, 1] = self.Y
        self.pos = pos

        self.mus = []
        self.sigmas = []
        self.ws = []

    def generate_mixture_fast(self, D_plot, x_offset=0, label_specific=None):
        np.random.seed(seed=0)
        color_array = (255 * np.random.rand(100, 3)).astype(np.uint8)
        Z = np.zeros((2 * self.canvas_shape[0], self.canvas_shape[1], 3), dtype=np.uint8)
        if('Hs' in D_plot.keys()):
            Hs = D_plot['Hs']
            mus = D_plot['mus']
            Ps = D_plot['Ps']
            ws = D_plot['ws']
            Ls = D_plot['Ls']
        else:
            Hs = 0

        for j in range(Hs):
            # In case we want to plot a specific label
            if(label_specific is not None and Ls[j] != label_specific):
                continue

            if(ws[j] > 0):
                # print("{}: {:.0f} {:.0f} {:.2f} {:.2f}".format(Ls[j], mus[j][0], mus[j][1], ws[j].item(), Ps[j][0, 0].item()))
                coords = ellipse_perimeter(int(mus[j][0] + x_offset), int(mus[j][1]), max(int(Ps[j][0, 0]), 1), max(1, int(Ps[j][1, 1]), 1))
                coords_x = np.clip(coords[0], 0, 2 * self.canvas_shape[0] - 1)
                coords_y = np.clip(coords[1], 0, self.canvas_shape[1] - 1)
                color_index = Ls[j] - 1
                for xx, yy in zip(coords_x, coords_y):
                    Z[xx, yy, :] = 1 * color_array[color_index % 100, :]

                coords = circle_perimeter(int(mus[j][0] + x_offset), int(mus[j][1]), 3)
                coords_x = np.clip(coords[0], 0, 2 * self.canvas_shape[0] - 1)
                coords_y = np.clip(coords[1], 0, self.canvas_shape[1] - 1)
                coords = (coords_x, coords_y)
                for xx, yy in zip(coords_x, coords_y):
                    Z[xx, yy, :] = 1 * color_array[color_index % 100, :]
        return Z


    def generate_mixture_fast_w_measurements(self, D_plot, x_offset=0, Zs=None, threshold=0):
        np.random.seed(seed=0)
        color_array = (255 * np.random.rand(100, 3)).astype(np.uint8)
        image = np.zeros((self.canvas_shape[0], self.canvas_shape[1], 3), dtype=np.uint8)
        if('Hs' in D_plot.keys()):
            Hs = D_plot['Hs']
            mus = D_plot['mus']
            Ps = D_plot['Ps']
            ws = D_plot['ws']
            Ls = D_plot['Ls']
        else:
            Hs = 0

        for j in range(Hs):
            if(ws[j] > threshold):
                # print("{}: {:.0f} {:.0f} {:.2f} {:.2f}".format(Ls[j], mus[j][0], mus[j][1], ws[j].item(), Ps[j][0, 0].item()))
                # eVe, eVa = np.linalg.eig(Ps[j].cpu().numpy())
                # print(eVa)
                angle = np.arctan2(Ps[j][1, 1].cpu().numpy(), Ps[j][0, 0].cpu().numpy())
                # angle = np.arctan2(eVa[1, 1], eVa[0, 0])
                coords = ellipse_perimeter(int(mus[j][0] + x_offset), int(mus[j][1]), max(int(Ps[j][0, 0]), 1), max(1, int(Ps[j][1, 1]), 1), orientation=angle)
                # coords = ellipse_perimeter(int(mus[j][0] + x_offset), int(mus[j][1]), max(int(eVe[1]), 1), max(1, int(eVe[0]), 1), orientation=angle)
                coords_x = np.clip(coords[0], 0, self.canvas_shape[0] - 1)
                coords_y = np.clip(coords[1], 0, self.canvas_shape[1] - 1)
                color_index = Ls[j] - 1
                for xx, yy in zip(coords_x, coords_y):
                    image[xx, yy, :] = 1 * color_array[color_index % 100, :]

                coords = circle_perimeter(int(mus[j][0] + x_offset), int(mus[j][1]), 3)
                coords_x = np.clip(coords[0], 0, self.canvas_shape[0] - 1)
                coords_y = np.clip(coords[1], 0, self.canvas_shape[1] - 1)
                coords = (coords_x, coords_y)
                for xx, yy in zip(coords_x, coords_y):
                    image[xx, yy, :] = 1 * color_array[color_index % 100, :]

        for i in range(Zs.shape[0]):
            coords = rectangle_perimeter((int(Zs[i, 0] + x_offset), int(Zs[i, 1] + x_offset)), extent=1)
            coords_x = np.clip(coords[0], 0, self.canvas_shape[0] - 1)
            coords_y = np.clip(coords[1], 0, self.canvas_shape[1] - 1)
            for xx, yy in zip(coords_x, coords_y):
                image[xx, yy, :] = np.array([255, 255, 255])

            y_c = int(Zs[i, 0])
            x_c = int(Zs[i, 1])
            letter_color = (255, 255, 255)
            # cv2.putText(image, str(i), org=(int(x_c) + 10, int(y_c) + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=letter_color)

        return image



    def add_component(self, mu, sigma, w=1):
        self.mus.append(np.array(mu))
        self.sigmas.append(np.array(sigma))
        self.ws.append(w)

    def generate_single_gaussian_2D_array(self, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """
        n = mu.shape[0]
        if(Sigma.shape[0] > 2):
            Sigma_plot = Sigma[0:2, 0:2]
        else:
            Sigma_plot = Sigma

        if(mu.shape[0] > 2):
            mu_plot = mu[0:2]
        else:
            mu_plot = mu
        Sigma_det = np.linalg.det(Sigma_plot)
        Sigma_inv = np.linalg.inv(Sigma_plot)
        N = np.sqrt((2 * np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', self.pos - mu_plot, Sigma_inv, self.pos - mu_plot)

        return np.exp(-fac / 2) / N

    def generate_mixture(self):
        Z = np.zeros(self.X.shape)
        for j in range(len(self.mus)):
            Z += self.generate_single_gaussian_2D_array(self.mus[j], self.sigmas[j])
        return Z

    def generate_mixture_from_dict(self, D_plot):
        Z = np.zeros(self.X.shape)
        if('Hs' in D_plot.keys()):
            Hs = D_plot['Hs']
            mus = D_plot['mus']
            Ps = D_plot['Ps']
            ws = D_plot['ws']
        else:
            Hs = 0

        for j in range(Hs):
            Z += ws[j] * self.generate_single_gaussian_2D_array(mus[j], Ps[j])
        return np.transpose(Z)

    def plot_all(self, Z, data=None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=cm.viridis)
        if(data is None):
            ax.contourf(self.X, self.Y, Z, zdir='z', offset=-0.20, cmap=cm.viridis)
        else:
            ax.contourf(self.X, self.Y, data, zdir='z', offset=-0.20, cmap=cm.viridis)
        # Adjust the limits, ticks and view angle
        ax.set_zlim(-0.15, 0.2)
        ax.set_zticks(np.linspace(0, 0.2, 5))
        ax.view_init(27, -21)
        plt.show()

    def save_D_plot(self, D_plot, directory="images/plots", i=0, data=None):
        Z = self.generate_mixture_from_dict(D_plot)
        if(not os.path.exists(directory)):
            os.makedirs(directory)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, 10 * Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                        cmap=cm.viridis)
        if(data is None):
            ax.contourf(self.X, self.Y, Z, zdir='z', offset=-0.20, cmap=cm.viridis)
        else:
            ax.contourf(self.X, self.Y, data, zdir='z', offset=-0.20, cmap=cm.viridis)
        # Adjust the limits, ticks and view angle
        ax.set_zlim(-0.15, 0.2)
        ax.set_zticks(np.linspace(0, 0.2, 5))
        ax.view_init(27, -21)
        plt.savefig(directory + "/gauss_pots_" + str(i).zfill(4) + ".png")
        plt.close()

    def save_inference(self, image, directory="images", k=0, specific_labels=None):
        np.random.seed(seed=0)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(directory + "/labels"):
            os.makedirs(directory + "/labels")

        if not os.path.exists(directory + "/colors"):
            os.makedirs(directory + "/colors")

        number_of_objects = image.max()
        color_array = (255 * np.random.rand(number_of_objects, 3)).astype(np.uint8)
        new_im = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for label in np.unique(image):
            if(label == 0):
                continue

            # for when plotting only a specific element
            if(specific_labels is not None and label not in specific_labels):
                continue
            for color in range(3):
                temp_im = np.zeros(image.shape)
                coords = np.where(image == label)
                x_c = coords[0].mean() - 15
                y_c = coords[1].mean() - 15
                temp_im[coords] = color_array[label - 1, color]
                letter_color = (int(color_array[label - 1, 0]), int(color_array[label - 1, 1]), int(color_array[label - 1, 2]))
                new_im = cv2.putText(new_im, str(label), org=(int(y_c), int(x_c)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=letter_color)
                new_im[:, :, color] += temp_im.astype(np.uint8)

        Image.fromarray(new_im).save(directory + '/colors/sequence_' + str(k).zfiill() + '.png')
        Image.fromarray((image).astype(np.uint16)).save(directory + '/labels/sequence_' + str(k).zfill(4) + '.png')
        return new_im

    def save_inference_w_super_position(self, image, raw_image, directory="images", k=0):
        if(len(raw_image.shape) == 2):
            raw_image = np.stack((raw_image, raw_image, raw_image), axis=2)
        np.random.seed(seed=0)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(directory + "/labels"):
            os.makedirs(directory + "/labels")

        if not os.path.exists(directory + "/colors"):
            os.makedirs(directory + "/colors")

        if not os.path.exists(directory + "/superimposed"):
            os.makedirs(directory + "/superimposed")

        # if not os.path.exists(directory + "/raw_data"):
        #    os.makedirs(directory + "/raw_data")

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

        Image.fromarray(new_im).save(directory + '/colors/sequence_' + str(k).zfill(4) + '.png')
        Image.fromarray((image).astype(np.uint16)).save(directory + '/labels/sequence_' + str(k).zfill(4) + '.png')
        Image.fromarray((super_imposed).astype(np.uint8)).save(directory + '/superimposed/sequence_' + str(k).zfill(4) + '.png')
        # Image.fromarray((raw_image).astype(np.uint8)).save(directory + '/raw_data/sequence_' + str((k).zfill(4) + '.png')

    def plot_with_pyplot(self, im_array, inference_image, directory=".", k=None, plot_stuff=True):
        if(plot_stuff):
            plt.figure(1)
            plt.clf()
            plt.imshow(im_array)

            plt.figure(2)
            plt.clf()
            plt.imshow(inference_image)

            plt.pause(0.01)

        if(k is not None):
            if not os.path.exists(directory + '/distributions'):
                os.makedirs(directory + '/distributions')

            temp_inf = np.zeros(im_array.shape, dtype=im_array.dtype)
            offset = im_array.shape[0] // 2 - 1
            temp_inf[offset:-1, :, :] = inference_image
            im_array[offset, :, :] = 255
            Image.fromarray(im_array + temp_inf).save(directory + '/distributions/sequence_' + str(k).zfill(4) + '.png')

    def plot_distributions_overalap(self, im_array, inference_image, directory=".", k=None, plot_stuff=False, name=''):
        if k is not None:
            if not os.path.exists(directory + '/distributions_overlap' + name):
                os.makedirs(directory + '/distributions_overlap' + name)

            num_channels = len(inference_image.shape)
            if(num_channels == 2):
                im_to_plot = np.stack((inference_image, inference_image, inference_image), axis=2).astype(np.float)
            else:
                im_to_plot = inference_image.astype(np.float)

            im_to_plot = ((im_to_plot / im_to_plot.max()) * 255).astype(np.uint8)
            # im_array = im_array[0:im_array.shape[0] // 2, ...]

            temp_im = np.maximum(im_array, im_to_plot).astype(np.uint8)
            Image.fromarray(temp_im).save(directory + '/distributions_overlap' + name + '/sequence_' + str(k).zfill(4) + '.png')


    def plot_double_distributions_overalap(self, im_array, im_array2, inference_image, directory=".", k=None, plot_stuff=True):
        if k is not None:
            if not os.path.exists(directory + '/birth_field'):
                os.makedirs(directory + '/birth_field')

            im_to_plot = np.stack((inference_image, inference_image, inference_image), axis=2).astype(np.float)
            im_to_plot = ((im_to_plot / im_to_plot.max()) * 255).astype(np.uint8)

            # im_array = im_array[0:im_array.shape[0] // 2, ...]
            # im_array2 = im_array2[0:im_array2.shape[0] // 2, ...]

            temp_im = np.maximum(im_array, im_to_plot).astype(np.uint8)
            # temp_im2 = np.maximum(im_array2, im_to_plot).astype(np.uint8)
            temp_im2 = cv2.addWeighted(im_array2, 0.8, im_to_plot, 0.7, 0)

            temp_im = np.concatenate((temp_im, temp_im2), axis=1)
            Image.fromarray(temp_im).save(directory + '/birth_field/sequence_' + str(k).zfill(4) + '.png')


    def save_distribution_dictionaries(self, D_preds, D_updates, Zs):
        result_string = ""
        for k in range(len(Zs)):
            result_string = result_string + "Time Step:" + str(k + 1) + "\n"

            Z = Zs[k]
            result_string = result_string + "Measures: \n"
            temp_string = ""
            for i in range(Z.shape[0]):
                temp_string = temp_string + "[{:.2f}, {:.2f}],  ".format(Z[i, 0], Z[i, 1])
            result_string = result_string + temp_string + "\n"

            D_pred = D_preds[k]
            result_string = result_string + "Pred: \n"
            if('Hs' in D_pred.keys()):
                for i in range(D_pred['Hs']):
                    temp_string = "{}: {:.2f} * [{:.2f}, {:.2f}], std=[{:.2f},{:.2f},{:.2f},{:.2f}] ".format(D_pred['Ls'][i], D_pred['ws'][i], D_pred['mus'][i][0], D_pred['mus'][i][1], D_pred['Ps'][i][0, 0], D_pred['Ps'][i][0, 1], D_pred['Ps'][i][1, 0], D_pred['Ps'][i][1, 1])
                    result_string = result_string + temp_string + "\n"
            D_updt = D_updates[k]
            result_string = result_string + "Updt: \n"
            if('Hs' in D_updt.keys()):
                for i in range(D_updt['Hs']):
                    temp_string = "{}: {:.2f} * [{:.2f}, {:.2f}], std=[{:.2f},{:.2f},{:.2f},{:.2f}] ".format(D_updt['Ls'][i], D_updt['ws'][i], D_updt['mus'][i][0], D_updt['mus'][i][1], D_updt['Ps'][i][0, 0], D_updt['Ps'][i][0, 1], D_updt['Ps'][i][1, 0], D_updt['Ps'][i][1, 1])
                    result_string = result_string + temp_string + "\n"
            result_string = result_string + "============================================\n\n\n"

        file = open("debug.txt", "w")
        file.writelines(result_string)
        file.close()

        return

    def save_individual_distributions(self, D_updates, label, directory, subsample=3):
        image = np.zeros((2 * self.canvas_shape[0], self.canvas_shape[1]))
        image_inference = np.zeros((2 * self.canvas_shape[0], self.canvas_shape[1], 3))
        counter = 0
        for D_udpt in D_updates:
            if(counter % subsample == 0):
                image = np.maximum(image, self.generate_mixture_fast(D_udpt, label_specific=label))
                image_inference = np.maximum(image_inference, inference_w_ellipsis_for_label(D_udpt, image, label))
            counter += 1

        if not os.path.exists(directory + "/single_tracks"):
            os.makedirs(directory + "/single_tracks")

        image_to_save = np.zeros((image.shape[0], image.shape[1], 3))
        image_to_save[:, :, 1] = (image / image.max()) * 255

        image_to_save += image_inference

        Image.fromarray((image_to_save).astype(np.uint8)).save(directory + '/single_tracks/track_' + str(labael).zfill(4) + '.png')
        return image


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

    def update_vector_field(self, vector_field, X, k):
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




def intermediates(p1, p2, nb_points=5):
    """"Return a list of nb_points equally spaced points
    between p1 and p2"""
    # If we have 8 intermediate points, we have 8+1=9 spaces
    # between p1 and p2
    x_spacing = (p2[0] - p1[0]) / (nb_points + 1)
    y_spacing = (p2[1] - p1[1]) / (nb_points + 1)

    return [[int(p1[0] + i * x_spacing), int(p1[1] +  i * y_spacing)] 
            for i in range(1, nb_points+1)]