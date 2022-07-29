from scipy.stats import invgamma
import numpy as np
import torch
import math

from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


def multivariate_normal_vector(Zk, mu, S_inv, denominator):
    diff = (Zk - mu).unsqueeze(2)
    diff_t = torch.transpose(diff, 1, 2)
    mult1 = torch.matmul(diff_t, S_inv)
    power = torch.matmul(mult1, diff)[:, 0, 0]
    p_x = torch.exp( -0.5 * power) / denominator
    return p_x

class Model:
    def __init__(self, parameters, device=None):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        #  basic parameters
        self.x_dim = 6  # dimension of state vector
        self.z_dim = 2  # dimension of observation vector

        #  dynamical model parameters (CV model)
        tau = parameters['tau']
        Q = parameters['Q']
        self.F = torch.tensor([[1, 0, tau, 0, 0, 0],
                               [0, 1, 0, tau, 0, 0],
                               [0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 1]]).float().to(device)

        self.F_T = torch.transpose(self.F, 0, 1)

        self.Q = torch.tensor([[tau**4 / 4.0 * Q, 0, tau**3 / 2.0 * Q, 0, 0, 0],
                                [0, tau**4 / 4.0 * Q, 0, tau**3 / 2.0 * Q, 0, 0],
                                [tau**3 / 2.0 * Q, 0, tau**2 * Q, 0, 0, 0],
                                [0, tau**3 / 2.0 * Q, 0, tau**2 * Q, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]
                                ]).float().to(device)

        # survival/death parameters
        self.P_S= .95
        self.Q_S=  1 - self.P_S

        #  birth parameters (LMB birth model, single component only)
        self.T_birth= 4        # no. of LMB birth terms
        '''
        self.L_birth= zeros(self.T_birth,1)                                         # no of Gaussians in each LMB birth term
        self.r_birth= zeros(self.T_birth,1)                                         # prob of birth for each LMB birth term
        self.w_birth= cell(self.T_birth,1)                                          # weights of GM for each LMB birth term
        self.m_birth= cell(self.T_birth,1)                                          # means of GM for each LMB birth term
        self.B_birth= cell(self.T_birth,1)                                          # std of GM for each LMB birth term
        self.P_birth= cell(self.T_birth,1)                                          # cov of GM for each LMB birth term

        self.L_birth(1)=1                                                            # no of Gaussians in birth term 1
        self.r_birth(1)=0.03                                                         # prob of birth
        self.w_birth{1}(1,1)= 1                                                      # weight of Gaussians - must be column_vector
        self.m_birth{1}(:,1)= [ 724 0 56 1 ]                                     # mean of Gaussians
        self.B_birth{1}(:,:,1)= diag([ 10 10 10 10 ])                             # std of Gaussians
        self.P_birth{1}(:,:,1)= self.B_birth{1}(:,:,1)*self.B_birth{1}(:,:,1)'     # cov of Gaussians
        '''
    
        # Birth Rate        
        self.r_birth = 0.90

        #  observation model parameters (noisy x/y only)
        self.H = torch.tensor([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]
                      ]).float().to(device)

        self.H_T = torch.transpose(self.H, 0, 1)

        R = parameters['R']
        self.R = torch.tensor([[R, 0, 0, 0],
                               [0, R, 0, 0],
                               [0, 0, R, 0],
                               [0, 0, 0, R]]).to(device)  # Measurement Noise

        #  detection parameters
        self.P_D = 0.98                                  # probability of detection in measurements
        self.Q_D = 1 - self.P_D

        # clutter parameters
        self.lambda_c = 30                               # poisson average rate of uniform clutter (per scan)
        self.range_c = None                              # [ -1000 1000; -1000 1000 ]  uniform clutter region
        self.pdf_c = 0.00000025                          # 1/prod(self.range_c(:,2)-self.range_c(:,1))  uniform clutter density

        self.H_upd = 100                                 # requested number of updated components/hypotheses
        self.Po = parameters['Po']


class Filter:
    def __init__(self, model):
        # filter parameters
        self.H_upd = 100                                 # requested number of updated components/hypotheses
        self.H_max = 500                                 # cap on number of posterior components/hypotheses
        self.hyp_threshold = 1e-15                       # pruning threshold for components/hypotheses


        self.L_max= 100                                  # limit on number of Gaussians in each track - not implemented yet
        self.elim_threshold= 1e-5                        # pruning threshold for Gaussians in each track - not implemented yet
        self.merge_threshold= 4                          # merging threshold for Gaussians in each track - not implemented yet

        self.z_dim = 2

        self.P_G= 0.9999999;                             # gate size in percentage
        self.gamma= invgamma(self.P_G, model.z_dim);     # inv chi^2 dn gamma value
        self.gate_flag= 1;                               # gating on or off 1/0

        self.Po = 5


class Track_List:
    def __init__(self, model=None, N_init=None, device=None):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if(N_init is None):
            self.mus = torch.zeros([], device=device)                               # Gaussian Means                            | mus is tensor of shape [Num_Ts x 6 x 1]
            self.Ps = torch.tensor([], device=device)                                # Gaussian Covs                             | Ps is tensor of shape [Num_Ts x 6, 6]
            self.ws = torch.tensor([], device=device)                                # weights of Gaussians                      | ws is tensor of shape [Num_Ts x 1]
            self.Ls = torch.tensor([], device=device)                                # track labels                              | Ls is list of lists. len(Ls) = Num_Ts
            self.Es = [None]                                         # track association history                 | Es is a list of lists. Len(Es) = Num_Ts
            self.Num_Ts = 0                                           # Number of tracks
        else:
            self.mus = torch.zeros((N_init, model.x_dim, 1), device=device)
            self.Ps = torch.zeros((N_init, model.x_dim, model.x_dim), device=device)
            self.ws = torch.zeros((N_init, 1), device=device)
            self.Ls = [None] * N_init
            self.Es = [None] * N_init
            self.Num_Ts = N_init

        self.Gated_m = torch.tensor([])                # Gated Measurements

    def kalman_prediction(self, model):
        '''
            Advance State according to F (F transpose) and Q matrices
        '''
        if(self.Num_Ts > 0):
            self.mus = torch.matmul(model.F, self.mus)
            self.Ps = torch.matmul(torch.matmul(model.F, self.Ps), model.F_T) + model.Q


    def kalman_update_2(self, pred_tracks, model, Zk, PHD_FILTER=False):
        '''
            Update State According to model parameters
        '''

        device = Zk.device
        # ADD MISDETECTION TRACKS
        if pred_tracks.Num_Ts == 0:
            return torch.zeros((0, Zk.shape[0]), device=device)
        indexes = torch.arange(0, pred_tracks.Num_Ts).long()
        self.mus[indexes] = pred_tracks.mus[indexes]
        self.Ps[indexes] = pred_tracks.Ps[indexes]

        if PHD_FILTER:
            self.ws[indexes] = ((1 - model.P_D) * pred_tracks.ws[indexes]).float()
        else:
            self.ws[indexes] = pred_tracks.ws[indexes].float()

        for tabidx in range(pred_tracks.Num_Ts):
            self.Ls[tabidx] = pred_tracks.Ls[tabidx]
            if(pred_tracks.Es[tabidx] is None):
                self.Es[tabidx] = [-1]
            else:
                self.Es[tabidx] = pred_tracks.Es[tabidx] + [-1]  # -1 means undetected track

        # ADD UPDATED WEIGHTS
        if pred_tracks.mus.shape[0] == 0:
            return torch.zeros((0, Zk.shape[0]), device=device)
        z_pred = torch.matmul(model.H, pred_tracks.mus)
        S_pred = torch.matmul(torch.matmul(model.H, pred_tracks.Ps), model.H_T) + model.R

        S_inv = torch.inverse(S_pred)
        P_H = torch.matmul(pred_tracks.Ps, model.H_T)
        K_S_inv = torch.matmul(P_H, S_inv)

        K_gain = torch.matmul(K_S_inv, model.H)
        P_updt = torch.matmul((torch.eye(model.H.shape[1], device=device) - K_gain), pred_tracks.Ps)


        Num_Ts_pred = pred_tracks.Num_Ts
        ms = Zk.shape[0]
        # Zk = torch.tensor(Zk)
        allcostm = torch.zeros((Num_Ts_pred, ms), device=device)

        # pre calculations for speed purposes
        det = torch.det(S_pred)
        S_pred_inv = torch.inverse(S_pred)
        two_pi_k = np.pi**S_pred.shape[-1]

        denominators = torch.sqrt(two_pi_k * det)

        if PHD_FILTER:
            normalizing_weights = torch.zeros(ms, device=denominators.device)

        for i in range(Num_Ts_pred):
            if pred_tracks.Gated_m[i].shape[0] > 0 :
                Zjs = Zk[pred_tracks.Gated_m[i], :]
            else:
                continue

            if PHD_FILTER:
                w_vector = model.P_D * pred_tracks.ws[i, :] * multivariate_normal_vector(Zjs, mu=z_pred[i, :, 0], S_inv=S_pred_inv[i, ...], denominator=denominators[i])
                normalizing_weights[pred_tracks.Gated_m[i]] += w_vector
            else:
                w_vector = multivariate_normal_vector(Zjs, mu=z_pred[i, :, 0], S_inv=S_pred_inv[i, ...], denominator=denominators[i])

            '''
            diff_old = (Zjs - z_pred[i, :, 0]).unsqueeze(2)
            diff = (Zjs - z_pred).unsqueeze(2)
            
            diff_t_old = torch.transpose(diff_old, 1, 2)
            diff_t = torch.transpose(diff, 1, 2)
            
            diff_t = 4, 1, 4
            S_pred = 4, 4
            diff = 4, 4, 1
            # power = torch.matmul(torch.matmul(diff_t.unsqueeze(1), S_pred_inv), diff)[:, 0, 0]
            temp2 = torch.matmul(diff_t.unsqueeze(1), S_pred_inv)
            power = torch.matmul(temp2.transpose(0, 1), diff)[:, :, 0, 0]
            p_x = torch.exp( -0.5 * power) / denominator
            '''
            mu_temp = pred_tracks.mus[i] + torch.matmul(K_S_inv[i, :, :], (Zjs.float() - z_pred[i, :, 0]).t())


            # Populated updated tracks
            # num_gated_measurements = Zjs.shape[0] # offset_idx = Num_Ts_pred * torch.arange(1, num_gated_measurements + 1, device=w_vector.device) + i
            offset_idx = Num_Ts_pred * (pred_tracks.Gated_m[i] + 1) + i
            self.mus[offset_idx, :, 0] = mu_temp.t()
            self.Ps[offset_idx, :, :] = P_updt[i, :, :]
            self.ws[offset_idx, 0] = w_vector.float()

            counter_j = 0
            for j in pred_tracks.Gated_m[i]:
                stoidx = Num_Ts_pred * (j + 1) + i
                self.Ls[stoidx] = pred_tracks.Ls[i]
                if pred_tracks.Es[i] is None:
                    self.Es[stoidx] = [j]
                else:
                    self.Es[stoidx] = pred_tracks.Es[i] + [j]

                allcostm[i, j] = w_vector[counter_j]
                counter_j += 1

        if PHD_FILTER:
            # Normalize hypothesis-measurement association weights
            for i in range(Num_Ts_pred):
                for j in pred_tracks.Gated_m[i]:
                    stoidx = Num_Ts_pred * (j + 1) + i
                    self.ws[stoidx, 0] = self.ws[stoidx, 0] / normalizing_weights[j]

        return allcostm

    def appearance_update(self, pred_tracks, Zk, apps_fts_k):
        '''
            Update State According to high level features
        '''

        device = Zk.device
        Num_Ts_pred = pred_tracks.Num_Ts
        ms = Zk.shape[0]
        sigma_2 = 4.0
        appearance_cost = torch.ones((Num_Ts_pred, ms), device=device) * (1.0 / (np.sqrt(2 * np.pi * sigma_2)) * np.exp(-1.0 / (2.0 * sigma_2)))

        rows, cols, chans = apps_fts_k.shape
        for i in range(Num_Ts_pred):
            pred_i, pred_j, pred_w, pred_h = pred_tracks.mus[i, [0, 1, 4, 5], 0].int()
            pred_w = 3
            pred_h = 3
            # index appearance matrix (5 x 5 x 512)
            if(pred_i + pred_h >= rows or pred_j + pred_w >= cols or pred_i < 0 or pred_j < 0):
                continue
            pred_vector = apps_fts_k[pred_i:pred_i + pred_w, pred_j:pred_j + pred_h, :].flatten()
            pred_vector = pred_vector / torch.norm(pred_vector) # appearance_vector = 1 x 12800
            for j in pred_tracks.Gated_m[i]:
                meas_i, meas_j, meas_w, meas_h = Zk[j, :].int()
                if(meas_i + meas_h >= rows or meas_j + meas_w >= cols):
                    continue
                meas_h = 3
                meas_w = 3
                meas_vector = apps_fts_k[meas_i:meas_i + meas_w, meas_j:meas_j + meas_h, :].flatten()
                meas_vector = meas_vector / torch.norm(meas_vector)
                dot_prod = torch.dot(meas_vector, pred_vector)
                s_i_j = 1.0 - dot_prod
                theta_i_j = 1.0 / (np.sqrt(2 * np.pi * sigma_2)) * torch.exp(-s_i_j / (2.0 * sigma_2))
                appearance_cost[i, j] = theta_i_j
        return appearance_cost

    def kalman_update(self, pred_tracks, model, Zk):
        '''
            Update State According to model parameters
        '''

        device = Zk.device
        # ADD MISDETECTION TRACKS
        for tabidx in range(pred_tracks.Num_Ts):
            self.mus[tabidx] = pred_tracks.mus[tabidx]
            self.Ps[tabidx] = pred_tracks.Ps[tabidx]
            self.ws[tabidx] = pred_tracks.ws[tabidx]
            self.Ls[tabidx] = pred_tracks.Ls[tabidx]

            if(pred_tracks.Es[tabidx] is None):
                self.Es[tabidx] = [-1]
            else:
                self.Es[tabidx] = pred_tracks.Es[tabidx] + [-1]  # -1 means undetected track

        # ADD UPDATED WEIGHTS
        H = model.H
        H_T = model.H_T

        if(pred_tracks.mus.shape[0] == 0):
            return torch.zeros((0, Zk.shape[0]))
        z_pred = torch.matmul(model.H, pred_tracks.mus)                                              # Num_Ts_pred x Z_dim x 1
        S_pred  = torch.matmul(torch.matmul(model.H, pred_tracks.Ps), model.H_T) + model.R          # Num_Ts_pred x Z_dim x Z_dim

        S_inv = torch.inverse(S_pred)                                                                # Num_Ts_pred x Z_dim x Z_dim
        P_H = torch.matmul(pred_tracks.Ps, model.H_T)                                                # Num_Ts_pred x X_dim x Z_dim
        K_S_inv = torch.matmul(P_H, S_inv)                                                           # Num_Ts_pred x X_dim x Z_dim

        K_gain = torch.matmul(K_S_inv, model.H)                                                     # Num_Ts_pred x X_dim x X_dim
        P_updt = torch.matmul((torch.eye(model.H.shape[1], device=device) - K_gain), pred_tracks.Ps)               # Num_Ts_pred x X_dim x X_dim


        Num_Ts_pred = pred_tracks.Num_Ts
        ms = Zk.shape[0]
        Zk = torch.tensor(Zk)
        allcostm = torch.zeros((Num_Ts_pred, ms), device=device)

        for i in range(Num_Ts_pred):
            for j in pred_tracks.Gated_m[i]:
                w_temp = pred_tracks.ws[i, :] * torch.tensor(multivariate_normal.pdf(Zk[j, :].cpu(), mean=z_pred[i, :, 0].cpu(), cov=S_pred[i, :, :].cpu()), device=device).unsqueeze(-1)
                mu_temp = pred_tracks.mus[i] + torch.matmul(K_S_inv[i, :, :], (Zk[j, :].float() - z_pred[i, :, 0])).unsqueeze(-1)                        # X_dim x 1
                P_temp = P_updt[i, :, :] 

                # Populated updated tracks
                stoidx = Num_Ts_pred * (j + 1) + i
                self.mus[stoidx] = mu_temp
                self.Ps[stoidx] = P_temp
                self.ws[stoidx] = w_temp / (w_temp + 0.00000001)
                self.Ls[stoidx] = pred_tracks.Ls[i]
                if(pred_tracks.Es[i] is None):
                    self.Es[stoidx] = [j.item()]
                else:
                    self.Es[stoidx] = pred_tracks.Es[i] + [j.item()]

                allcostm[i, j] = w_temp
                # self.ws[i, :] = w_temp / (w_temp + 0.00000001)

        allcostm = allcostm
        return allcostm

    def get_gated_measurements(self, Zk, distance_thres=30, perform_gating=True):
        '''
            Update Gated Measurements (TO DO)
        '''
        if len(self.mus.shape) > 0 and Zk.shape[0] > 0 and perform_gating:
            # Do measurement - component association
            mus = self.mus[:, 0:2, 0]
            cost = torch.cdist(mus, Zk[:, 0:2], p=2)
            gates = cost < distance_thres
            Gated_m = []
            for com in range(gates.shape[0]):
                gate_indexes = gates[com, :].nonzero()
                if gate_indexes.shape[0] > 0:
                    Gated_m.append(gate_indexes[:, 0])
                else:
                    Gated_m.append(torch.tensor([], device=self.mus.device))
        else:
            Gated_m = [torch.arange(Zk.shape[0], device=self.mus.device)] * self.Num_Ts
        self.Gated_m = Gated_m

    def index_w_tensor(self, indexes, wb=1.0):
        new_track_list = Track_List()
        new_track_list.mus = self.mus[indexes, :, :]
        new_track_list.Ps = self.Ps[indexes, :, :]
        new_track_list.ws = wb * self.ws[indexes]
        new_track_list.Ls = [self.Ls[i] for i in indexes]
        new_track_list.Es = [self.Es[i] for i in indexes]
        new_track_list.Num_Ts = len(new_track_list.Ls)
        return new_track_list


class glmb_instance:
    def __init__(self, track_list=None, device=None):

        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # initial prior
        self.tt = Track_List()                      # track table for GLMB (cell array of structs for individual tracks)
        self.w = torch.tensor([], device=device)                  # vector of GLMB component/hypothesis weights
        self.I = None

        # cell of GLMB component/hypothesis labels (labels are indices/entries in track table)
        self.n = []                                  # vector of GLMB component/hypothesis cardinalities
        self.cdn = []                               # cardinality distribution of GLMB (vector of cardinality distribution probabilities)


        if(track_list is not None):
            self.tt = track_list

        if track_list is None:
            self.w = torch.tensor([1])
            self.n = [0]
            self.cdn = [1]

    def prune(self, threshold=0.01):
        # prune components with weights lower than specified threshold
        idxkeep = (self.w > threshold).nonzero()[:, 0]

        # Update I SETS
        if self.I is not None:
            self.I = [self.I[i] for i in idxkeep]

        self.w = self.w[idxkeep]
        norm_w = self.w.sum()
        self.w = self.w / norm_w
        self.n = self.n[idxkeep]

        # Recalculate Cardinality Distribution
        temp_cdn = []
        for card in range(self.n.max() + 1):
            card_bin = (self.w[self.n == card]).sum()
            temp_cdn.append(card_bin)

        self.cdn = torch.tensor(temp_cdn)

    def extract_estimates(self):
        '''
            extract estimates via best cardinality, then
            best component/hypothesis given best cardinality, then
            best means of tracks given best component/hypothesis and cardinality
        '''
        Dk = {}
        Dk['mus'] = []
        Dk['ws'] = []
        Dk['Ps'] = []
        Dk['Ls'] = []
        Dk['Hs'] = 0

        N = torch.argmax(self.cdn).item()
        if self.tt.Num_Ts == 0:
            return Dk

        X = torch.zeros((self.tt.mus[0].shape[0], N))
        L = torch.zeros((2, N))

        idxcmp = torch.argmax(self.w * (self.n == N))

        X = {}
        for n in range(N):
            track_index = self.I[idxcmp][n].long()
            label = self.tt.Ls[track_index][1]
            X[label] = self.tt.mus[track_index, :, 0].cpu().numpy()

            Dk['mus'].append(self.tt.mus[track_index, :, 0].cpu().numpy())
            Dk['Ls'].append(torch.tensor(self.tt.Ls[track_index])[1].item())
            Dk['ws'].append(1)
            Dk['Ps'].append(self.tt.Ps[track_index])

        Dk['Hs'] = N
        return X


    def extract_all_estimates(self):
        '''
            extract estimates via best cardinality, then
            best component/hypothesis given best cardinality, then
            best means of tracks given best component/hypothesis and cardinality
        '''
        Dk = {}
        Dk['mus'] = []
        Dk['ws'] = []
        Dk['Ps'] = []
        Dk['Ls'] = []
        Dk['Hs'] = 0

        N = torch.argmax(self.cdn)
        if(self.tt.Num_Ts == 0):
            return Dk

        for track_index in range(self.tt.Num_Ts):
            track_index = track_index

            Dk['mus'].append(self.tt.mus[track_index, :, 0].cpu().numpy())
            Dk['Ls'].append(torch.tensor(self.tt.Ls[track_index])[1].item())
            Dk['ws'].append(1)
            Dk['Ps'].append(torch.tensor(self.tt.Ps[track_index]))

        Dk['Hs'] = self.tt.Num_Ts
        return Dk

    def inference(self, all=False):
        '''

        :param all:
        :return: returns dictionary Dk['Hs'], Dk['mus'], Dk['Ps']...etc
        '''
        if all:
            return self.extract_all_estimates()
        else:
            return self.extract_estimates()


# This code was heavily based on Vo's implementation
def jointpredictupdate_a_birth(glmb_update, model, Zk, birth_field, k):
    '''
        Generate next glmb state
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    Zk = torch.tensor(Zk, device=device, dtype=torch.float32)
    # apps_fts_k = torch.tensor(apps_fts_k, device=device)

    # print("Getting Measurements Surviving/Residual")
    # SURVIVING TRACKS: ADVANCE STATE
    tt_survive = glmb_update.tt
    tt_survive.kalman_prediction(model)

    # BIRTH TRACKS: MEASUREMENT DRIVEN PROPOSAL
    max_label = torch.tensor(glmb_update.tt.Ls)[:, 1].max().item() if glmb_update.tt.Num_Ts > 0 else 0

    tt_birth, Z_surviving = get_birth_gm_w_birth_field(Zk, birth_field, tt_survive.mus, current_label=max_label + 1, initial_P=model.Po, k=k, distance_thres=20)

    # PREDICTION TRACKS: CONCATENATION BIRTH + SURVIVAL
    tt_pred = concatenate_tracks(tt_birth, tt_survive)
    # GATE MEASUREMENT (NEED TO DO)
    tt_pred.get_gated_measurements(Zk, distance_thres=20, perform_gating=True)
    # TRACK UPDATE STEP
    m = Zk.shape[0]
    tt_update = Track_List(model=model, N_init=((1 + m) * tt_pred.Num_Ts))  # [(1 + m) * Num_Ts_predict]
    # ADD MEASUREMENT UPDATED TRACKS

    allcostm = tt_update.kalman_update_2(tt_pred, model, Zk)

    # appearance_cost = tt_update.appearance_update(tt_pred, Zk, apps_fts_k)
    # allcostm = allcostm * appearance_cost

    # JOINT COST MATRIX
    avps = torch.zeros((tt_pred.Num_Ts), device=device)  # precalculation for average survival/death probabilities
    avps[:tt_birth.Num_Ts] = model.r_birth
    avps[tt_birth.Num_Ts:] = model.P_S
    avqs= 1 - avps

    avpd = model.P_D * torch.ones((tt_pred.Num_Ts), device=device) # precalculation loop for average detection/missed probabilities
    avqd= 1 - avpd

    jointcostm = torch.cat((torch.diag(avqs),
                            torch.diag(avps * avqd),
                           (avps * avpd).unsqueeze(1).repeat(1, m) * allcostm / (model.lambda_c * model.pdf_c)), 1)

    # GATED MEASUREMENT INDEX
    gatemeasidxs = torch.zeros((tt_pred.Num_Ts, m), device=device)
    for tabidx in range(tt_pred.Num_Ts):
        gatemeasidxs[tabidx, 0:len(tt_pred.Gated_m[tabidx])] = tt_pred.Gated_m[tabidx]
    gatemeasindc = gatemeasidxs >= 0

    # GLMB COMPONENTS UPDATE
    glmb_posterior = glmb_instance(track_list=tt_update)
    runidx = 1
    for pidx in range(len(glmb_update.w)):
        # calculate best updated hypotheses/components
        cpreds = tt_pred.Num_Ts
        nbirths= tt_birth.Num_Ts
        if(glmb_update.I is not None):
            nexists = len(glmb_update.I[pidx])
        else:
            nexists = 0

        ntracks= nbirths + nexists

        if glmb_update.I is not None:
            offstet_indices = glmb_update.I[pidx] + nbirths
        else:
            offstet_indices = torch.tensor([])
        tindices = torch.cat((torch.arange(nbirths), offstet_indices)).long().to(device)
        lselmask = torch.zeros((tt_pred.Num_Ts, m), dtype=torch.bool, device=device)  # logical selection mask to index gating matrices
        lselmask[tindices, :] = gatemeasindc[tindices, :]  # logical selection mask to index gating matrices

        mindices = torch.unique(gatemeasidxs[lselmask])  # union indices of gated measurements for corresponding tracks
        col_idxs = torch.cat((tindices, cpreds + tindices, (2 * cpreds) + mindices)).long()
        costm = jointcostm[tindices, :]  # cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
        costm = costm[:, col_idxs]
        neglogcostm = -torch.log(costm)  # negative log cost

        N_hat = max(torch.round(model.H_upd * torch.sqrt(glmb_update.w[pidx]) / torch.sqrt(glmb_update.w).sum()).int().item(), 1)
        uasses, nlcost = mbestwrap_updt_gibbsamp(neglogcostm, N_hat)  # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        # nlcost is of shape [1, 3] (it represents the cost of the possible combinations (3 combs))
        uasses[uasses < ntracks] = float('-inf')  # set not born/track deaths to -inf assignment
        uasses[torch.logical_and(uasses >= ntracks, uasses < 2 * ntracks)] = -1  # set survived & missed to -1
        uasses[uasses >= 2 * ntracks] = uasses[uasses >= 2 * ntracks] - 2 * ntracks  # set survived+detected to 1:|Z|
        uasses[uasses >= 0] = mindices[uasses[uasses >= 0].long()]  # restore original indices of gated measurements

        # jointcostm: the larger the better" ie. [0.1, 0.01, 7.44
        # nlcost: the more negative the better
        # GENERATE JOINTLY PREDICTED/UPDATED hypotheses/components
        for hidx in range(nlcost.shape[1]):
            update_hypcmp_tmp = uasses[hidx, :]
            if(glmb_update.I is not None):
                offset_labels = (nbirths + glmb_update.I[pidx]).to(device)
                temp = torch.cat((torch.arange(0, nbirths, device=device), offset_labels))
            else:
                temp = torch.arange(0, nbirths, device=device)
            update_hypcmp_idx = (cpreds * (update_hypcmp_tmp + 1)) + temp

            clutter_weight = -model.lambda_c + m * math.log(model.lambda_c * model.pdf_c)
            track_weight = math.log(glmb_update.w[pidx] + 1e-10)
            hypothesis_weight = nlcost[0, hidx]
            new_weight = clutter_weight + track_weight - hypothesis_weight
            glmb_posterior.w = torch.cat((glmb_posterior.w, torch.tensor([new_weight], device=device)))
            if glmb_posterior.I is None:
                glmb_posterior.I = [update_hypcmp_idx[update_hypcmp_idx >= 0]]
            else:
                glmb_posterior.I.append(update_hypcmp_idx[update_hypcmp_idx >= 0])
            glmb_posterior.n.append(torch.sum(update_hypcmp_idx >= 0))

            runidx = runidx + 1

    glmb_posterior.w = torch.exp(glmb_posterior.w - logsumexp(glmb_posterior.w))

    glmb_posterior.n = torch.tensor(glmb_posterior.n, device=device)
    # extract cardinality distribution
    for card in range(max(glmb_posterior.n) + 1):
        card_bin = (glmb_posterior.w[glmb_posterior.n == card]).sum()
        glmb_posterior.cdn.append(card_bin)

    glmb_posterior.cdn = torch.tensor(glmb_posterior.cdn)
    # remove duplicate entries and clean track table
    glmb_posterior = clean_update(clean_predict(glmb_posterior))

    glmb_posterior.prune(threshold=0.01)
    return glmb_posterior


def jointpredictupdate(glmb_update, model, filter_prameters, Zk, k):
    '''
        Generate next glmb state
    '''
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    Zk = torch.tensor(Zk, device=device, dtype=torch.float32)
    # SURVIVING TRACKS: ADVANCE STATE
    tt_survive = glmb_update.tt
    tt_survive.kalman_prediction(model)

    # BIRTH TRACKS: MEASUREMENT DRIVEN PROPOSAL
    if glmb_update.tt.Num_Ts > 0:
        max_label = torch.tensor(glmb_update.tt.Ls)[:, 1].max().item()
    else:
        max_label = 0

    tt_birth, Z_surviving = get_birth_gm_from_Z_disk_torch(Zk, glmb_update.tt.mus, current_label=max_label + 1, initial_P=filter_prameters.Po, k=k, distance_thres=20)

    # PREDICTION TRACKS: CONCATENATION BIRTH + SURVIVAL
    tt_pred = concatenate_tracks(tt_birth, tt_survive)

    # GATE MEASUREMENT (NEED TO DO)
    tt_pred.get_gated_measurements(Zk)

    # TRACK UPDATE STEP
    m = Zk.shape[0]
    tt_update = Track_List(model=model, N_init=((1 + m) * tt_pred.Num_Ts))  # [(1 + m) * Num_Ts_predict]

    # ADD MEASUREMENT UPDATED TRACKS
    allcostm = tt_update.kalman_update_2(tt_pred, model, Zk)

    # all_costm2 = tt_update.kalman_update(tt_pred, model, Zk)
    # JOINT COST MATRIX
    avps = torch.zeros((tt_pred.Num_Ts), device=device)  # precalculation for average survival/death probabilities
    avps[:tt_birth.Num_Ts] = model.r_birth
    avps[tt_birth.Num_Ts:] = model.P_S
    avqs= 1 - avps

    avpd = model.P_D * torch.ones((tt_pred.Num_Ts), device=device) # precalculation loop for average detection/missed probabilities
    avqd= 1 - avpd

    jointcostm = torch.cat((torch.diag(avqs),
                            torch.diag(avps * avqd),
                           (avps * avpd).unsqueeze(1).repeat(1, m) * allcostm / (model.lambda_c * model.pdf_c)), 1)

    # GATED MEASUREMENT INDEX
    gatemeasidxs = torch.zeros((tt_pred.Num_Ts, m), device=device)
    for tabidx in range(tt_pred.Num_Ts):
        gatemeasidxs[tabidx, 0:len(tt_pred.Gated_m[tabidx])] =tt_pred.Gated_m[tabidx]
    gatemeasindc = gatemeasidxs >= 0

    # GLMB COMPONENTS UPDATE
    glmb_posterior = glmb_instance(track_list=tt_update)
    runidx = 1
    for pidx in range(len(glmb_update.w)):
        # calculate best updated hypotheses/components
        cpreds = tt_pred.Num_Ts
        nbirths= tt_birth.Num_Ts
        if(glmb_update.I is not None):
            nexists = len(glmb_update.I[pidx])
        else:
            nexists = 0

        ntracks= nbirths + nexists 
        
        if glmb_update.I is not None:
            offstet_indices = glmb_update.I[pidx] + nbirths
        else:
            offstet_indices = torch.tensor([])
        tindices = torch.cat((torch.arange(nbirths), offstet_indices)).long().to(device)
        lselmask = torch.zeros((tt_pred.Num_Ts, m), dtype=torch.bool, device=device)  # logical selection mask to index gating matrices
        lselmask[tindices, :] = gatemeasindc[tindices, :]  # logical selection mask to index gating matrices

        mindices = torch.unique(gatemeasidxs[lselmask])  # union indices of gated measurements for corresponding tracks
        col_idxs = torch.cat((tindices, cpreds + tindices, (2 * cpreds) + mindices)).long()
        costm = jointcostm[tindices, :]  # cost matrix - [no_birth/is_death | born/survived+missed | born/survived+detected]
        costm = costm[:, col_idxs]
        neglogcostm = -torch.log(costm)  # negative log cost

        N_hat = max(torch.round(filter_prameters.H_upd * torch.sqrt(glmb_update.w[pidx]) / torch.sqrt(glmb_update.w).sum()).int().item(), 1)
        uasses, nlcost = mbestwrap_updt_gibbsamp(neglogcostm, N_hat)  # murty's algo/gibbs sampling to calculate m-best assignment hypotheses/components
        # nlcost is of shape [1, 3] (it represents the cost of the possible combinations (3 combs))
        uasses[uasses < ntracks] = float('-inf')  # set not born/track deaths to -inf assignment
        uasses[torch.logical_and(uasses >= ntracks, uasses < 2 * ntracks)] = -1  # set survived & missed to -1
        uasses[uasses >= 2 * ntracks] = uasses[uasses >= 2 * ntracks] - 2 * ntracks  # set survived+detected to 1:|Z|
        uasses[uasses >= 0] = mindices[uasses[uasses >= 0].long()]  # restore original indices of gated measurements

        # jointcostm: the larger the better" ie. [0.1, 0.01, 7.44
        # nlcost: the more negative the better
        # GENERATE JOINTLY PREDICTED/UPDATED hypotheses/components
        for hidx in range(nlcost.shape[1]):
            update_hypcmp_tmp = uasses[hidx, :]
            if(glmb_update.I is not None):
                offset_labels = nbirths + torch.tensor(glmb_update.I[pidx], device=device)
                temp = torch.cat((torch.arange(0, nbirths, device=device), offset_labels))
            else:
                temp = torch.arange(0, nbirths, device=device)
            update_hypcmp_idx = (cpreds * (update_hypcmp_tmp + 1)) + temp

            clutter_weight = -model.lambda_c + m * math.log(model.lambda_c * model.pdf_c)
            track_weight = math.log(glmb_update.w[pidx] + 1e-10)
            hypothesis_weight = nlcost[0, hidx]
            new_weight = clutter_weight + track_weight - hypothesis_weight
            glmb_posterior.w = torch.cat((glmb_posterior.w, torch.tensor([new_weight], device=device)))
            if(glmb_posterior.I is None):
                glmb_posterior.I = [update_hypcmp_idx[update_hypcmp_idx >= 0]]
            else:
                glmb_posterior.I.append(update_hypcmp_idx[update_hypcmp_idx >= 0])                                                                                              # hypothesis/component tracks (via indices to track table)
            glmb_posterior.n.append(torch.sum(update_hypcmp_idx >= 0))                                                                                                            # hypothesis/component cardinality

            runidx = runidx + 1

    glmb_posterior.w = torch.exp(glmb_posterior.w - logsumexp(glmb_posterior.w))                                                                                                                 # normalize weights

    glmb_posterior.n = torch.tensor(glmb_posterior.n, device=device)
    # extract cardinality distribution
    for card in range(max(glmb_posterior.n) + 1):
        card_bin = (glmb_posterior.w[glmb_posterior.n == card]).sum()
        glmb_posterior.cdn.append(card_bin)                                                                                                       # extract probability of n targets

    glmb_posterior.cdn = torch.tensor(glmb_posterior.cdn)
    # remove duplicate entries and clean track table
    glmb_posterior = clean_update(clean_predict(glmb_posterior))

    return glmb_posterior


def logsumexp(w):
    val = w.max()
    return torch.log((torch.exp(w - val)).sum()) + val

def mbestwrap_updt_gibbsamp(P0, m, p=None):
    '''
        P0 is of shape [ (see table in paper)]
    '''
    device = P0.device
    n1, n2 = P0.shape
    assignments = torch.zeros(m, n1, device=device)
    costs = torch.zeros(1, m, device=device)
    
    currsoln = torch.arange(n1, 2 * n1, device=device)    # use all missed detections as initial solution
    assignments[0, :] = currsoln

    # in the case of an empty hypothesis
    if n1 == 0:
        return assignments[0, :].unsqueeze(0), costs[:, 0].unsqueeze(1)
    temp_cost = P0[:, currsoln]
    idx_to_gather = torch.arange(0, n1, device=device).unsqueeze(-1)
    temp_cost = torch.gather(temp_cost, dim=1, index=idx_to_gather)
    costs[0, 0] = temp_cost.sum()
    for sol in range(1, m):
        for var in range(n1):
            tempsamp = torch.exp(-P0[var, :])                           # grab row of costs for current association variable
            lock_idxs = torch.cat((currsoln[0:var], currsoln[var + 1:]))
            tempsamp[lock_idxs] = 0 # lock out current and previous iteration step assignments except for the one in question
            idx_old = (tempsamp > 0).nonzero(as_tuple=True)
            tempsamp = tempsamp[idx_old]
            cdf = tempsamp / tempsamp.sum()
            cdf_bins = torch.cat((torch.tensor([0], device=device), torch.cumsum(cdf, dim=0)), 0)
            sample = torch.rand(1, 1, device=device)
            bin_idx = (cdf_bins > sample).nonzero()[0][1] - 1           # Get first idx that is larger than sample (-1 becase we are also counting 0)
            currsoln[var] = idx_old[0][bin_idx]
        assignments[sol, :] = currsoln
        temp_cost = P0[:, currsoln]
        temp_cost = torch.gather(temp_cost, dim=1, index=idx_to_gather)
        costs[0, sol] = temp_cost.sum()

    C, inverse_I = torch.unique(assignments, sorted=True, return_inverse=True, dim=0)
    perm = torch.arange(inverse_I.size(0), device=device)
    inverse_I, perm = inverse_I.flip([0]), perm.flip([0])
    I = inverse_I.new_empty(C.size(0)).scatter_(0, inverse_I, perm)

    assignments = C
    costs = costs[:, I]
    return assignments, costs


def get_birth_gm_from_Z_disk_torch(Z, mu_pred, current_label=1, k=0, distance_thres=100, initial_P=20, device=None):
    '''
        Generate Birth Objects from previous measurements that were not labeled
    '''
    if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # mu_pred = np.array(mu_pred)

    absolute_indexes = []

    if(len(mu_pred.shape) > 0 and Z.shape[0] > 0):
        # Do measurement - component association
        cost = torch.cdist(Z[:, 0:2], mu_pred[:, 0:2, 0], p=2)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        distances = cost[row_ind, col_ind]
        reduced_indexes = (distances < distance_thres).nonzero()[:, 0]
        absolute_indexes = row_ind[reduced_indexes.cpu().numpy()]

    label = current_label
    H_birth = 0
    new_mu = []

    L_birth = []

    for i in range(len(Z)):
        if i in absolute_indexes:
            continue

        z = Z[i, :]
        new_z = [z[0].cpu().item(), z[1].cpu().item(), 0, 0, 5, 5]
        new_mu.append(new_z)
        L_birth.append([k, label])
        label += 1
        H_birth += 1

    current_label = label
    w_birth = torch.tensor(np.ones(H_birth) / float(H_birth)).unsqueeze(-1).to(device)
    mu_birth = torch.tensor(new_mu).unsqueeze(-1).float().to(device)
    if len(new_mu) == 0:
        mu_birth = mu_birth.unsqueeze(-1)

    Po = initial_P
    P = np.array([[Po, 0, 0, 0, 0, 0],
                  [0, Po, 0, 0, 0, 0],
                  [0, 0, Po, 0, 0, 0],
                  [0, 0, 0, Po, 0, 0],
                  [0, 0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 0, 3]
                  ])
    P_birth = np.repeat(P[np.newaxis, :, :], H_birth, axis=0).astype(np.float)
    P_birth = torch.tensor(P_birth).float().to(device)

    tt_birth = Track_List()
    tt_birth.ws = w_birth
    tt_birth.mus = mu_birth
    tt_birth.Ps = P_birth
    tt_birth.Ls = L_birth
    tt_birth.Es = [None] * H_birth 
    tt_birth.Num_Ts = H_birth

    if Z.shape[0] > 0:
        surviving_measurements = Z[absolute_indexes, :]
    else:
        surviving_measurements = torch.tensor([], device=mu_birth.device)

    return tt_birth, surviving_measurements


def concatenate_tracks(tt_left, tt_right):
    '''
        Append tt_left to tt_right
    '''

    new_tracks = Track_List()
    if len(tt_right.mus.shape) == 0:
        new_tracks = tt_left
        return new_tracks
    if len(tt_left.mus.shape) == 0 or tt_left.mus.shape[0] == 0:
        new_tracks = tt_right
        return new_tracks

    new_tracks.mus = torch.cat((tt_left.mus, tt_right.mus), 0)                                 # tensor concat
    new_tracks.Ps = torch.cat((tt_left.Ps, tt_right.Ps), 0)                                    # tensor concat
    new_tracks.ws = torch.cat((tt_left.ws, tt_right.ws), 0)                                    # tensor concat
    new_tracks.Ls = tt_left.Ls + tt_right.Ls                                                   # list concatenation
    new_tracks.Es = tt_left.Es + tt_right.Es                                                   # list concatenation
    new_tracks.Num_Ts = tt_left.Num_Ts + tt_right.Num_Ts                                       # list concatenation
    return new_tracks


def clean_update(glmb_temp):
    # flag used tracks
    usedindicator = torch.zeros(glmb_temp.tt.Num_Ts)
    for hidx in range(len(glmb_temp.w)):
        usedindicator[glmb_temp.I[hidx].long()] = usedindicator[glmb_temp.I[hidx].long()] + 1
    track_indices = usedindicator > 0
    trackcount = track_indices.sum()

    # remove unused tracks and reindex existing hypotheses/components
    newindices = torch.zeros(glmb_temp.tt.Num_Ts)
    newindices[track_indices] = torch.arange(trackcount, dtype=newindices.dtype)

    cleaned_track_list = glmb_temp.tt.index_w_tensor(track_indices.nonzero()[:, 0])
    glmb_clean = glmb_instance(track_list=cleaned_track_list)
    glmb_clean.w = glmb_temp.w
    glmb_clean.n = glmb_temp.n
    glmb_clean.cdn = glmb_temp.cdn

    for hidx in range(len(glmb_temp.w)):
        if(hidx == 0):
            glmb_clean.I = [newindices[glmb_temp.I[hidx].long()]]
        else:
            glmb_clean.I.append(newindices[glmb_temp.I[hidx].long()])

    return glmb_clean


def clean_predict(glmb_raw):
    # hash label sets, find unique ones, merge all duplicates
    return glmb_raw
    '''
    for hidx in range(glmb_raw.w.shape[0]):
        #glmb_raw.hash{hidx}= sprintf('%i*',sort(glmb_raw.I{hidx}(:)_T))


    [cu,~,ic]= unique(glmb_raw.hash);

    glmb_temp.tt= glmb_raw.tt;
    glmb_temp.w= zeros(length(cu),1);
    glmb_temp.I= cell(length(cu),1);
    glmb_temp.n= zeros(length(cu),1);
    for hidx= 1:length(ic)
            glmb_temp.w(ic(hidx))= glmb_temp.w(ic(hidx))+glmb_raw.w(hidx);
            glmb_temp.I{ic(hidx)}= glmb_raw.I{hidx};
            glmb_temp.n(ic(hidx))= glmb_raw.n(hidx);
    glmb_temp.cdn= glmb_raw.cdn;
    '''


def prune(glmb_updated):
    # prune components with weights lower than specified threshold
    idxkeep = (glmb_updated.w > 0.01).nonzero()[:, 0]
    glmb_pruned = glmb_instance(track_list=glmb_updated.tt)
    glmb_pruned.w = glmb_updated.w[idxkeep]
    if(glmb_updated.I is not None):
        glmb_pruned.I = [glmb_updated.I[i] for i in idxkeep]
    else:
        glmb_pruned.I = None
    glmb_pruned.n = glmb_updated.n[idxkeep]

    glmb_pruned.w = glmb_pruned.w / glmb_pruned.w.sum()

    for card in range(glmb_pruned.n.max() + 1):
        card_bin = (glmb_pruned.w[glmb_pruned.n == card]).sum()
        glmb_pruned.cdn.append(card_bin)

    glmb_pruned.cdn = torch.tensor(glmb_pruned.cdn)

    return glmb_pruned



def extract_all_possible_tracks(glmb_update):
    '''
        extract estimates via best cardinality, then
        best component/hypothesis given best cardinality, then
        best means of tracks given best component/hypothesis and cardinality
    '''
    Dk = {}
    Dk['mus'] = []
    Dk['ws'] = []
    Dk['Ps'] = []
    Dk['Ls'] = []
    Dk['Hs'] = 0

    N = torch.argmax(glmb_update.cdn)
    if(glmb_update.tt.Num_Ts == 0):
        return Dk

    for track_index in range(glmb_update.tt.Num_Ts):
        Dk['mus'].append(glmb_update.tt.mus[track_index, :, 0].numpy())
        Dk['Ls'].append(torch.tensor(glmb_update.tt.Ls[track_index])[1].item())
        Dk['ws'].append(1)
        Dk['Ps'].append(torch.tensor(glmb_update.tt.Ps[track_index]))

    Dk['Hs'] = N
    return Dk


def get_birth_gm_w_birth_field(Z, birth_field, mu_pred, current_label=1, k=0, distance_thres=100, initial_P=20, device=None):
    '''
        Generate Birth Objects from previous measurements that were not labeled
    '''
    if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # mu_pred = np.array(mu_pred)

    absolute_indexes = []

    if(len(mu_pred.shape) > 0 and Z.shape[0] > 0):
        # Do measurement - component association
        cost = torch.cdist(Z[:, 0:2], mu_pred[:, 0:2, 0], p=2)
        row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
        distances = cost[row_ind, col_ind]
        reduced_indexes = (distances < distance_thres).nonzero()[:, 0]
        absolute_indexes = row_ind[reduced_indexes.cpu().numpy()]

    label = current_label
    H_birth = 0
    new_mu = []

    L_birth = []

    for i in range(len(Z)):
        if i in absolute_indexes:
            continue

        z = Z[i, :]
        px, py = z[0].cpu().item(), z[1].cpu().item()
        vx, vy = birth_field[int(px), int(py), 0:2]
        new_z = [z[0].cpu().item(), z[1].cpu().item(), vx, vy, 5, 5]
        new_mu.append(new_z)
        L_birth.append([k, label])
        label += 1
        H_birth += 1

    current_label = label
    w_birth = torch.tensor(np.ones(H_birth) / float(H_birth)).unsqueeze(-1).to(device)
    mu_birth = torch.tensor(new_mu).unsqueeze(-1).float().to(device)
    if len(new_mu) == 0:
        mu_birth = mu_birth.unsqueeze(-1)

    Po = initial_P
    P = np.array([[Po, 0, 0, 0, 0, 0],
                  [0, Po, 0, 0, 0, 0],
                  [0, 0, Po, 0, 0, 0],
                  [0, 0, 0, Po, 0, 0],
                  [0, 0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 0, 3]
                  ])
    P_birth = np.repeat(P[np.newaxis, :, :], H_birth, axis=0).astype(np.float)
    P_birth = torch.tensor(P_birth).float().to(device)

    tt_birth = Track_List()
    tt_birth.ws = w_birth
    tt_birth.mus = mu_birth
    tt_birth.Ps = P_birth
    tt_birth.Ls = L_birth
    tt_birth.Es = [None] * H_birth
    tt_birth.Num_Ts = H_birth

    if Z.shape[0] > 0:
        surviving_measurements = Z[absolute_indexes, :]
    else:
        surviving_measurements = torch.tensor([], device=mu_birth.device)

    return tt_birth, surviving_measurements

