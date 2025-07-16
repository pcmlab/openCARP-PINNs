import os, sys, subprocess, re, struct, errno
import scipy.io
import deepxde as dde
from deepxde.backend import tf
import numpy as np
from typing import Any
import torch

class system_dynamics():
    
    def __init__(self, **config: Any):
        
        ## PDE Parameters
        if config['data']['ionic_model_name'] == 'AP':
            self.a = 0.15
            self.b = 0.15
            self.D = 0.5 #0.05
            self.k = 8
            self.mu_1 = 0.2 #Can be obtained form the OpenCARP parameter file
            self.mu_2 = 0.3 #Can be obtained form the OpenCARP parameter file
            self.epsilon = 0.002
            self.touAcm2 = 100/12.9
            self.t_norm = 12.9
            self.ionic_model = 'AP'

        if config['data']['ionic_model_name'] == 'MS':
            self.V_gate = 0.13
            self.a_crit = 0.13  # comment in openCARP: 0.13
            self.tau_in = 0.05 # 0.3 in openCARP #0.05 in Belhamadia et al
            self.tau_out = 1 # 5.0 in opencarp #1 in Belhamadia et al
            self.tau_open = 95 # 120.0 in openCARP #95 in Belhamadia et al
            self.tau_close = 162 # 150 in openCARP # 162 in Belhamadia et al
            self.D = 0.01
            self.ionic_model = 'MS'

        ## Geometry Parameters
        self.min_x = 0
        self.max_x = 10            
        self.min_y = 0 
        self.max_y = 10
        self.min_t = 0
        self.max_t = 99
        self.spacing = 0.1

    def read_array_igb(self, igbfile):
        """
        Purpose: Function to read a .igb file
        """
        data = []
        file = open(igbfile, mode="rb")
        header = file.read(1024)
        words = header.split()
        word = []
        for i in range(4):
            word.append(int([re.split(r"(\d+)", s.decode("utf-8")) for s in [words[i]]][0][1]))

        nnode = word[0] * word[1] * word[2]

        for _ in range(os.path.getsize(igbfile) // 4 // nnode):
            data.append(struct.unpack("f" * nnode, file.read(4 * nnode)))

        file.close()
        return data

    def read_pts(self, modname, n=3, vtx=False, item_type=float):
        """Read pts/vertex file"""
        with open(modname + (".vtx" if vtx else ".pts")) as file:
            count = int(file.readline().split()[0])
            if vtx:
                file.readline()

            pts = np.empty((count, n), item_type)
            for i in range(count):
                pts[i] = [item_type(val) for val in file.readline().split()[0:n]]

        return pts if n > 1 else pts.flat

    def generate_data(self, v_file_name, w_file_name, pt_file_name, scenario_name, ionic_model_name): #Temporary becasue we dont have W yet!!Plz add w_file_name later
        
        data_V = np.array(self.read_array_igb(v_file_name)) #new parser for vm.igb for voltage
        data_W = np.array(self.read_array_igb(w_file_name)) #new parser for vm.igb for W #Temporary becasue we dont have W yet!!
        coordinates = np.array(self.read_pts(pt_file_name)) #new parser for .pt file


        #if ionic_model_name == 'MS':
            #t = np.arange(0, data_V.shape[0]/100, 0.01).reshape(-1, 1)
        if ionic_model_name == 'AP':
            t = np.arange(0, data_V.shape[0]).reshape(-1, 1)
        
        coordinates = (coordinates - np.min(coordinates))/1000
        coordinates = coordinates[:, 0:2]
        
        x = np.unique(coordinates[:, 0]).reshape((1, -1))
        y = np.unique(coordinates[:, 1]).reshape((1, -1))
        len_x = x.shape[1]
        len_y = y.shape[1]
        len_t = t.shape[0]

        no_of_nodes = coordinates.shape[0]
        repeated_array = np.repeat(coordinates, len_t, axis=0)
        xy_concatenate = np.vstack(repeated_array)
        t_concatenate = np.concatenate([t] * no_of_nodes, axis=0)
        grid = np.concatenate([xy_concatenate, t_concatenate], axis=1)
        
        if ionic_model_name == 'AP':
            data_V = (data_V + 80)/100
            #data_W = (data_W + 80)/100
        data_V = data_V.T
        data_W = data_W.T

        shape = [len_x, len_y, len_t]
        V = data_V.reshape(-1, 1)
        W = data_W.reshape(-1, 1)

        shape = [len_x, len_y, len_t]
        Vsav = V.reshape(len_x, len_y, len_t)

        Wsav = W.reshape(len_x, len_y, len_t)

        ##Computing in Cardiology Extrapolation from source
        if scenario_name == 'Single_Corner':
            midpt_x = np.max(grid[:,0])*0.5
            midpt_y = np.max(grid[:,1])*0.5
            idx_data_smaller = np.where((grid[:,0]<=midpt_x) & (grid[:,1]<=midpt_y))
            idx_data_larger = np.where((grid[:,0]>midpt_x) | (grid[:,1]>midpt_y))

        if scenario_name == 'Planar' or scenario_name == 'Double_Corner':
            first_quarter_x = np.max(grid[:,0])*0.25
            idx_data_smaller = np.where((grid[:,0]<=first_quarter_x))
            idx_data_larger = np.where((grid[:,0]>first_quarter_x))


        ##Computing in Cardiology Inverse Extrapolation
        #first_quat_x = np.max(grid[:,0])*0.25
        #first_quat_y = np.max(grid[:,1])*0.25
        #third_quat_x = np.max(grid[:,0])*0.75
        #third_quat_y = np.max(grid[:,1])*0.75

        #idx_data_smaller = np.where((grid[:,0]>=first_quat_x) & (grid[:,0]<=third_quat_x) & (grid[:,1]>=first_quat_y) & (grid[:,1]<=third_quat_y))
        #idx_data_larger = np.where((grid[:,0]<first_quat_x) | (grid[:,0]>third_quat_x) | (grid[:,1]<first_quat_y) | (grid[:,1]>third_quat_y))


        #The lower quadrant   
        smaller_grid = grid[idx_data_smaller]
        smaller_V = V[idx_data_smaller]
        smaller_W = W[idx_data_smaller]

        #The other 3 quadrant   
        larger_grid = grid[idx_data_larger]
        larger_V = V[idx_data_larger]
        larger_W = W[idx_data_larger]

        #Shuffling the data
        def shiffling(grid, V, W):
            num_rows = grid.shape[0]
            indices = np.arange(num_rows)
            np.random.shuffle(indices)
            
            grid = grid[indices]
            V = V[indices]
            W = W[indices]
            
            return grid, V, W

        observe_train, v_train, w_train = shiffling(smaller_grid, smaller_V, smaller_W)
        observe_test, v_test, w_test = shiffling(larger_grid, larger_V, larger_W)

        return observe_train, observe_test, v_train, v_test, w_train, w_test, Vsav, V, len_t, idx_data_larger, Wsav, W

    def geometry_time(self):  
        geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
        timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        return geomtime

    def params_to_inverse(self,**config: Any):
        
        params = []
        #if not config['data']['inverse']:
        if self.ionic_model == 'AP':
            return self.a, self.b, self.D, params
        if self.ionic_model == 'MS':
            return self.V_gate, self.a_crit, params
    
    def pde_2D(self, x, y):

        if self.ionic_model == 'AP':
            V, W = y[:, 0:1], y[:, 1:2]
            dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
            dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
            dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
            ## Coupled PDE+ODE Equations
            eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + (self.k*V*(V-self.a)*(V-1) +W*V)
            eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))

        if self.ionic_model == 'MS':

            V, W = y[:, 0:1], y[:, 1:2]

            dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
            dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)

            # Define the conditions based on V and V_gate
            condition = V[:, 0] < self.V_gate

            # Extract V and h based on the condition
            V_p_ext, W_p_ext = V[condition], W[condition]
            V_q_ext, W_q_ext = V[~condition], W[~condition]
            Vp = V_p_ext.reshape(-1, 1)
            Wp = W_p_ext.reshape(-1, 1)
            Vq = V_q_ext.reshape(-1, 1)
            Wq = W_q_ext.reshape(-1, 1)

            # compute derivatives
            #Vp, Wp = y[:, 0:1], y[:, 1:2]
            #Vq, Wq = y[:, 0:1], y[:, 1:2]
            dVp_dt = dde.grad.jacobian(y, x, i=0, j=2)
            dWp_dt = dde.grad.jacobian(y, x, i=1, j=2)
            dVq_dt = dde.grad.jacobian(y, x, i=0, j=2)
            dWq_dt = dde.grad.jacobian(y, x, i=1, j=2)

            # if V < V_gate
            eq_a_p = dVp_dt - (Wp * Vp * (Vp  - self.a_crit) * (1 - Vp) / self.tau_in + (-Vp / self.tau_out)) - self.D*(dv_dxx + dv_dyy)
            eq_b_p = dWp_dt - (1. - Wp) / self.tau_open

            # if V > V_gate
            eq_a_q = dVq_dt - (Wq * Vq * (Vq  - self.a_crit) * (1 - Vq) / self.tau_in + (-Vq / self.tau_out)) - self.D*(dv_dxx + dv_dyy)
            eq_b_q = dWq_dt + Wq / self.tau_close

            # Combine the equations based on the conditions
            eq_a = torch.cat((eq_a_p, eq_a_q))
            eq_b = torch.cat((eq_b_p, eq_b_q))

        return [eq_a, eq_b]  
 
    def IC_func(self,observe_train, v_train):
        
        T_ic = observe_train[:,2].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,0))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)
    
    def BC_func(self, geomtime):
        bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), self.boundary_func_2d, component=0)
        return bc
    
    def boundary_func_2d(self,x, on_boundary):
            return on_boundary and ~(x[0:2]==[self.min_x,self.min_y]).all() and  ~(x[0:2]==[self.min_x,self.max_y]).all() and ~(x[0:2]==[self.max_x,self.min_y]).all()  and  ~(x[0:2]==[self.max_x,self.max_y]).all() 
   

