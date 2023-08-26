import sys
import scipy.signal
# import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import random

class TileWorld:
    """
    Crudely approximating diffusion with kernel applied to a grid of cells:
    - Calculating curvature by fitting parabola part of expansion to a cell and its neighbors, gets:
        d^2 T / d x^2 (x_n) ~= T(x_{n-1}) + T(x-{n+1}) - 2*T(x_n)
    - Combining x and y together gives
        nabla^2 T ~= T_(x_{n-1},y_n) + T_(x_{n+1},y_n) + T_(x_n,y_{n-1}) + T_(x_n,y_{n+1}) - 4 * T(x_n, y_n)
        which can be expressed as a kernel:
        [[0, 1, 0]
        [1, -4, 0]
        [0, 1, 0]]
    So for d T / d t = (const) * nabla^2 T
        dT = timestep * const * nabla^2 T
    
    """
    def __init__(self, width=100,height=100):
        self.width = width
        self.height = height
        
        #Concentrations P and S
        # Initialize P from random pixels
        # self.P_concentrations  = np.random.uniform(0,0.5,width*height).reshape((width, height))
        self.P_concentrations = np.zeros((width, height), dtype=float)
        
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        self.P_concentrations = 0.5*(1 + np.sin(xx * ( 2 * np.pi)/8))
        # for i in range(50):
        #     dot_x = random.randint(0,width)
        #     dot_y = random.randint(0,height)
        # # for i in range(10):
        # #     # dot_x = random.randint(0,width)
        # #     # for dot_x in [10,20,34,40,50,60]:
        # #     for j in range(10):
        # #         dot_y = random.randint(0,height)
        #     # for dot_y in [10,21,30,40,50,60]:
            
        #     dot_mask = ( (xx - dot_x)**2 + (yy - dot_y)**2 ) < dot_rad**2
        #     self.P_concentrations[dot_mask] = 0.9
        
        
        self.S_concentrations = np.zeros( (width, height), dtype=float)
        
        
        # Contribution per timestep to neighbor concentrations.
        # Scaling these linearly is equivalent to scaling the timestep, so we add speed_factor
        speed_factor = 0.005
        self.P_diffuse_constant = 0.05 * speed_factor
        self.S_diffuse_constant = 0.025 * speed_factor
        
        
        self.P_self_promo_factor = 0.9 * speed_factor
        self.P_inhibition_by_S_factor = 0.1#0.8 * speed_factor
        
        self.S_promotion_by_P_factor = 0.05# 0.8 * speed_factor       
        self.S_self_inhibition_factor = 0.8 * speed_factor
        
        # self.max 
    
    def do_timestep(self):
        # Calculate contribution from diffusion - using a rough convolution that would conserve mass,
        # but is not necessarily accurate.
        
        P_k_adj = self.P_diffuse_constant / 4
        P_kernel = np.array([
            [0,          P_k_adj,                  0],
            [P_k_adj, - self.P_diffuse_constant, P_k_adj],
            [0,          P_k_adj,                  0],
        ])
        
        S_k_adj = self.S_diffuse_constant / 4
        S_kernel = np.array([
            [0,          S_k_adj,                  0],
            [S_k_adj, - self.P_diffuse_constant, S_k_adj],
            [0,          S_k_adj,                  0],
        ])
        
        P_diffuse_contribution = scipy.signal.convolve2d(self.P_concentrations, P_kernel, mode='same')
        S_diffuse_contribution = scipy.signal.convolve2d(self.S_concentrations, S_kernel, mode='same')
        
        print("Total P diffuse contribution: ", np.sum(P_diffuse_contribution))
        print("Total S diffuse contribution; ", np.sum(S_diffuse_contribution))
        
        P_self_promotion_contribution = self.P_self_promo_factor * self.P_concentrations
        P_inhibition_contribution = - self.P_inhibition_by_S_factor *  self.S_concentrations
        print("Total P self promotion", np.sum(P_self_promotion_contribution))
        print('Total P inhibition from S', np.sum(P_inhibition_contribution))
        # print("")
        
        S_promotion_from_P = self.S_promotion_by_P_factor * self.P_concentrations
        # print("")
        S_self_inhibition = - self.S_self_inhibition_factor * self.S_concentrations
        # S has no self promotion factor
        
        self.P_concentrations += (P_diffuse_contribution + P_self_promotion_contribution + P_inhibition_contribution)
        self.S_concentrations += (S_diffuse_contribution + S_promotion_from_P + S_self_inhibition)
        
        # Concentrations must be between 0 and 1
        self.P_concentrations[self.P_concentrations < 0.0] = 0.0
        self.S_concentrations[self.S_concentrations < 0.0] = 0.0
        
        self.P_concentrations[self.P_concentrations > 1.0] = 1.0
        self.S_concentrations[self.S_concentrations > 1.0] = 1.0

def start_simulation():
    w = TileWorld()
    
    # fig, ax = plt.subplots()
    # plt.show
    fig, (ax0, ax1) = plt.subplots(1,2)
    P_image = ax0.imshow(w.P_concentrations, vmin=0, vmax=1)
    S_image = ax1.imshow(w.S_concentrations, vmin=0, vmax=1)
    # ax9.imshow(w.)
    while True:
        P_image.set_data(w.P_concentrations)
        S_image.set_data(w.S_concentrations)
        # plt.clf()
        fig.canvas.flush_events()
        # print("Pausing...")
        plt.pause(0.05)
        # time.sleep(1)
        # key = cv2.waitKey(1)#pauses for 3 seconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
        #     cv2.destroyAllWindows()
        #     break
        for x in range(5):
            w.do_timestep()
        # cv2.waitKey()
    


if __name__ == "__main__":
    start_simulation()

