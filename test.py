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
    def __init__(self, width=50,height=50):
        self.width = width
        self.height = height
        
        self.cell_size = 0.5
        self.timestep = 0.05
        
        self.F = 0.035
        self.k = 0.0625
        self.D_u = 0.01
        self.D_v = 0.005
        
        self.U_concentrations = np.ones((width,height),dtype=float)
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # dot_mask = (xx-width/2)**2 +(yy-height/2)**2 < 8**2
        dot_mask = np.logical_or(
            (xx>20)*(xx < 30)*(yy>20)*(yy<30),
            (xx>5)*(xx < 15)*(yy>5)*(yy<15)
        )
        self.U_concentrations[dot_mask]=0.5
        
        self.V_concentrations = np.zeros((width, height))
        self.V_concentrations[dot_mask]=0.2
        
        self.make_diffusion_kernel()
        
        
    
    def make_diffusion_kernel(self):
        
        raw_kernel = np.array([
            [0,1,0],
            [1,-4,1],
            [0,1,0],
        ],dtype=float)
        
        
        self.diff_kernel = raw_kernel
        
        
    
    def do_timestep(self):
        # Calculate contribution from diffusion - using a rough convolution that would conserve mass,
        # but is not necessarily accurate.
        
        Del_sq_U = scipy.signal.convolve2d(self.U_concentrations, self.diff_kernel, mode='same')
        # U_diffuse_contribution = self.D_u * Del_sq_U
        
        Del_sq_V = scipy.signal.convolve2d(self.V_concentrations, self.diff_kernel, mode='same')
        # V_diffuse_contribution = self.D_v * Del_sq_V
        U_diffusion_rate = self.D_u * Del_sq_U
        V_diffusion_rate = self.D_v * Del_sq_V
        self.U_concentrations += U_diffusion_rate
        self.V_concentrations += V_diffusion_rate
        
        U_eaten_rate = (self.U_concentrations * (self.V_concentrations ** 2))
        U_growing_rate = self.F * (1 - self.U_concentrations)
        U_rateofchange = (
            - U_eaten_rate +
            U_growing_rate
        )
        # print("total u eaten: ", np.sum(U_eaten_rate))
        # print("total u growth promotion factor: ", np.sum(U_growing_rate))
        # print("total u diffusion rate", np.sum(U_diffusion_rate))
        
        V_rateofchange = (
            self.U_concentrations * (self.V_concentrations ** 2) +
            (- (self.F + self.k) * self.V_concentrations)
        )
        
        self.U_concentrations += U_rateofchange * self.timestep
        self.V_concentrations += V_rateofchange * self.timestep
        
        # Concentrations are limited to be within 0 and 1
        self.U_concentrations[self.U_concentrations < 0] = 0.0
        self.V_concentrations[self.V_concentrations < 0] = 0.0
        
        self.U_concentrations[self.U_concentrations > 1] = 1.0
        self.V_concentrations[self.V_concentrations > 1] = 1.0

def start_simulation():
    w = TileWorld()
    
    # plt.show
    fig, (ax0, ax1) = plt.subplots(1,2)
    P_image = ax0.imshow(w.U_concentrations, cmap = plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.colorbar(P_image)
    S_image = ax1.imshow(w.V_concentrations, cmap = plt.get_cmap('gray'),vmin=0, vmax=1)
    plt.colorbar(S_image)
    ax0.set_title("U")
    ax1.set_title("V")
    
    idx=0
    while True:
        P_image.set_data(w.U_concentrations)
        S_image.set_data(w.V_concentrations)
        # plt.clf()
        fig.canvas.flush_events()
        # print("Pausing...")
        plt.pause(0.001)
        # time.sleep(1)
        # key = cv2.waitKey(1)#pauses for 3 seconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
        #     cv2.destroyAllWindows()
        #     break
        for x in range(200):
            idx+=1
            w.do_timestep()
        # cv2.waitKey()
    


if __name__ == "__main__":
    start_simulation()

