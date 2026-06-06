import numpy as np
#import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from scipy.linalg import expm
from scipy.linalg import solve_discrete_are


# ====================================================================
#
#                            RocketModel
#
# ====================================================================

class RocketModel:
    """
    Planar Vertical Take-Off and Landing (PVTOL) rocket model
    State:   x = [px, py, vx, vy, theta, omega]
    Control: u = [Fe, Fs, phi]
    """
    def __init__(self, Tmax=50, dt=0.02, rx = 0.0, rv = 0.0):

        # ph    
        self.m  = 530.4058532714844   # kg
        self.ls = 0.24                # m
        self.l1 = 2.8466666666666667  # m
        self.l2 = 2.1350000000000002  # m
        self.h  = 0.82                # m
        self.I  = 1209.53515625       # N*m   // I = (1/12) * m * (l1+l2+h)^2
        self.g  = 9.81 
        self.dt = dt

        self.rocket_shape  = np.array([
                    [-self.ls,  self.l2],        # upper-left
                    [-self.ls, -self.l1],        # lower-left
                    [ self.ls, -self.l1],        # lower-right
                    [ self.ls,  self.l2],        # upper-right
                    [-self.ls,  self.l2],        # upper-left
                    [    0,     self.l2+self.h], # Apex of the cone
                    [ self.ls,  self.l2]         # upper-right
                ])
        

        # modeling space
        self.gr_min_x = -20
        self.gr_max_x = +40
        self.gr_min_y = -5
        self.gr_max_y = +40
        
        self.platf_w =  8
        self.platf_h  = 2
        self.platf_rh = 3.4 #

        self.ry = self.platf_h + self.platf_rh # ref coord y, m
        self.rx = rx                           # ref coord x, m
        self.rv = rv                           # ref speed (drx/dt), m/sec
        self.rytol = 0.4 # m 
        
        # Input constraints (Control):
        self.max_Fe  = 16118.518518518518  # H, 0<Fe<max_Fe
        self.abs_Fs  = 322.3703703703704   # H, abs(Fs)<abs_Fs
        self.abs_phi = 0.2617993877991494  # rad, abs(phi)<abs_phi
        # Norm. Matrix
        self.B_norm = np.diag ([self.max_Fe, self.abs_Fs, self.abs_phi])
        self.u_eq_norm = np.array([self.m * self.g / self.max_Fe, 0.00, 0.00])  # equilibrium normalized control

        self.Kanim_Fe  = 10  # H, 0<Fe<max_Fe
        self.Kanim_Fs  = 2   # H, abs(Fs)<abs_Fs
        self.Kanim_phi = self.abs_phi*2  # rad, abs(phi)<abs_phi
 
        # State constraints (terminate conditions)
        self.Nmax = int(Tmax // dt) # max steps limit
        self.max_abs_theta = 0.42
        self.min_x = self.gr_min_x
        self.max_x = self.gr_max_x 
        self.min_y = self.ry - self.rytol
        self.max_y = self.gr_max_y  
        
        # Landing criteria
        self.ref = np.array([self.rx, self.ry, self.rv, 0.0, 0.0, 0.0]) # Reference
        self.tol = np.array([2, 0.28, 0.1, 0.1, 0.04, 0.02]) # Landing tolerances +/-
  
        # State trajectories and Control stored as column vectors.
        self.X  = np.zeros((6, self.Nmax + 1))
        self.U  = np.zeros((3, self.Nmax))
        self.Un = np.zeros((3, self.Nmax))        
        self._step_idx = 0          # steps executed
        self.state = np.zeros(6)
        self.success = False          # 
        self.done = False             # 
        self.terminate_error = None   # string description if episode ended abnormally, else None

    def reset(self, x0=None):
        """Resets the model to the initial state. Prepares a new episode"""
        if x0 is None:
            self.state = np.zeros(6)
        else:
            x0 = np.array(x0, dtype=float).flatten()
            if x0.shape[0] != 6:
                raise ValueError("The state vector x0 must contain exactly 6 elements.")
            self.state = x0.copy()

        self._step_idx = 0
        self.success = False
        self.done = False
        self.terminate_error = None
        #self.X.fill(0.0)
        #self.U.fill(0.0)
        self.X  = np.zeros((6, self.Nmax + 1))
        self.U  = np.zeros((3, self.Nmax))
        self.Un = np.zeros((3, self.Nmax))         
        self.X[:, 0] = self.state.copy()

    def _dynamics(self, state, u):
        x, y, dx, dy, theta, dtheta = state
        F_E, F_S, phi = u
        m, I, l1, l2, g = self.m, self.I, self.l1, self.l2, self.g

        ddx = (-F_E * np.sin(phi + theta)  + F_S * np.cos(theta)) / m
        ddy = ( F_E * np.cos(phi + theta)  + F_S * np.sin(theta)) / m - g
        ddtheta = (-F_E * l1 * np.sin(phi) - F_S * l2) / I
        return np.array([dx, dy, ddx, ddy, dtheta, ddtheta])

    def step(self, action):
        """
        Performs one integration step (Ts) and returns the new state.
        """
        if self.done:
            raise RuntimeError("The episode is finished. You must call reset() to start a new one..")

        u_unlim = np.array(action, dtype=float).flatten()
        if u_unlim.shape[0] != 3:
            raise ValueError("action  must contain exactly 3 elements: Fe, Fs, phi")
    
        # Control signal saturation
        F_E = np.clip(u_unlim[0], 0.0, 1)    # 0 <= Fe <= max_Fe
        F_S = np.clip(u_unlim[1], -1.0, 1.0) # |Fs| <= abs_Fs
        phi = np.clip(u_unlim[2], -1.0, 1.0) # |phi| <= abs_phi
        u_norm = np.array([F_E, F_S, phi])
        u = self.B_norm @ u_norm
        
        # RK4
        dt = self.dt
        s = self.state
        k1 = self._dynamics(s, u)
        k2 = self._dynamics(s + 0.5 * dt * k1, u)
        k3 = self._dynamics(s + 0.5 * dt * k2, u)
        k4 = self._dynamics(s + dt * k3, u)
        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Saves the control input and the new state (as column vectors).
        self.Un[:, self._step_idx] = u_norm
        self.U[:, self._step_idx]  = u
        self.X[:, self._step_idx + 1] = self.state
        self._step_idx += 1

        # Checking termination conditions
        self._check_termination()
        return self.state.copy()

    def _check_termination(self):
        """ 
        Checks termination conditions:
          1) successful landing (within tolerances),
          2) maximum number of steps exceeded,
          3) state constraints violated.
        Sets self.done and records the reason in self.terminate_error.
        """
        x = self.state
        r = self.ref    
        
        # 1) landing has been achieved
        if np.all(np.abs(x-r) < self.tol):
            self.success = True
            self.done = True
            self.terminate_error = "no error"
            return

        # 2) the time limit is exceeded
        if self._step_idx >= self.Nmax:
            self.done = True
            self.terminate_error = "maximum number of steps exceeded"
            return

        # 3) constraints have been violated
        if abs(x[4]) > self.max_abs_theta:               # 
            self.done = True
            self.terminate_error = "pitch angle limit exceeded"
            return
        if (x[0] > self.max_x) or (x[0] < self.min_x):   # 
            self.done = True
            self.terminate_error = "horizontal position out of bounds"
            return
        if (x[1] > self.max_y) or (x[1] < self.min_y):   # 
            self.done = True
            self.terminate_error = "vertical position out of bounds"
            return
            

    @property
    def N(self):
        return self._step_idx  # current step (discrete time)

    def episode(self, physic_val=False):
        """
        Returns the time series of the current episode:
        X : (6, N+1) – states (columns over time),
        U : (3, N)   – controls (columns over time).
        If the episode is not completed, returns the current data.
        """
        N = self._step_idx
        # Trimming to the actual number of steps
        X_out  = self.X [:, :N + 1].copy()
        if physic_val:
            U_out  = self.U[:, :N].copy()
        else:
            U_out = self.Un[:, :N].copy()
        
        return X_out, U_out, N, self.done, self.success


    def fuel_consumed(self, ve=2000.0):  # !!! not used
        """
        Estimated fuel mass consumption (kg).
        ve – exhaust velocity, m/s.
        """
        F_E = self.U[0, :self._step_idx]   #
        dt = self.dt
        return np.sum(np.abs(F_E)) * dt / ve

    def get_linear_css(self):
        """
        Computes the simplified, analytical linearization matrices A (6x6) and B (6x3)
        strictly for the vertical hover operating point:
        x_eq = [0, 0, 0, 0, 0, 0]^T
        u_eq = [m*g, 0, 0]^T
        """

        # Physical parameters of the system
        m, I, l1, l2, g = self.m, self.I, self.l1, self.l2, self.g

        # --- System Dynamics Matrix A (6x6) ---
        A = np.zeros((6, 6))    

        # Kinematic relationships
        A[0, 2] = 1.0  # px' = vx
        A[1, 3] = 1.0  # py' = vy
        A[4, 5] = 1.0  # theta' = omega

        # The only non-zero dynamic coupling: tilt causes horizontal acceleration
        A[2, 4] = -g

        # --- Control Input Matrix B (6x3) ---
        B = np.zeros((6, 3))

        # Row 2: Linear acceleration along x-axis (f3)
        B[2, 1] = 1.0 / m       # df3 / dF_s
        B[2, 2] = -g            # df3 / dphi

        # Row 3: Linear acceleration along y-axis (f4)
        B[3, 0] = 1.0 / m       # df4 / dF_e

        # Row 5: Angular acceleration (f6)
        B[5, 1] = -l2 / I       # df6 / dF_s
        B[5, 2] = -(m * g * l1) / I  # df6 / dphi

        B_norm = np.diag ([self.max_Fe, self.abs_Fs, self.abs_phi])
        Bn = B@B_norm # normalized Control Matrix
        return A, Bn

    def get_linear_dss(self, Ts):
        """
        Returns discrete matrices A_d (6x6) and B_d (6x3)
        for the linearized model with sampling period Ts (ZOH-method).
        """
        A, B = self.get_linear_css()
        n = A.shape[0]   # Number of states
        m = B.shape[1]   # Number of control inputs
    
        # Augmented matrix [[A, B], [0, 0]] with dimensions (n+m) x (n+m)
        Z = np.vstack([
            np.hstack([A, B]),
            np.zeros((m, n + m))
        ])
    
        # Matrix exponential with time step Ts
        expZ = expm(Ts * Z)
    
        # Extracting blocks: top-left is A_d, top-right is B_d
        A_d = expZ[:n, :n]
        B_d = expZ[:n, n:]
    
        return A_d, B_d

    def coords_rocket (self, x, u):
        px    = x[0]
        py    = x[1]
        theta = x[4]
    
        Fe  = u[0]*self.Kanim_Fe
        Fs  = u[1]*self.Kanim_Fs
        phi = u[2]*self.Kanim_phi
       
        rocket_shape = self.rocket_shape
        ls = self.ls
        l1 = self.l1
        l2 = self.l2
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        shift_matrix = np.array([px, py])
        
        if Fs>0:
            flow_lr_shape =  np.array([[-ls,  l2],[-ls-Fs,  l2]]) # left
        else:
            flow_lr_shape =  np.array([[ ls,  l2],[ ls-Fs,  l2]]) # right
            
        flow_e_shape =  np.array([[0,  -l1],[np.sin(phi)*Fe,  -l1-np.cos(phi)*Fe]])
        
        rocket_coords  = np.dot(rocket_shape,  rotation_matrix) + shift_matrix
        flow_lr_coords = np.dot(flow_lr_shape, rotation_matrix) + shift_matrix
        flow_e_coords  = np.dot(flow_e_shape,  rotation_matrix) + shift_matrix
        return rocket_coords, flow_lr_coords, flow_e_coords, px, py
   
    
    def draw_rocket(self, ax, x, u):
        rocket_coords, flow_lr_coords, flow_e_coords, px, py = self.coords_rocket (x, u)
        rocket  = patches.Polygon(rocket_coords,  facecolor='none', edgecolor='black', linewidth=2.0)
        flow_lr = patches.Polygon(flow_lr_coords, facecolor='none', edgecolor='red',   linewidth=2.1)
        flow_e  = patches.Polygon(flow_e_coords,  facecolor='none', edgecolor='red',   linewidth=4.2)
    
        ax.add_patch(rocket)
        ax.add_patch(flow_lr)
        ax.add_patch(flow_e)
        ax.plot(px, py, 'b*') # Rocket's center of mass
        
        return 

    def draw_seaplatform (self, ax, rx, draw_ref=False, ref_alfa=0.2 ):

        min_x = self.gr_min_x
        max_x = self.gr_max_x
        xlong = max_x - min_x

        min_y = self.gr_min_y

        platf_w = self.platf_w
        px = rx - 0.5*platf_w
        
        sea = patches.Rectangle((min_x, min_y), xlong, -min_y, facecolor='blue', alpha=0.4) # sea min_y..0
        platform = patches.Rectangle((px, 0), platf_w, self.platf_h, facecolor='black' ) # platform 0..platf_h
             
        ax.add_patch(sea)
        ax.add_patch(platform)

        if draw_ref:
            drx = self.tol[0]
            dry = self.tol[1]
            ref = patches.Rectangle((self.rx-drx, self.ry-dry), 2*drx, 2*dry, facecolor='green', alpha=ref_alfa )
            #print ((self.rx-drx, self.ry-dry))
            ax.add_patch(ref)
        return

    def draw_fulltrajectory (self, ax, X, linestyle = 'b--', linewidth=1.0, label = '' ):
        ax.plot(X[0,  :], X[1,  :], linestyle, linewidth=linewidth, label = label) # full trajectory
        ax.plot(X[0,  0], X[1,  0], 'bo')  # Start
        ax.plot(X[0, -1], X[1, -1], 'ro')  # Finish
        return
    
    
    def draw_fig (self, ax, title = 'Trajectory', gridlinestyle="--", xlabel = 'p_x, m', ylabel = 'p_y, m', show_legend = False, fname=None, dpi=100 ):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.gr_min_x, self.gr_max_x)
        ax.set_ylim(self.gr_min_y, self.gr_max_y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=gridlinestyle, alpha=0.5)
        if show_legend:
            ax.legend()
        if fname is not None:
            plt.savefig(fname, dpi=dpi) # bbox_inches='tight'
        plt.show()
        return 


    def plot_states (self,X, figsize = (12,8), fname=None, dpi=100):
        
        Nx = X.shape[1]
        tx = np.arange(Nx) * self.dt    # 

        # Create a figure with 3 rows and 1 column
        fig, ax = plt.subplots(3, 2, figsize=figsize)
        
        ax[0, 0].plot(tx, X[0, :], 'r-')
        ax[0, 0].set_ylabel('p_x, m')
        ax[0, 0].set_title('Position x')
        ax[0, 0].grid(True, linestyle="--", alpha=0.6)
        
        ax[1, 0].plot(tx, X[1, :], 'g-')
        ax[1, 0].set_ylabel('p_y, m')
        ax[1, 0].set_title('Position y')
        ax[1, 0].grid(True, linestyle="--", alpha=0.6)
        
        ax[2, 0].plot(tx, X[4, :], 'b-')
        ax[2, 0].set_xlabel('time, sec')
        ax[2, 0].set_ylabel('θ, rad')
        ax[2, 0].set_title('Angle θ')
        ax[2, 0].grid(True, linestyle="--", alpha=0.6)
        
        
        ax[0, 1].plot(tx, X[2, :], 'r-')
        ax[0, 1].set_ylabel('v_x, m/s') 
        ax[0, 1].set_title('Velocity x')
        ax[0, 1].grid(True, linestyle="--", alpha=0.6)
        
        ax[1, 1].plot(tx, X[3, :], 'g-')
        ax[1, 1].set_ylabel('v_y, m/s') 
        ax[1, 1].set_title('Velocity y')
        ax[1, 1].grid(True, linestyle="--", alpha=0.6)
        
        ax[2, 1].plot(tx, X[5, :], 'b-')
        ax[2, 1].set_xlabel('time, sec')
        ax[2, 1].set_ylabel('ω, rad/s') 
        ax[2, 1].set_title('Angular velocity ω')
        ax[2, 1].grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout() 
        if fname is not None:
            plt.savefig(fname, dpi=dpi) # bbox_inches='tight'
        plt.show()
        return

    def plot_control (self, U, figsize = (12,8), fname=None, dpi=100):        
        Nu = U.shape[1]
        tu = np.arange(Nu) * self.dt    # Control time grid (N points)
        
        # Create a figure with 3 rows and 1 column
        fig, ax = plt.subplots(3, 1, figsize=figsize)
        
        # Row 0: Main Thrust
        ax[0].plot(tu, U[0, :], 'r-')
        ax[0].set_ylabel('F_e')
        ax[0].set_title('Main Thrust F_e')
        ax[0].grid(True, linestyle="--", alpha=0.6)
        
        # Row 1: Side Force
        ax[1].plot(tu, U[1, :], 'g-')
        ax[1].set_ylabel('F_s')
        ax[1].set_title('Side Force F_s')
        ax[1].grid(True, linestyle="--", alpha=0.6)
        
        # Row 2: Nozzle Gimbal Angle
        ax[2].plot(tu, U[2, :], 'b-')
        ax[2].set_xlabel('Time, sec')
        ax[2].set_ylabel('φ')
        ax[2].set_title('Nozzle Deflection Angle φ')
        ax[2].grid(True, linestyle="--", alpha=0.6)
        
        # Final layout adjustment and rendering
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname, dpi=dpi) # bbox_inches='tight'
        plt.show()
        return        

    def anim_trajectory (self, X, U, figsize = (12,8)):
        num_frames = X.shape[1] - 1 # X.shape[1] - 1
        
        fig, ax = plt.subplots(figsize=figsize)
        self.draw_fulltrajectory (ax, X)
        self.draw_seaplatform(ax, self.rx, draw_ref=True, ref_alfa=0.2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(self.gr_min_x, self.gr_max_x)
        ax.set_ylim(self.gr_min_y, self.gr_max_y)
        ax.grid(True, linestyle="--", alpha=0.4)
        
        # init empty rocket-lines
        (rocket_line,)  = ax.plot([], [], lw=2, color="navy")
        (flow_lr_line,) = ax.plot([], [], lw=2, color="red")
        (flow_e_line,)  = ax.plot([], [], lw=4, color="red")
        
        def anim_init():
            rocket_line.set_data([], [])
            flow_lr_line.set_data([], [])
            flow_e_line.set_data([], [])
            return rocket_line, flow_lr_line, flow_e_line
        
        def anim_update(frame):
            rocket_coords, flow_lr_coords, flow_e_coords, px, py  = self.coords_rocket (X[:,frame], U[:,frame])
            rocket_line.set_data (rocket_coords[:, 0], rocket_coords[:, 1])
            flow_lr_line.set_data(flow_lr_coords[:, 0], flow_lr_coords[:, 1])
            flow_e_line.set_data (flow_e_coords[:, 0], flow_e_coords[:, 1])
            return rocket_line, flow_lr_line, flow_e_line
        
        ani = FuncAnimation(
            fig,
            anim_update,
            frames=num_frames,
            init_func=anim_init,
            blit=True,
            interval=int(self.dt * 1000),
        )
        plt.close()
        return ani
