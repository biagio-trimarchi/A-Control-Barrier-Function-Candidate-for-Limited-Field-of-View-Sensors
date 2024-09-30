from utilities import *
font_size = 30
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

class Simulation:
    def __init__(self):
        pass

    def initialize(self):
        ### TIME RELATED DATA
        self.T = 100.000               # Total simulation timej
        self.dt = 0.001              # Simulation step size
        self.t = 0.0                 # Current time
        self.time_stamps = []  # Time stamps for plot

        ### TRAJECTORY DATA
        # Spiral
        self.radius = 4.0
        self.center = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = 1.0
        self.climb_velocity = 1.0

        ### AGENT DATA
        # Agent state
        self.p = np.array([0.0, 0.0, 0.0])    # Current position
        self.v = np.array([0.0, 0.0, 0.0])             # Current velocity
        self.phi = 0.0                                 # Current roll
        self.theta = 0.0                               # Current pitch
        self.psi = 0.0                                 # Current yaw
        self.R = angleToRotationMatrix(0.0, 0.0, 0.0)  # Current orientation
        self.w = np.array([0.0, 0.0, 0.0])             # Current angular velocity

        # Camera paramters
        self.FOV = PI / 6.0                          # Half camera field of view
        self.camera_axis = np.array([1.0, 0.0, 0.0])  # Camera axis in body frame

        ### FEATURES
        self.features = []
        self.features.append(np.array([7.0, -1.5, 1.5]))
        self.features.append(np.array([7.0, 1.5,  1.5]))
        self.features.append(np.array([6.0, 1.5, -1.5]))
        self.features.append(np.array([6.0, -1.5, -1.5]))

        self.old_u = []
        for ff in self.features:
            u = ff - self.p
            d = np.linalg.norm(u)
            u = u / d
            self.old_u.append(u.copy())

        ### PLOT
        self.norm_ep = []
        self.plot_h = []

        ### ANIMATION
        # Setup
        self.animation_figure = plt.figure(1)
        self.animation_axis = plt.axes(projection='3d')
        self.camera = Camera(self.animation_figure)

    def run(self):
        ### RUN SIMULATION
        ### AUXILIARY VARIABLES
        tspan = np.arange(0.0, self.T, self.dt) # Time span for tqdm
        counter = 32                             # Counter to slow down animation

        ### MAIN LOOP
        for tt in tqdm(tspan):
            # Increase time
            self.t += self.dt
            counter += 1
            self.time_stamps.append(self.t)

            # Compute control input
            self.a = np.array([0.0, 0.0, 0.0])
            self.alpha = np.array([0.0, 0.0, 0.0])

            self.PD()
            self.CBF()

            # Update state
            self.p = self.p + self.v * self.dt
            self.v = self.v + self.a * self.dt

            # w dt = u phi 
            rotation_angle = 0.0
            rotation_axis = np.zeros((DIM,))
            if np.linalg.norm(self.w) > 1e-6:
                rotation_axis = self.w / np.linalg.norm(self.w)
                rotation_angle = np.linalg.norm(self.w) * self.dt
            
            self.R = self.R @ exponentialMap(rotation_axis, rotation_angle)
            self.w = self.w + self.alpha * self.dt

            # Snap frame for animation
            if counter == 33:
                counter = 0
                #self.addFrame()

    def animation(self):
        print("Creating animation")
        animation = self.camera.animate()
        plt.show()
        print("Done")
    
    def addFrame(self):
        self.animation_axis.plot(self.p[0], self.p[1], self.p[2], 'ob')
        self.animation_axis.plot(self.p_des[0], self.p_des[1], self.p_des[2], 'og')
        for f in self.features:
            self.animation_axis.plot(f[0], f[1], f[2], 'xr')

        drawHalfCone(self.animation_axis, self.p,
                     self.FOV, R=self.R, plot_type='circle')

        self.animation_axis.set_title("Agent FOV")
        self.animation_axis.set_xlabel("x (m)")
        self.animation_axis.set_ylabel("y (m)")
        self.animation_axis.set_zlabel("z (m)")
        self.animation_axis.set_xlim([-5.0, 10.0])
        self.animation_axis.set_ylim([-5.0, 5.0])
        self.animation_axis.set_zlim([-5.0, 5.0])

        self.camera.snap()

    def PD(self):
        # Static proportional gain control law
        K_p = 20.8               # Position error gain
        K_v = 13.3               # Velocity error gain

        # Desired trajectory
        rx = 1.0
        omega_x = 0.3

        ry = 10.0
        omega_y = 0.2

        rz = 2.0
        omega_z = 0.2

        # Desired position
        self.p_des = np.array([  rx * sin(omega_x * self.t),
                                 ry * sin(omega_y * self.t),
                                 rz * sin(omega_z * self.t)])
        
        self.v_des = np.array([ omega_x * rx * cos(omega_x * self.t),
                                omega_y * ry * cos(omega_y * self.t),
                                omega_z * rz * cos(omega_z * self.t)])

        self.a_des = np.array([ (omega_x)**2 * rx * -sin(omega_x * self.t),
                                (omega_y)**2 * ry * -sin(omega_y * self.t),
                                (omega_z)**2 * rz * -sin(omega_z * self.t)])

        # Control input
        self.a = -K_p * (self.p - self.p_des) -K_v * (self.v - self.v_des) + self.a_des
        self.norm_ep.append(self.p - self.p_des)

    def CBF(self):
        ### MEASURABLE VALUES
        z = self.R @ self.camera_axis
        
        ### QP VARIABLES
        feat_n = len(self.features)
        opt_var_n = 2*DIM + 4*feat_n
        inequality_A = np.zeros((5*feat_n, opt_var_n))
        inequality_B = np.zeros((5*feat_n, ))

        equality_A = np.zeros((feat_n, opt_var_n))
        equality_B = np.zeros((feat_n, ))

        input_nom = np.zeros((opt_var_n,))
        input_nom[0:3] = self.a.copy()
        input_nom[3:6] = self.alpha.copy()

        ### CBF
        gamma1 = 80.0
        gamma2 = 2.0

        k1 = gamma2 * gamma1
        k2 = gamma1 + gamma2
        a = 3.0
        K = (4.0 * gamma1 * gamma2) / ((gamma1 + gamma2) ** 2.0)
        dMax = 15.0
        c2_lower = (K * dMax - a) / (dMax - 1.0)

        min_h = 1.0
        for i in range(len(self.features)):
            ### BEARING
            # Measurements
            u = self.features[i] - self.p
            norm = np.linalg.norm(u)
            u = u / norm
            du = (u - self.old_u[i]) / self.dt
            self.old_u[i] = u.copy()

            # Barrier function quantities
            h = u @ z - cos(self.FOV)
            if h < min_h:
                min_h = h

            h = h - 0.0005
            grad_p = (-z + (u @ z) * u) / norm
            grad_R = -(self.R @ hatMap(self.camera_axis)).T @ u 
            dh = (grad_p @ self.v + grad_R @ self.w)
            P = np.eye(DIM) - u.reshape((DIM,1)) @ u.reshape((1, DIM))

            hess_RR = hatMap(self.R.T @ u) @ hatMap(self.camera_axis)
            hess_p_psi_v = -(self.R @ hatMap(self.camera_axis)).T @ du 

            hess_p_p = 3*(z@u)*(-P) + (-u.reshape((DIM,1)) @ z.reshape((1,DIM)) + 2*z@u*np.eye(DIM) - z.reshape((DIM,1)) @ u.reshape((1,DIM)))

            inequality_A[i, 0:3] = (-z + (u @ z) * u)
            #inequality_A[i, 2*DIM + i] = 1.0
            inequality_A[i, 2*DIM + i] = k2 * dh + k1 * h
            inequality_A[i, 2*DIM + 2*feat_n + i] = 1.0

            inequality_A[i + len(self.features), 3:6] = -(self.R @ hatMap(self.camera_axis)).T @ u 
            #inequality_A[i + len(self.features), 2*DIM + i + len(self.features)] = 1.0
            inequality_A[i + feat_n, 2*DIM + feat_n+i] = k2 * dh + k1 * h
            inequality_A[i + feat_n, 2*DIM + 3*feat_n+i] = 1.0

            # Delta positive
            inequality_A[2*feat_n+i, 1+DIM+2*feat_n+i] = 1.0
            inequality_A[3*feat_n+i, 1+DIM+3*feat_n+i] = 1.0

            equality_A[i, 2*DIM + i] = 1.0
            equality_A[i, 2*DIM + i + len(self.features)] = 1.0

            ### Hessian splitting
            # Component without distance 
            A = -(z@u) * (du @ du)
            # Computing velocity component parallel to (d / dt) u
            if (np.linalg.norm(du) > 1e-6):
                v_t =  (self.v @ du) * du / (np.linalg.norm(du) ** 2)
            else:
                v_t = np.zeros((DIM,))

            # Component with distance
            B = -2.0 * (self.v @ u) * (z @ v_t)

            inequality_B[i] = B
            inequality_B[i + len(self.features)] = \
                              A + 2 * hess_p_psi_v @ self.w + \
                              (hess_RR @ self.w) @ self.w 
            equality_B[i] = a

        for i in range(feat_n):
            inequality_A[-i-1, 1+DIM+feat_n+i] = 1.0
            inequality_B[-i-1] = -c2_lower

        ### QP COST
        H = np.zeros((opt_var_n, opt_var_n))
        H[0,0] = H[1,1] = H[2,2] = 1.0
        H[3,3] = H[4,4] = H[5,5] = 1.0
        for i in range(feat_n):
            H[-1-i, -1-i] = 10.0
            H[-feat_n -i-1, -feat_n-i-1] = 10.0
        qp_solution = solve_qp(H, -H@input_nom, -inequality_A, inequality_B, equality_A, equality_B, solver='gurobi')
        self.a = qp_solution[0:3]
        self.alpha = qp_solution[3:6]

        self.plot_h.append(min_h)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_stamps, self.norm_ep, linewidth=2)
        ax.set_title("Position Tracking Error", fontsize=20)
        ax.legend([r"$x$", r"$y$", r"$z$"], fontsize=20)
        ax.set_xlabel("Time (s)", fontsize=font_size)
        ax.set_xticks([20*i for i in range(5)])
        ax.tick_params(axis='x', labelsize=font_size)
        ax.set_ylabel(r"$p - p_{des}$(m)", fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.grid()
        ax.set_xlim([self.dt, self.T])
        fig.savefig("/home/biagio/Projects/VisionBasedCBF/Images/DITrackingError.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        zero_baseline = [0.0 for i in self.time_stamps]
        ax.plot(self.time_stamps, self.plot_h, linewidth=2)
        ax.plot(self.time_stamps, zero_baseline, "--", linewidth=2)
        ax.set_title("Barrier Function Value", fontsize=20)
        ax.set_xlabel("Time (s)", fontsize=font_size)
        ax.set_xlabel("Time (s)", fontsize=font_size)
        ax.set_ylabel(r"$\min_i h_i(x)$ (m)", fontsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)
        ax.set_xticks([20*i for i in range(5)])
        ax.grid()
        ax.set_xlim([self.dt, self.T])
        fig.savefig("/home/biagio/Projects/VisionBasedCBF/Images/DIBarrier.png", bbox_inches='tight')

        plt.show()


def main():
    simulation = Simulation()
    simulation.initialize()
    simulation.run()
    #simulation.animation()
    simulation.plot()

if __name__ == '__main__':
    main()
