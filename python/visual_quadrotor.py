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
        self.T = 100.000              # Total simulation time
        self.dt = 0.001              # Simulation step size
        self.framerate = 33
        self.t = 0.0                 # Current time
        self.time_stamps = []        # Time stamps for plot

        ### AGENT DATA
        # Agent parameters
        self.m = 4.34                        # Mass
        self.J = np.eye(DIM)                 # Inertia moment
        self.J[0,0] = 0.082
        self.J[1,1] = 0.0845
        self.J[2,2] = 0.1377
        self.Jinv = np.linalg.inv(self.J)
        self.g = np.array([0.0, 0.0, -G])  # Gravity

        # Agent state
        self.p = np.array([0.0, 0.0, 0.0])         # Current position
        self.v = np.array([0.0, 0.0, 0.0])                  # Current velocity
        self.phi = 0.0                                      # Current roll
        self.theta = 0.0                                    # Current pitch
        self.psi = 0.0                                      # Current yaw
        self.R = angleToRotationMatrix(0.0, 0.0, self.psi)  # Current orientation
        self.w = np.array([0.0, 0.0, 0.0])                  # Current angular velocity
        self.u = np.array([0.0, 0.0, 0.0, 0.0])             # Control input

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
        self.norm_ev = []
        self.norm_eR = []
        self.norm_ew = []
        self.norm_wdes = []
        self.norm_w = []
        self.norm_alpha = []
        self.norm_alpha_nom = []
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
        counter = self.framerate - 1                             # Counter to slow down animation

        ### MAIN LOOP
        for tt in tqdm(tspan):
            if (np.linalg.det(self.R) < 0.0):
                print(self.t)
                assert False

            # Increase time
            self.t += self.dt
            counter += 1
            self.time_stamps.append(self.t)

            # Compute control input
            self.u = np.array([0.0, 
                               0.0, 
                               0.0, 
                               0.0])

            self.trajectory()
            self.differentialFlatness()
            self.controller()
            self.CBF()

            # Update state
            self.dynamics()

            # Snap frame for animation
            if counter == self.framerate:
            #if self.t > self.T - 2*self.dt:
                counter = 0
                #self.addFrame()

    def animation(self):
        print("Creating animation")
        animation = self.camera.animate()
        animation.save('complex.mp4', fps=30, codec='libx264')
        #plt.show()
        print("Done")
    
    def addFrame(self):
        # Plot reference rotation
        x_axis_to_follow = np.array([1.0, 0.0, 0.0])
        x_axis_to_follow = self.p + self.R_des @ x_axis_to_follow
        self.animation_axis.plot([self.p[0], x_axis_to_follow[0]],
                                 [self.p[1], x_axis_to_follow[1]],
                                  '-g',
                                  zs=[self.p[2], x_axis_to_follow[2]])

        y_axis_to_follow = np.array([0.0, 1.0, 0.0])
        y_axis_to_follow = self.p + self.R_des @ y_axis_to_follow
        self.animation_axis.plot([self.p[0], y_axis_to_follow[0]],
                                 [self.p[1], y_axis_to_follow[1]],
                                  '-b',
                                  zs=[self.p[2], y_axis_to_follow[2]])

        z_axis_to_follow = np.array([0.0, 0.0, 1.0])
        z_axis_to_follow = self.p + self.R_des @ z_axis_to_follow
        self.animation_axis.plot([self.p[0], z_axis_to_follow[0]],
                                 [self.p[1], z_axis_to_follow[1]],
                                  '-r',
                                  zs=[self.p[2], z_axis_to_follow[2]])

        z_axis = np.array([0.0, 0.0, 1.0])
        z_axis = self.p + self.R @ z_axis
        self.animation_axis.plot([self.p[0], z_axis[0]],
                                 [self.p[1], z_axis[1]],
                                  '--r',
                                  zs=[self.p[2], z_axis[2]])


        drawQuadrotor(self.animation_axis, self.p, self.R, 1.2)
        self.animation_axis.plot(self.p_des[0], self.p_des[1], self.p_des[2], 'og')
        for f in self.features:
            self.animation_axis.plot(f[0], f[1], f[2], 'xr')

        drawHalfCone(self.animation_axis, self.p,
                     self.FOV, R=self.R, plot_type='circle')

        self.animation_axis.set_title("Agent FOV")
        self.animation_axis.set_xlabel("x (m)")
        self.animation_axis.set_ylabel("y (m)")
        self.animation_axis.set_zlabel("z (m)")
        self.animation_axis.set_xlim([-2.0, 10.0])
        self.animation_axis.set_ylim([-5.0, 5.0])
        self.animation_axis.set_zlim([-1.0, 5.0])

        self.camera.snap()

    def dynamics(self):
        # Auxiliary quantities
        e3 = np.array([0.0, 0.0, 1.0])
        rotation_angle = 0.0
        rotation_axis = np.zeros((DIM,))
        if np.linalg.norm(self.w) > 1e-6:
            rotation_axis = self.w / np.linalg.norm(self.w)
            rotation_angle = np.linalg.norm(self.w) * self.dt

        # Translation
        self.p = self.p + self.v * self.dt
        self.v = self.v + (1 / self.m) * (self.R @ e3 * self.u[0] + self.m * self.g) * self.dt

        # Rotation
        self.R = self.R @ exponentialMap(rotation_axis, rotation_angle)
        self.w = self.w + np.linalg.inv(self.J) @ (self.u[1:4] - hatMap(self.w) @ (self.J @ self.w)) * self.dt
        self.norm_w.append(self.w.copy())

    def trajectory(self):
        # Desired trajectory
        rx = 1.0
        omega_x = 0.5

        ry = 10.0
        omega_y = 1.0

        rz = 2.0
        omega_z = 0.5

        self.p_des = np.array([  rx * sin(omega_x * self.t),
                                 ry * sin(omega_y * self.t),
                                 rz * sin(omega_z * self.t)])
        
        self.v_des = np.array([ omega_x * rx * cos(omega_x * self.t),
                                omega_y * ry * cos(omega_y * self.t),
                                omega_z * rz * cos(omega_z * self.t)])

        self.a_des = np.array([ (omega_x)**2 * rx * -sin(omega_x * self.t),
                                (omega_y)**2 * ry * -sin(omega_y * self.t),
                                (omega_z)**2 * rz * -sin(omega_z * self.t)])

        self.j_des = np.array([ (omega_x)**3 * rx * -cos(omega_x * self.t),
                                (omega_y)**3 * ry * -cos(omega_y * self.t),
                                (omega_z)**3 * rz * -cos(omega_z * self.t)])
                    

        self.s_des = np.array([  (omega_x)**4 * rx * sin(omega_x * self.t),
                                 (omega_y)**4 * ry * sin(omega_y * self.t),
                                 (omega_z)**4 * rz * sin(omega_z * self.t)])

        roll, pitch, yaw = rotationMatrixToAngle(self.R)
        self.yaw_des = yaw
        #self.yaw_des   = 0.0

        M = eulerRatesToBodyRatesMatrix(roll, pitch, yaw)
        self.dyaw_des = (np.linalg.inv(M) @ self.w)[2]
        #self.dyaw_des  = 0.0

        self.ddyaw_des = 0.0

    def differentialFlatness(self):
        # Compute desired rotation matrix, angular velocities, and 
        # accelerations from given trajectory

        g = np.array([0.0, 0.0, -G])
        self.R_des = I.copy()
        self.w_des = np.zeros((DIM,))

        # Desired orientation
        z_des = self.a_des - g
        trust = self.m * np.linalg.norm(z_des)
        if np.linalg.norm(z_des) > 1e-3:
            z_des = z_des / np.linalg.norm(z_des)
        else:
            z_des = np.array([0.0, 0.0, 1.0])

        x_c = np.array([cos(self.yaw_des), sin(self.yaw_des), 0.0])
        y_des = np.cross(z_des, x_c)
        y_des = y_des / np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)

        self.R_des[:, 0] = x_des
        self.R_des[:, 1] = y_des
        self.R_des[:, 2] = z_des

        # Desired angular velocities
        h_w = (self.m / trust) * (self.j_des - (z_des @ self.j_des) * z_des)
        self.w_des[0] = -h_w @ y_des                                         # p
        self.w_des[1] = h_w @ x_des                                          # q
        self.w_des[2] = self.dyaw_des * (np.array([0.0, 0.0, 1.0]) @ z_des)  # r
        self.norm_wdes.append(self.w_des.copy())

        # Desired angular velocites
        self.alpha_des = np.zeros((DIM,))

        aux1 = self.m * self.s_des
        aux2 = np.cross(self.w_des, np.cross(self.w_des, trust * z_des))
        aux3 = 2 * np.cross(self.w_des, (self.m * self.j_des @ z_des) * z_des)
        aux4 = aux1 @ z_des - aux2 @ z_des
        ha = (aux1 - aux4 * z_des - aux2 - aux3) / trust

        self.alpha_des[0] = - ha @ y_des
        self.alpha_des[1] =   ha @ x_des
        self.alpha_des[2] =   self.ddyaw_des * z_des[2]

    def controller(self):
        # Lyapunov Function takes from
        # https://ieeexplore.ieee.org/document/5717652
        # https://arxiv.org/pdf/1003.2005
        ### Auxiliary quantities
        g = np.array([0.0, 0.0, -G])

        # Translational components
        k_p = 20.8
        k_v = 13.3
        e_p = self.p - self.p_des
        e_v = self.v - self.v_des

        F = -(k_p * e_p + k_v * e_v + self.m * g - self.m * self.a_des)

        self.u[0] = F @ (self.R[:,2]).reshape((DIM,))
        z_des = F.copy()
        if np.linalg.norm(z_des) > 1e-3:
            z_des = z_des / np.linalg.norm(z_des)
            if (z_des @ np.array([0.0, 0.0, -1.0]) > 0.0):
                z_des = -z_des
                self.u[0] = 0.0
        else:
            z_des = np.array([0.0, 0.0, 1.0])

        feature_sum = np.zeros((3,))
        for ff in self.features:
            u = ff - self.p
            u = u / np.linalg.norm(u)
            feature_sum += u
        x_c = feature_sum / np.linalg.norm(feature_sum)
        x_c = np.array([cos(self.yaw_des), sin(self.yaw_des), 0.0])

        y_des = np.cross(z_des, x_c)
        y_des = y_des / np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)

        self.R_des[:, 0] = x_des
        self.R_des[:, 1] = y_des
        self.R_des[:, 2] = z_des

        # Rotational components
        k_R = 54.81
        k_w = 10.54
        e_w = self.w - self.w_des
        e_R = 0.5 * veeMap(self.R_des.T @ self.R - self.R.T @ self.R_des)

        #e_R = e_R * (1.0 / np.sqrt(1.0 + np.trace(self.R_des.T @ self.R)))

        M = -k_R * e_R - k_w * e_w + \
             np.cross(self.w, self.J @ self.w) - \
             self.J @ (hatMap(self.w) @ self.R.T @ self.R_des @ self.w_des) + \
             self.J @ (self.R.T @ self.R_des @ self.alpha_des)

        self.u[1:4] = M

        self.norm_ep.append(e_p)
        self.norm_ev.append(e_v)
        self.norm_eR.append(e_R)
        self.norm_ew.append(e_w)

    def CBF(self):
        ### MEASURABLE VALUES
        z = self.R @ self.camera_axis
        
        ### QP VARIABLES
        feat_n = len(self.features)
        opt_var_n = 1 + DIM + 4*feat_n
        inequality_A = np.zeros((5*feat_n, opt_var_n))
        inequality_B = np.zeros((5*feat_n, ))

        equality_A = np.zeros((feat_n, opt_var_n))
        equality_B = np.zeros((feat_n, ))

        # Nominal input
        input_nom = np.zeros((opt_var_n,))
        input_nom[0] = self.u[0]
        input_nom[1:4] = np.linalg.inv(self.J) @ (self.u[1:4] - hatMap(self.w) @ (self.J @ self.w)) 
        self.norm_alpha_nom.append(input_nom[1:4])

        ### CBF
        gamma1 = 80.0
        gamma2 = 2.0

        k1 = gamma2 + gamma1
        k2 = gamma2 * gamma1
        a = 3.0
        K = (4.0 * gamma1 * gamma2) / ((gamma1 + gamma2) ** 2.0)
        dMax = 15.0
        c2_lower = (K * dMax - a) / (dMax - 1.0)
        #print("a: ", a)
        #print("K: ", K)
        #print("c2: ", c2_lower)

        min_h = 1.0
        for i in range(len(self.features)):
            ### BEARING
            # Measurements
            u = self.features[i] - self.p
            norm = np.linalg.norm(u)
            if norm > dMax:
                print("d: ", norm)
                exit()
            u = u / norm
            du = (u - self.old_u[i]) / self.dt
            self.old_u[i] = u.copy()

            # Barrier function quantities
            h = u @ z - cos(self.FOV)
            if h < min_h:
                min_h = h

            assert h > -0.5
            grad_p = (-z + (u @ z) * u)
            grad_p = grad_p / norm
            grad_R = -(self.R @ hatMap(self.camera_axis)).T @ u 
            dh = (grad_p @ self.v + grad_R @ self.w) # In theory I should compute it trough du...
            grad_p = grad_p * norm
            P = np.eye(DIM) - u.reshape((DIM,1)) @ u.reshape((1, DIM))

            hess_RR = hatMap(self.R.T @ u) @ hatMap(self.camera_axis)
            hess_p_psi_v = -(self.R @ hatMap(self.camera_axis)).T @ du 

            hess_p_p = 3*(z@u)*(-P) + (-u.reshape((DIM,1)) @ z.reshape((1,DIM)) + 2*z@u*np.eye(DIM) - z.reshape((DIM,1)) @ u.reshape((1,DIM)))
            #hess_p_p = hess_p_p / (norm ** 2)

            inequality_A[i, 0] = (1 / self.m) * grad_p @ (self.R[:, 2].reshape((DIM,)))
            #inequality_A[i, 1+DIM + i] = 1.0
            inequality_A[i, 1+DIM + i] = k2 * dh + k1 * h
            inequality_A[i, 1+DIM + 2*feat_n + i] = 1.0

            inequality_A[i + feat_n, 1:4] = -(self.R @ hatMap(self.camera_axis)).T @ u 
            #inequality_A[i + feat_n, 1+DIM + feat_n+i] = 1.0
            inequality_A[i + feat_n, 1+DIM + feat_n+i] = k2 * dh + k1 * h
            inequality_A[i + feat_n, 1+DIM + 3*feat_n+i] = 1.0

            # Delta positive
            inequality_A[2*feat_n+i, 1+DIM+2*feat_n+i] = 1.0
            inequality_A[3*feat_n+i, 1+DIM+3*feat_n+i] = 1.0

            equality_A[i, 1+DIM + i] = 1.0
            equality_A[i, 1+DIM + feat_n+i] = 1.0

            ### Hessian splitting
            # Component without distance
            A = -(z@u) * (du @ du)
            # Computing velocity component parallel to (d / dt) u
            if (np.linalg.norm(du) > 1e-6):
                v_t =  (self.v @ du) * du / (np.linalg.norm(du) ** 2)
            else:
                v_t = np.zeros((DIM,))

            # Component with distance
            B = -2.0 * (self.v @ u) * ( z @ v_t)

            inequality_B[i] = B + grad_p @ self.g

            inequality_B[i + feat_n] = (hess_RR @ self.w) @ self.w + \
                                       2 * hess_p_psi_v @ self.w + \
                                       A

            #equality_B[i] = k2 * dh + k1 * h
            equality_B[i] = a

        # Positive trust constraint
        #inequality_A[-1,0] = 1.0
        #inequality_B[-1] = 0.0

        for i in range(feat_n):
            inequality_A[-i-1, 1+DIM+feat_n+i] = 1.0
            inequality_B[-i-1] = -c2_lower

        ### QP COST
        H = np.zeros((opt_var_n, opt_var_n))
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,2] = 1.0
        H[3,3] = 1.0
        for i in range(feat_n):
            H[-1-i, -1-i] = 100.0
            H[-feat_n -i-1, -feat_n-i-1] = 100.0
        #print(inequality_A)
        #print(inequality_B)
        #print(H)
        #exit()

        qp_solution = solve_qp(H, -H@input_nom, -inequality_A, inequality_B, equality_A, equality_B, solver='gurobi')
        #print("c2: ", qp_solution[-1])
        #print("c2: ", qp_solution[-2])
        #print("c2: ", qp_solution[-3])
        #print("c2: ", qp_solution[-4])
        #print("lower: ", c2_lower)
        #print(" ")
        self.u[0] = qp_solution[0]
        self.alpha = qp_solution[1:4]
        self.norm_alpha.append(self.alpha.copy())
        self.u[1:4] = self.J @ self.alpha + np.cross(self.w, self.J @ self.w)

        self.plot_h.append(min_h)
        #print(qp_solution)

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
        #fig.savefig("/home/biagio/Projects/VisionBasedCBF/Images/DroneTrackingError.png", bbox_inches='tight')

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
        #fig.savefig("/home/biagio/Projects/VisionBasedCBF/Images/DroneBarrier.png", bbox_inches='tight')

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_ev)
        #ax.set_title("Velocity error")
        #ax.legend(["x", "y", "z"])

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_eR)
        #ax.set_title("Rotation error")
        #ax.legend(["x", "y", "z"])

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_ew)
        #ax.set_title("Angular velocity error")
        #ax.legend(["x", "y", "z"])

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_wdes)
        #ax.set_title("Angular velocity target")
        #ax.legend(["x", "y", "z"])

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_w)
        #ax.set_title("Angular velocity")
        #ax.legend(["x", "y", "z"])

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_alpha)
        #ax.set_title("Angular acceleration")
        #ax.legend(["x", "y", "z"])

        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_alpha_nom)
        #ax.set_title("Angular acceleration nominal")
        #ax.legend(["x", "y", "z"])


        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # Desired trajectory
        rx = 1.0
        omega_x = 0.5
        ry = 10.0
        omega_y = 0.2
        rz = 2.0
        omega_z = 0.3

        traj_plot = []
        for tt in range(700):
            t = tt * 0.1
            traj_plot.append(np.array([  rx * sin(omega_x * t),
                                 ry * sin(omega_y * t),
                                 rz * sin(omega_z * t)]))
        ax.plot3D([p[0] for p in traj_plot], 
                  [p[1] for p in traj_plot], 
                   'xk', 
                   zs=[p[2] for p in traj_plot], 
                   markersize=2.0)

        for ff in self.features:
            ax.plot3D(ff[0], ff[1], "or", zs=ff[2], markersize=10.0)

        ax.plot([self.features[0][0], self.features[1][0]],
                [self.features[0][1], self.features[1][1]],
                'r-',
                zs=[self.features[0][2], self.features[1][2]],
                linewidth=2)

        ax.plot([self.features[1][0], self.features[2][0]],
                [self.features[1][1], self.features[2][1]],
                'r-',
                zs=[self.features[1][2], self.features[2][2]],
                linewidth=2)

        ax.plot([self.features[2][0], self.features[3][0]],
                [self.features[2][1], self.features[3][1]],
                'r-',
                zs=[self.features[2][2], self.features[3][2]],
                linewidth=2)

        ax.plot([self.features[3][0], self.features[0][0]],
                [self.features[3][1], self.features[0][1]],
                'r-',
                zs=[self.features[3][2], self.features[0][2]],
                linewidth=2)

        drawQuadrotor(ax, self.p, self.R, 1.2)
        drawHalfCone(ax, self.p,
                     self.FOV, R=self.R, plot_type='circle')
        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.show()

def main():
    simulation = Simulation()
    simulation.initialize()
    simulation.run()
    #simulation.animation()
    simulation.plot()

if __name__ == '__main__':
    main()
