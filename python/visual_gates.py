from utilities import *
import quadraticBezierPathCreator as bezQP

class Simulation:
    def __init__(self):
        pass

    def initialize(self):
        ### TIME RELATED DATA
        self.T = 120.000              # Total simulation time
        self.dt = 0.001              # Simulation step size
        self.framerate = 333
        self.t = 0.0                 # Current time
        self.t_aux = 0.0             # For trajectory
        self.time_stamps = []        # Time stamps for plot

        ### TRAJECTORY DATA
        # Spiral
        self.radius = 4.0
        self.center = np.array([0.0, 0.0, 1.0])
        self.angular_velocity = 1.0
        self.climb_velocity = 0.0

        ### AGENT DATA
        # Agent parameters
        self.state = "INIT"                  # HOVER - NEXT - PASS
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
        self.FOV = PI / 4.0                           # Half camera field of view
        self.camera_axis = np.array([1.0, 0.0, 0.0])  # Camera axis in body frame

        ### GATES
        self.gate_number = 0
        gate_size = 2.5
        map_scale = 5.0
        self.gates = []
        self.regions = []
        counter = 0
        xspan = 200.0
        cscale = 0.8

        self.bearing_des = []
        self.bearing_des.append( np.array([1.0, 1.0, 1.0]))
        self.bearing_des.append( np.array([1.0, -1.0, 1.0]))
        self.bearing_des.append( np.array([1.0, 1.0, -1.0]))
        self.bearing_des.append( np.array([1.0, -1.0, -1.0]))

        # Add 1st gate
        gate = {}
        gate['features'] = []
        center = map_scale * np.array([3.0, 0.0, 0.0])
        rotation = angleToRotationMatrix(0.0, 0.0, 0.0)
        gate['features'].extend(self.create_gate(center, gate_size, rotation))
        gate['desired'] = []
        for b in self.bearing_des:
            gate['desired'].append(rotation @ b)
        self.gates.append(gate)

        self.regions.append(bezQP.PolygonalRegion())
        self.regions[counter].createBox(center, rotation, xspan, cscale*gate_size, cscale*gate_size)
        self.regions[counter].buildConstraintMatrices()
        counter += 1

        # Add 2nd gate
        gate = {}
        gate['features'] = []
        center = map_scale * np.array([8.0, -2.0, 0.0])
        rotation = angleToRotationMatrix(0.0, 0.0, -PI/4)
        gate['features'].extend(self.create_gate(center, gate_size, rotation))
        gate['desired'] = []
        for b in self.bearing_des:
            gate['desired'].append(rotation @ b)
        self.gates.append(gate)

        self.regions.append(bezQP.PolygonalRegion())
        self.regions[counter].createBox(center, rotation, xspan, cscale*gate_size, cscale*gate_size)
        self.regions[counter].buildConstraintMatrices()
        counter += 1

        # Add 3nd gate
        #gate = {}
        #gate['features'] = []
        #center = map_scale * np.array([8.0, -3.0, 1.0])
        #rotation = angleToRotationMatrix(0.0, 0.0, -PI/3)
        #gate['features'].extend(self.create_gate(center, gate_size, rotation))
        #gate['desired'] = []
        #for b in self.bearing_des:
            #gate['desired'].append(rotation @ b)
        #self.gates.append(gate)

        # Add 4th gate
        #gate = {}
        #gate['features'] = []
        #center = map_scale * np.array([9.0, -6.0, 0.0])
        #rotation = angleToRotationMatrix(0.0, 0.0, -PI/2)
        #gate['features'].extend(self.create_gate(center, gate_size, rotation))
        #gate['desired'] = []
        #for b in self.bearing_des:
            #gate['desired'].append(rotation @ b)
        #self.gates.append(gate)

        # Add 5th gate
        gate = {}
        gate['features'] = []
        center = map_scale * np.array([12.0, -8.0, 0.0])
        rotation = angleToRotationMatrix(0.0, 0.0, -PI/2)
        gate['features'].extend(self.create_gate(center, gate_size, rotation))
        gate['desired'] = []
        for b in self.bearing_des:
            gate['desired'].append(rotation @ b)
        self.gates.append(gate)

        self.regions.append(bezQP.PolygonalRegion())
        self.regions[counter].createBox(center, rotation, xspan, cscale*gate_size, cscale*gate_size)
        self.regions[counter].buildConstraintMatrices()
        counter += 1

        # Add 6th gate
        #gate = {}
        #gate['features'] = []
        #center = map_scale * np.array([7.0, -10.0, 0.0])
        #rotation = angleToRotationMatrix(0.0, 0.0, -2 * PI/3)
        #gate['features'].extend(self.create_gate(center, gate_size, rotation))
        #gate['desired'] = []
        #for b in self.bearing_des:
            #gate['desired'].append(rotation @ b)
        #self.gates.append(gate)

        # Add 7th gate
        gate = {}
        gate['features'] = []
        center = map_scale * np.array([5.0, -15.0, 0.0])
        rotation = angleToRotationMatrix(0.0, 0.0, 1*PI/6)
        gate['features'].extend(self.create_gate(center, gate_size, rotation))
        gate['desired'] = []
        for b in self.bearing_des:
            gate['desired'].append(rotation @ b)
        self.gates.append(gate)

        self.regions.append(bezQP.PolygonalRegion())
        self.regions[counter].createBox(center, rotation, xspan, cscale*gate_size, cscale*gate_size)
        self.regions[counter].buildConstraintMatrices()
        counter += 1

        # Add 8th gate
        #gate = {}
        #gate['features'] = []
        #center = map_scale * np.array([1.0, -7.0, 0.0])
        #rotation = angleToRotationMatrix(0.0, 0.0, 0.0)
        #gate['features'].extend(self.create_gate(center, gate_size, rotation))
        #gate['desired'] = []
        #for b in self.bearing_des:
            #gate['desired'].append(rotation @ b)
        #self.gates.append(gate)

        # Add 9th gate
        gate = {}
        gate['features'] = []
        center = map_scale * np.array([-2.0, -5.0, 0.0])
        rotation = angleToRotationMatrix(0.0, 0.0, 2*PI/3)
        gate['features'].extend(self.create_gate(center, gate_size, rotation))
        gate['desired'] = []
        for b in self.bearing_des:
            gate['desired'].append(rotation @ b)
        self.gates.append(gate)

        self.regions.append(bezQP.PolygonalRegion())
        self.regions[counter].createBox(center, rotation, xspan, cscale*gate_size, cscale*gate_size)
        self.regions[counter].buildConstraintMatrices()
        counter += 1

        # Add 10th gate
        #gate = {}
        #gate['features'] = []
        #center = map_scale * np.array([-2.0, -2.0, 0.0])
        #rotation = angleToRotationMatrix(0.0, 0.0, PI/3)
        #gate['features'].extend(self.create_gate(center, gate_size, rotation))
        #gate['desired'] = []
        #for b in self.bearing_des:
            #gate['desired'].append(rotation @ b)
        #self.gates.append(gate)

        # Loop back
        center = map_scale * np.array([3.0, 0.0, 0.0])
        rotation = angleToRotationMatrix(0.0, 0.0, 0.0)
        self.regions.append(bezQP.PolygonalRegion())
        self.regions[counter].createBox(center, rotation, xspan, cscale*gate_size, cscale*gate_size)
        self.regions[counter].buildConstraintMatrices()
        counter += 1

        ### FEATURES
        self.features = []
        self.features = self.gates[0]["features"].copy()
        self.bearing_des = self.gates[0]["desired"].copy()

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

        ### CURVE
        self.curves = []
        order = 7
        derivative = 4
        bz = bezQP.BezierQuadraticPlanner(order, derivative)
        for region in self.regions:
            bz.addRegion(region)
            self.curves.append(Bezier(20.0))

        p0 = self.p.copy()
        v0 = np.zeros((3,))
        a0 = np.zeros((3,))
        j0 = np.zeros((3,))
        initial_conditions = [p0, v0, a0, j0]

        pf = self.p.copy()
        vf = np.zeros((3,))
        af = np.zeros((3,))
        jf = np.zeros((3,))
        final_conditions = [pf, vf, af, jf]

        conditions = [initial_conditions, final_conditions]
        bz.addContinuityConditions(conditions)
        bz.createPath()

        for i in range(bz.regions_number):
            for j in range(order+1):
                index = i*3*(order+1) + j*3
                self.curves[i].addControlPoint(bz.solution[index:index+3])

    def create_gate(self, center, size, rotation):
        gate = []
        gate.append(center + size * rotation @ np.array([0.0, np.cos(PI/4), np.sin(PI/4)]))
        gate.append(center + size * rotation @ np.array([0.0, -np.cos(PI/4), np.sin(PI/4)]))
        gate.append(center + size * rotation @ np.array([0.0, -np.cos(PI/4), -np.sin(PI/4)]))
        gate.append(center + size * rotation @ np.array([0.0, np.cos(PI/4), -np.sin(PI/4)]))
        return gate.copy()
    
    def drawGates(self):
        for gate in self.gates:
            for f in gate['features']:
                self.animation_axis.plot(f[0], f[1], f[2], 'xr')

            for i in range(len(gate['features'])):
                f1 = gate['features'][i]
                f2 = gate['features'][(i+1)%4]
                self.animation_axis.plot([f1[0], f2[0]],
                                         [f1[1], f2[1]],
                                         '-k',
                                         zs=[f1[2], f2[2]],
                                         linewidth=2)

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
            self.compute_distance()
            self.stateMachine()
            self.differentialFlatness()
            self.controller()
            self.CBF()

            # Update state
            self.dynamics()

            # Snap frame for animation
            if counter == self.framerate:
                counter = 0
                self.addFrame()

    def animation(self):
        print("Creating animation")
        animation = self.camera.animate()
        #animation.save('quadrotor_CBF.mp4', fps=30, codec='libx264')
        plt.show()
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


        drawQuadrotor(self.animation_axis, self.p, self.R, 0.6)
        #self.animation_axis.plot(self.p_des[0], self.p_des[1], self.p_des[2], 'og')

        self.drawGates()

        drawHalfCone(self.animation_axis, self.p,
                     self.FOV, R=self.R, plot_type='circle')

        for curve in self.curves:
            curve.plot(self.animation_axis)
        #for region in self.regions:
            #region.plot(self.animation_axis)

        self.animation_axis.set_title("Agent FOV")
        self.animation_axis.set_xlabel("x (m)")
        self.animation_axis.set_ylabel("y (m)")
        self.animation_axis.set_zlabel("z (m)")
        scale = 5.0
        self.animation_axis.set_xlim([scale*-5.0, scale*12.0])
        self.animation_axis.set_ylim([scale*-15.0, scale*5.0])
        self.animation_axis.set_zlim([scale*-5.0, scale*5.0])

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

    def stateMachine(self):
        if self.state == "INIT":
            if True:
                self.state = "NEXT"
        if self.state == "NEXT":
            if self.d < 4.0:
                self.state = "NEXT"
                self.gate_number = self.gate_number + 1
                if (self.gate_number >= len(self.gates)):
                    self.gate_number = 0

                self.bearing_des = self.gates[self.gate_number]["desired"].copy()
                self.features = self.gates[self.gate_number]["features"].copy()
                self.old_u = []
                for ff in self.features:
                    u = ff - self.p
                    d = np.linalg.norm(u)
                    u = u / d
                    self.old_u.append(u.copy())

                self.timer = (self.d + 2.0) / np.linalg.norm(self.v)
                self.v_des_tmp = self.v.copy()

        if self.state == "PASS":
            self.v_des = self.v_des_tmp.copy()
            self.timer -= self.dt
            if self.timer < 0.0:
                self.state = "NEXT"

    def trajectory(self):
        self.t_aux = self.t + self.dt
        if (self.t_aux > 20.0 * len(self.curves)):
            self.t_aux = 0.0

        j = -1
        t_curve = self.t_aux
        while t_curve > 0.0:
            t_curve -= 20.0
            j += 1
        t_curve += 20.0

        self.p_des = self.curves[j].evaluate(t_curve)

        self.v_des = self.curves[j].evaluate_derivative(t_curve, 1)

        self.a_des = self.curves[j].evaluate_derivative(t_curve, 2)

        self.j_des = self.curves[j].evaluate_derivative(t_curve, 3)

        self.s_des = self.curves[j].evaluate_derivative(t_curve, 4)

        self.yaw_des   = 0.0
        roll, pitch, yaw = rotationMatrixToAngle(self.R)
        self.yaw_des = yaw

        self.dyaw_des  = 0.0
        M = eulerRatesToBodyRatesMatrix(roll, pitch, yaw)
        self.dyaw_des = (np.linalg.inv(M) @ self.w)[2]

        self.ddyaw_des = 0.0

    def compute_distance(self):
        distance = 0.0
        avg = np.zeros((3,))
        for f in self.features:
            avg += f
        avg = 0.25 * avg
        self.d = np.linalg.norm(self.p - avg)

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
        e_p = 0.0
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
        inequality_A = np.zeros((2*len(self.features) + 1, 1 + DIM + len(self.features)))
        inequality_B = np.zeros((2*len(self.features) + 1, ))

        input_nom = np.zeros((1 + DIM + len(self.features)))
        input_nom[0:4] = self.u.copy()
        input_nom[1:4] = np.linalg.inv(self.J) @ (self.u[1:4] - hatMap(self.w) @ (self.J @ self.w)) 
        self.norm_alpha_nom.append(input_nom[1:4])

        ### CBF
        gamma1 = 5.0
        gamma2 = 3.0

        k1 = gamma2 + gamma1
        k2 = gamma2 * gamma1

        min_h = 1.0
        for i in range(len(self.features)):
            ### BEARING
            u = self.features[i] - self.p
            norm = np.linalg.norm(u)
            u = u / norm
            du = (u - self.old_u[i]) / self.dt
            self.old_u[i] = u.copy()

            # Barrier function quantities
            h = u @ z - cos(self.FOV)
            if h < min_h:
                min_h = h

            #assert h > -0.5
            grad_p = (-z + (u @ z) * u) / norm
            grad_R = -(self.R @ hatMap(self.camera_axis)).T @ u 
            dh = (grad_p @ self.v + grad_R @ self.w)
            P = np.eye(DIM) - u.reshape((DIM,1)) @ u.reshape((1, DIM))

            hess_RR = hatMap(self.R.T @ u) @ hatMap(self.camera_axis)
            hess_p_psi_v = -(self.R @ hatMap(self.camera_axis)).T @ du 

            hess_p_p = 3*(z@u)*(-P) + (-u.reshape((DIM,1)) @ z.reshape((1,DIM)) + 2*z@u*np.eye(DIM) - z.reshape((DIM,1)) @ u.reshape((1,DIM)))
            hess_p_p = hess_p_p / (norm ** 2)

            inequality_A[i, 0] = (1 / self.m) * grad_p @ (self.R[:, 2].reshape((DIM,)))
            inequality_A[i, 1:4] = -(self.R @ hatMap(self.camera_axis)).T @ u 
            inequality_A[i, 1 + DIM + i] = 1.0
            inequality_A[len(self.features) + i, 1 + DIM + i] = 1.0

            inequality_B[i] = (hess_RR @ self.w) @ self.w + \
                              2 * hess_p_psi_v @ self.w + \
                              (hess_p_p @ self.v) @ self.v + \
                              k2 * dh + k1 * h + \
                              grad_p @ self.g

        # Positive trust constraint
        inequality_A[-1,0] = 1.0
        inequality_B[-1] = 0.0

        ### QP COST
        H = 100*np.eye(1 + DIM + len(self.features))
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,2] = 1.0
        H[3,3] = 1.0
        qp_solution = solve_qp(H, -H@input_nom, -inequality_A, inequality_B, solver='gurobi')
        self.u[0] = qp_solution[0]
        self.alpha = qp_solution[1:4]
        self.norm_alpha.append(self.alpha.copy())
        self.u[1:4] = self.J @ self.alpha + np.cross(self.w, self.J @ self.w)
        self.plot_h.append(min_h)

    def plot(self):
        #fig, ax = plt.subplots()
        #ax.plot(self.time_stamps, self.norm_ep)
        #ax.set_title("Position error")
        #ax.legend(["x", "y", "z"])

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

        fig, ax = plt.subplots()
        ax.plot(self.time_stamps, self.plot_h)
        ax.set_title("Barrier")

        plt.show()

def main():
    simulation = Simulation()
    simulation.initialize()
    simulation.run()
    simulation.animation()
    simulation.plot()

if __name__ == '__main__':
    main()
