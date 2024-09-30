# Libraries
import math as mt
import numpy as np                      # Linear algebra library
from qpsolvers import solve_qp          # Quadratic program library
import matplotlib.pyplot as plt         # Plot and graph library
from mpl_toolkits import mplot3d        # 3D plotting utilities
from matplotlib.patches import Wedge    # To draw polygon
from tqdm import tqdm                   # Loading bar
from celluloid import Camera            # Animation library
from scipy.integrate import solve_ivp   # Solve initial value problems
plt.rcParams['text.usetex'] = True

### CONSTANTS AND LAMBDA
G = 9.8
DIM = 3
PI = np.pi
I = np.eye(DIM)
cos = lambda x: np.cos(x)
sin = lambda x: np.sin(x)
tan = lambda x: np.tan(x)

### UTILITY FUNCTIONS
# Rotations utilities
def xAxisRotation(theta):
    R = [[1.0,         0.0,         0.0], 
         [0.0,  cos(theta), -sin(theta)],
         [0.0,  sin(theta),  cos(theta)]]
    R = np.array(R)
    return R

def yAxisRotation(theta):
    R = [[ cos(theta), 0.0, sin(theta) ], 
         [        0.0, 1.0,         0.0],
         [-sin(theta), 0.0,  cos(theta)]]
    R = np.array(R)
    return R

def zAxisRotation(theta):
    R = [[cos(theta), -sin(theta), 0.0], 
         [sin(theta),  cos(theta), 0.0],
         [       0.0,         0.0, 1.0]]
    R = np.array(R)
    return R

def angleToRotationMatrix(phi, theta, psi):
    # Roll (phi x), pitch (theta y), yaw (psi z)
    # Rotation formula: R_z * R_x * R_y
    # Meaning: first rotation about z, then around x, then around y (?)
    R = zAxisRotation(psi) @ xAxisRotation(phi) @ yAxisRotation(theta)
    return R

def rotationMatrixToAngle(R):
    yaw = np.arctan2(-R[0,1], R[1,1])
    roll = np.arcsin(R[2,1])
    pitch = np.arctan2(-R[2,0], R[2,2]) 
    
    return roll, pitch, yaw

def eulerRatesToBodyRatesMatrix(phi, theta, psi):
    # Notice that is not invertible when phi = PI/2
    R = [
         [ cos(theta), 0.0, -cos(phi) * sin(theta)],
         [        0.0, 1.0,               sin(phi)],
         [ sin(theta), 0.0,  cos(phi) * cos(theta)]
        ]
    R = np.array(R)
    return R

def eulerAccelerationToBodyAcceleration(phi, theta, psi, dphi, dtheta, dpsi, ddphi, ddtheta, ddpsi):
    R = eulerRatesToBodyRatesMatrix(phi, theta, psi)
    eulerRates = np.array([dphi, dtheta, dpsi])
    eulerAccelerations = np.array([ddphi, ddtheta, ddpsi])

    dR = [
          [ -sin(theta) * dtheta, 0.0,  sin(phi) * sin(theta) * dphi - cos(phi) * cos(theta) * dtheta],
          [                  0.0, 0.0,                                                           0.0],
          [  cos(theta) * dtheta, 0.0, -sin(phi) * cos(theta) * dphi - cos(phi) * sin(theta) * dtheta]
         ]
    dR = np.array(dR)

    return dR @ eulerRates + R @ eulerAccelerations

def hatMap(v):
    v_hat = [[  0.0, -v[2], v[1]],
             [ v[2],  0.0, -v[0]],
             [-v[1],  v[0],  0.0]]
    v_hat = np.array(v_hat)
    return v_hat

def veeMap(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def exponentialMap(u, theta):
    u_hat = hatMap(u)
    tmp = I.copy() + sin(theta) * u_hat + (1 - cos(theta) ** 2) * (u_hat @ u_hat)

    # Projection onto SO3
    U, S, Vh = np.linalg.svd(tmp)
    V = Vh.T
    return U.dot(np.diag([1, 1, np.linalg.det(U.dot(V.T))]).dot(V.T))

# Plotting utilities
def drawHalfCone(axis, vertex, aperture, R = I.copy(), height = 10.0, resolution = 0.5, plot_type="circle"):
    # Draw a half circular cone, default is with 
    # axis parallel to the x axis of the plot
    # Parameters
    #            axis       : axis on which to draw
    #            vertex     : vertex of the cone
    #            aperture   : half the aperture of the cone
    #            R          : rotation matrix to rotate the cone
    #            height     : height of the cone
    #            resolution : distance between line/circles
    #            plot_type  : type of plot ("line", "circle")

    if plot_type != "line" and plot_type != "circle":
        print("Unknown plot type, using default type: circle")
        plot_type = "circle"

    if plot_type == "circle":
        # Plot the cone as a series of circles
        circles = []                   # List to store circles
        depth = resolution             # Initialize depth 
        while depth < height:          # Loop from vertex to total height
            radius = depth * tan(aperture)                       # Radius of current circle
            current_circle = []                                  # List of points of the current circle
            for theta in np.arange(0, 2*PI+0.1, 0.1):                # Loop around the circle
                point = [depth, radius*cos(theta), radius*sin(theta)] # Compute point on circle along x axis
                point = R @ np.array(point)                           # Rotate point to align with desired orientation
                point = point + vertex                                # Add vertex coordinates
                current_circle.append(point)                          # Add point to circle
            circles.append(current_circle)                        # Add circle to list
            depth += resolution                                  # Update depth

        for circle in circles:
            axis.plot([point[0] for point in circle],
                      [point[1] for point in circle],
                      [point[2] for point in circle], 'b-', alpha=0.2)

    if plot_type == "line":
        end_points = []
        radius = height * tan(aperture)
        for theta in np.arange(0, 2*PI+0.1, resolution):
            end_point = [height, radius*cos(theta), radius*sin(theta)]
            end_point = R @ np.array(end_point)
            end_point = end_point + vertex
            end_points.append(end_point)

        for end_point in end_points:
            axis.plot([vertex[0], end_point[0]], 
                      [vertex[1], end_point[1]], 
                      'b-',
                      zs=[vertex[2], end_point[2]],
                      alpha=0.2)

def drawQuadrotor(axis, position, orientation, arm_lenght):
    ### Draw quadrotor with X configuration
    linewidth = 3

    ### Define auxiliary variables
    # Forward right arm end point
    rotor_FR_position     = np.zeros((3,))
    rotor_FR_position[0] = arm_lenght * cos(PI/4)
    rotor_FR_position[1] = -arm_lenght * sin(PI/4)
    rotor_FR_position = orientation @ rotor_FR_position
    rotor_FR_position += position

    # Forward left arm end point
    rotor_FL_position     = np.zeros((3,))
    rotor_FL_position[0] += arm_lenght * cos(PI/4)
    rotor_FL_position[1] += arm_lenght * sin(PI/4)
    rotor_FL_position = orientation @ rotor_FL_position
    rotor_FL_position += position

    # Backward right arm end point
    rotor_BR_position     = np.zeros((3,))
    rotor_BR_position[0] -= arm_lenght * cos(PI/4)
    rotor_BR_position[1] -= arm_lenght * sin(PI/4)
    rotor_BR_position = orientation @ rotor_BR_position
    rotor_BR_position += position

    # Backward left arm end point
    rotor_BL_position     = np.zeros((3,))
    rotor_BL_position[0] -= arm_lenght * cos(PI/4)
    rotor_BL_position[1] += arm_lenght * sin(PI/4)
    rotor_BL_position = orientation @ rotor_BL_position
    rotor_BL_position += position

    ### Plot
    # Plot forward right arm
    axis.plot([position[0], rotor_FR_position[0]],
              [position[1], rotor_FR_position[1]],
              'b-',
              zs=[position[2], rotor_FR_position[2]],
              linewidth=linewidth)

    # Plot forward left arm
    axis.plot([position[0], rotor_FL_position[0]],
              [position[1], rotor_FL_position[1]],
              'b-',
              zs=[position[2], rotor_FL_position[2]],
              linewidth=linewidth)


    # Plot backward right arm
    axis.plot([position[0], rotor_BR_position[0]],
              [position[1], rotor_BR_position[1]],
              'b-',
              zs=[position[2], rotor_BR_position[2]],
              linewidth=linewidth)

    # Plot backward left arm
    axis.plot([position[0], rotor_BL_position[0]],
              [position[1], rotor_BL_position[1]],
              'b-',
              zs=[position[2], rotor_BL_position[2]],
              linewidth=linewidth)

    # Plot center of mass
    axis.plot(position[0], position[1], position[2], 'ob', markersize=5)

class Bezier:
    def __init__(self, time):
        self.points = []
        self.totalTime = time
        self.degree = -1

    def addControlPoint(self, P):
        self.points.append(P)
        self.degree += 1

    def evaluate(self, t):
        t = t/self.totalTime
        total = np.zeros((3,))
        for k in range(self.degree + 1):
            total += self.points[k] * mt.comb(self.degree, k) * (t ** k) * ((1 - t) ** (self.degree - k))
        return total

    def evaluate_derivative(self, t, order):
        derivative_points = self.points.copy()
        count = 0
        degree = self.degree

        while count < order:
            for k in range(degree - count):
                derivative_points[k] = (degree - count) * (derivative_points[k+1] - derivative_points[k]) / self.totalTime
            derivative_points.pop(-1)
            count += 1

        n = degree - order
        total = 0.0
        t = t/self.totalTime
        for k in range(n+1): 
            total += derivative_points[k] * mt.comb(n, k) * (t ** k) * ((1 - t) ** (n - k))

        return total

    def plot(self, ax):
        samples = 100
        curve = np.zeros((3, samples))
        for tt in range(samples):
            t = (float(tt) / samples) * self.totalTime
            curve[:, tt] = self.evaluate(t)

        ax.plot3D(curve[0, :], curve[1, :], curve[2, :], 'r-', linewidth=2)

        for point in self.points:
            ax.plot3D(point[0], point[1], point[2], 'go', linewidth=2)
