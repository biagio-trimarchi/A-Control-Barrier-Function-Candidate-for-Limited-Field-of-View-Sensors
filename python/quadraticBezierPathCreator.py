# Libraries
import math as mt                       # Mathematical library
import numpy as np                      # Linear algebra library
from qpsolvers import solve_qp          # Quadratic program library
import matplotlib.pyplot as plt         # Plot and graph library
from mpl_toolkits import mplot3d        # 3D plotting utilities
from scipy.linalg import block_diag     # Create diagonal block matrix from given matrices

import sys
np.set_printoptions(threshold=sys.maxsize)

class PolygonalRegion:
    def __init__(self):
        self.faces = []
        pass

    def addFace(self, face):
        # A face must be a convex polygon
        self.faces.append(face)

    def createBox(self, center, R, xsize, ysize, zsize):
        V0 = center + R @ np.array([-xsize/2.0, -ysize/2.0, -zsize/2.0])
        V1 = center + R @ np.array([-xsize/2.0,  ysize/2.0, -zsize/2.0])
        V2 = center + R @ np.array([ xsize/2.0,  ysize/2.0, -zsize/2.0])
        V3 = center + R @ np.array([ xsize/2.0, -ysize/2.0, -zsize/2.0])

        V4 = center + R @ np.array([-xsize/2.0, -ysize/2.0, zsize/2.0])
        V5 = center + R @ np.array([-xsize/2.0,  ysize/2.0, zsize/2.0])
        V6 = center + R @ np.array([ xsize/2.0,  ysize/2.0, zsize/2.0])
        V7 = center + R @ np.array([ xsize/2.0, -ysize/2.0, zsize/2.0])

        faces = []
        faces.append([V0, V1, V2, V3])
        faces.append([V0, V4, V7, V3])
        faces.append([V3, V7, V6, V2])
        faces.append([V1, V2, V6, V5])
        faces.append([V0, V1, V5, V4])
        faces.append([V4, V5, V6, V7])

        for face in faces:
            self.addFace(face)

    def createAxisAlignedBox(self, lower_left_vertex, xsize, ysize, zsize):
        V0 = lower_left_vertex + np.array([0.0, 0.0, 0.0])
        V1 = lower_left_vertex + np.array([0.0, ysize, 0.0])
        V2 = lower_left_vertex + np.array([xsize, ysize, 0.0])
        V3 = lower_left_vertex + np.array([xsize, 0.0, 0.0])

        V4 = lower_left_vertex + np.array([0.0, 0.0, zsize])
        V5 = lower_left_vertex + np.array([0.0, ysize, zsize])
        V6 = lower_left_vertex + np.array([xsize, ysize, zsize])
        V7 = lower_left_vertex + np.array([xsize, 0.0, zsize])
        
        faces = []
        faces.append([V0, V1, V2, V3])
        faces.append([V0, V4, V7, V3])
        faces.append([V3, V7, V6, V2])
        faces.append([V1, V2, V6, V5])
        faces.append([V0, V1, V5, V4])
        faces.append([V4, V5, V6, V7])

        for face in faces:
            self.addFace(face)

    def buildConstraintMatrices(self):
        # Ax <= b, a line for each face
        self.A = np.zeros((len(self.faces), 3)) 
        self.b = np.zeros((len(self.faces),  )) 
    
        # Compute centroid
        centroid = np.zeros((3, ))
        vertices = 0
        for face in self.faces:
            for vertex in face:
                vertices += 1
                centroid += vertex
        centroid = (1.0 / vertices) * centroid

        # Find outer normal and add it as constraint
        i = 0
        for face in self.faces:
            normal = np.cross(face[2] - face[0], face[1] - face[0])
            normal = normal / np.linalg.norm(normal)
            if (normal @ (centroid - face[0])) > 0.0:
                normal = -normal

            # n @ (x - V_0) <= 0
            self.A[i, :] = normal
            self.b[i]    = normal @ face[0]
            i += 1

    def getConstraintMatrices(self):
        return self.A.copy(), self.b.copy()

    def plot(self, ax, 
             xrange=np.linspace(-100.0, 100.0, 51), 
             yrange=np.linspace(-100.0, 100.0, 51),
             zrange=np.linspace(-10.0, 10.0, 31)):

        for x in xrange:
            for y in yrange:
                for z in zrange:
                    p = np.array([x, y, z])
                    if np.all(self.b > self.A @ p):
                        ax.scatter(p[0], p[1], p[2], 'b')

class BezierQuadraticPlanner:
    def __init__(self, order, shape_parameter):
        self.regions = []
        self.order = order
        self.shape_parameter = shape_parameter
        self.d = 3
        self.regions_number = 0
        self.vMax = 100.0
        self.aMax = 100.0

        #self.A_ineq = np.array([])
        #self.b_ineq = np.array([])
        #self.A_eq = np.array([])
        #self.b_eq = np.array([])

    def createPath(self):
        if self.regions_number < 1:
            print("No regions provided, optimization failed")
        
        self.addRegionConstraints()
        self.addDynamicConstraints(self.vMax, self.aMax)
        #addContinuityConditions()

        self.createCostMatrix()

        opt_var_num = self.d * (self.order+1) * self.regions_number
        aux = np.zeros((opt_var_num,))
        self.solution = solve_qp(self.H, aux, self.A_ineq, self.b_ineq, self.A_eq, self.b_eq, solver="quadprog")

    def addRegion(self, region):
        self.regions.append(region)
        self.regions_number += 1

    def addRegionConstraints(self):
        n = self.order
        I = np.eye(n+1)
        one = np.ones((n+1,))
        A_regions = np.array([])
        b_regions = np.array([])
        for region in self.regions:
            region.buildConstraintMatrices()
            A, b = region.getConstraintMatrices()
            A = np.kron(I, A)
            b = np.kron(one, b)

            A_regions = block_diag(A_regions, A)
            b_regions = np.hstack((b_regions, b))

        A_regions = A_regions[1:, :] # Remove first row (an artifact created by block_diag)

        # Stack on to A, b
        self.A_ineq = A_regions
        self.b_ineq = b_regions

    def addContinuityConditions(self, conditions):
        # Add continuity conditions up to order n+1 // 2 
        n = self.order
        d = self.d
        opt_var_num = d * (n+1) * self.regions_number
        continuity_order = (n-1) // 2

        # Initial and final constraints
        initial_conditions = conditions[0]
        final_conditions = conditions[1]

        A_initial = np.zeros((d * (continuity_order + 1), opt_var_num))
        b_initial = np.zeros((d * (continuity_order + 1),))
        A_final = np.zeros((d * (continuity_order + 1), opt_var_num))
        b_final = np.zeros((d * (continuity_order + 1),))
        for i in range(continuity_order+1):
            D = self.computeD(i)
            mask1 = np.zeros((1, n+1-i))
            mask1[0, 0] = 1.0
            mask1 = np.kron(mask1, np.eye(d))
            A_initial[d*i:d*i+d, 0:(n+1)*d] = mask1 @ D

            mask2 = np.zeros((1, n+1-i))
            mask2[0, -1] = 1.0
            mask2 = np.kron(mask2, np.eye(d))
            A_final[d*i:d*i+d, -(n+1)*d:] = mask2 @ D

        for i in range(continuity_order+1):
            b_initial[d*i:d*i+d] = initial_conditions[i]
            b_final[d*i:d*i+d] = final_conditions[i]

        # Continuity constraints
        constraint_num = d * (continuity_order + 1) * (self.regions_number-1)
        A_connect = np.zeros((constraint_num, opt_var_num))
        b_connect = np.zeros((constraint_num,))
        for i in range(0, continuity_order+1):
            # i = 0 position, 1 velocity, 2 acceleration, 3 jerk, ...
            D = self.computeD(i)

            mask1 = np.zeros((1, n+1-i))
            mask1[0, -1] = 1.0
            mask1 = np.kron(mask1, np.eye(d))

            mask2 = np.zeros((1, n+1-i))
            mask2[0, 0] = -1.0
            mask2 = np.kron(mask2, np.eye(d))

            D1 = mask1 @ D
            D2 = mask2 @ D

            for j in range(0, self.regions_number-1):
                index = (i*(self.regions_number-1)*d)+(j*d)
                A_connect[index:index+d, :] = np.hstack((np.zeros((d, (n+1)*d*j)), D1, D2, np.zeros((d, (n+1)*d*(self.regions_number - (j+2))))))

        self.A_eq = np.vstack((A_initial, A_final, A_connect))
        self.b_eq = np.hstack((b_initial, b_final, b_connect))

    def addDynamicConstraints(self, vMax, aMax):
        # This includes only velocity and acceleration constraints
        n = self.order
        d = self.d
        opt_var_num = d * (n+1) * self.regions_number

        D_vel = self.computeD(1)
        mask = np.vstack((np.eye(self.regions_number), -np.eye(self.regions_number)))
        A_vel = np.kron(D_vel, mask)
        b_vel = vMax * np.ones((2*d*n*self.regions_number, ))

        D_acc = self.computeD(2)
        A_acc = np.kron(D_acc, mask)
        b_acc = aMax * np.ones((2*d*(n-1)*self.regions_number, ))

        self.A_ineq = np.vstack((self.A_ineq, A_vel, A_acc))
        self.b_ineq = np.hstack((self.b_ineq, b_vel, b_acc))

    def computeD(self, r):
        n = self.order
        D = np.eye(n + 1)
        I = np.eye(self.d)

        for j in range(0, r):
            Daux = D.copy()
            D = np.zeros((n - j, n + 1 - j))
            for k in range(0, n - j):
                D[k, k] = -1.0 #* (n-j)
                D[k, k+1] = 1.0 #* (n-j)
            D = D @ Daux
        
        D = np.kron(D, I)
        return D

    def createCostMatrix(self):
        n = self.order
        r = self.shape_parameter
        D = self.computeD(r)
        I = np.eye(self.d)
        self.H = np.zeros((n-r+1, n-r+1))

        factor = n
        for i in range(n-r+1, n):
            factor *= i

        for i in range(0, n-r+1):
            for j in range(0, n-r+1):
                factor = (mt.comb(n-r, i) * mt.comb(n-r, j)) / (mt.comb(2*(n-r), i+j))
                factor = factor / (2 * (n-r) + 1)
                self.H[i, j] = factor

        self.H = np.kron(self.H, I)
        self.H = D.T @ self.H @ D
        self.H = (self.H.T @ self.H) / 2.0

        I = np.eye(self.regions_number)
        self.H = np.kron(self.H, I)

        # Avoid numerical issues
        self.H = self.H + 0.001 * np.eye(self.H.shape[0])

# DEBUG
if __name__ == '__main__':
    polygon = PolygonalRegion()
    V0 = np.array([0.0, 0.0, 0.0])
    polygon.createAxisAlignedBox(V0, 5.0, 4.0, 2.0)
    polygon.buildConstraintMatrices()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim([-1.0, 7.0])
    ax.set_ylim([-1.0, 7.0])
    ax.set_zlim([-3.0, 6.0])
    #polygon.plot(ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


    region1 = PolygonalRegion()
    region1.createAxisAlignedBox(np.array([-1.0, -1.0, -1.0]), 5.0, 3.0, 2.0)
    region2 = PolygonalRegion()
    region2.createAxisAlignedBox(np.array([2.0, 0.0, -0.5]), 5.0, 2.0, 2.0)
    region3 = PolygonalRegion()
    region3.createAxisAlignedBox(np.array([5.0, 0.0, 0.0]), 5.0, 2.0, 2.0)

    #plt.show()
    
    order = 7
    derivative = 4
    bz = BezierQuadraticPlanner(order, derivative)
    bz.addRegion(region1)
    bz.addRegion(region2)
    bz.addRegion(region3)

    conditions = []
    # Initial conditions
    p0 = np.zeros((3,))
    v0 = np.zeros((3,))
    a0 = np.zeros((3,))
    j0 = np.zeros((3,))
    initial_conditions = [p0, v0, a0, j0]

    pf = np.array([8.0, 1.0, 1.0])
    vf = np.zeros((3,))
    af = np.zeros((3,))
    jf = np.zeros((3,))
    final_conditions = [pf, vf, af, jf]

    conditions = [initial_conditions, final_conditions]
    bz.addContinuityConditions(conditions)

    bz.createPath()
    
    from utilities import *

    curves = []
    curves.append(Bezier(5.0))
    curves.append(Bezier(5.0))
    curves.append(Bezier(5.0))

    for i in range(bz.regions_number):
        for j in range(order+1):
            index = i*3*(order+1) + j*3
            curves[i].addControlPoint(bz.solution[index:index+3])

    for curve in curves:
        curve.plot(ax)

    plt.show()
