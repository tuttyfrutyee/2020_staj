import math
import numpy as np
import bisect

import matplotlib.pyplot as plt

class Spline:
    u"""
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        u"""
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        u"""
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        u"""
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        u"""
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        u"""
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        u"""
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)
        return B


class Spline2D:
    u"""
    2D Cubic Spline class

    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        u"""
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        u"""
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        u"""
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1): #ds parca sayisini belirler
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s




def addNoiseToPaths(objectPaths_, std, noiseMode):
    
    
    noisyObjectPaths = []
    
    for objectPath_ in objectPaths_:
        
        noisyObjectPath = np.copy(objectPath_)
        
        for i,point in enumerate(noisyObjectPath[1:-1, :]):
            
            if(noiseMode == "orthogonalNoise"):
            
                m = -1.0/((noisyObjectPath[i+1][1] - noisyObjectPath[i-1][1]) / (noisyObjectPath[i+1][0] - noisyObjectPath[i-1][0]))
                unitVector = np.array([ np.cos(np.arctan(m)), np.sin(np.arctan(m)) ])
                
                noisyObjectPath[i] = noisyObjectPath[i] + unitVector * np.random.normal(0, std)
                
            elif (noiseMode == "uniform"):
                
                noisyObjectPath[i][0] += np.random.normal(0, std)
                noisyObjectPath[i][1] += np.random.normal(0, std)
        
        noisyObjectPaths.append(noisyObjectPath)
            
    return noisyObjectPaths
    
    
def generateObjectPaths(objectCount, std, pointDistance, frame, pathInnerPointCount, noiseMode="uniform", points=None, random=True, plot=False):
    
    noisyObjectPaths = []
    originalObjectPaths = []
    
    maxW = frame["w"]
    maxH = frame["h"]
    
    
    
    if(random):
        
        for i in range(objectCount):
            
            x = np.array([])
            y = np.array([])
            for k in range(pathInnerPointCount + 2):
                x = np.append(x, math.floor(np.random.rand() * (maxW-1) * 2 - maxW))
                y = np.append(y, math.floor(np.random.rand() * (maxH-1) * 2 - maxH))
            
            x = x.reshape((pathInnerPointCount + 2,1))
            y = y.reshape((pathInnerPointCount + 2,1))
            
            if(i == 0):
                
                points = [np.concatenate((x,y), axis=1)]
                
            else:
                
                points.append(np.concatenate((x,y), axis = 1))
                          
                        
                          
        points = np.array(points).reshape(len(points), pathInnerPointCount + 2, 2)
    
        print("points shape : ", points.shape)
                
        
    for i,point in enumerate(points):
    
        x = point[:,0].tolist()
        y = point[:,1].tolist()
        sp = Spline2D(x, y)
        s = np.arange(0, sp.s[-1], pointDistance)
        
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))
        
        
        if(plot):
            flg, ax = plt.subplots(i)
            plt.plot(x, y, "D", label="input")
            plt.plot(rx, ry, "-r", label="spline") #rx ve ry senin noktalarin
            plt.grid(True)
            plt.axis("equal")
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend()
            plt.show()
            
        rx = np.array(rx).reshape((len(rx),1))
        ry = np.array(ry).reshape((len(rx),1))
        
        if(i == 0):
            originalObjectPaths = [np.concatenate((rx,ry), axis = 1)]
        
        else:
            
            originalObjectPaths.append(np.concatenate((rx,ry), axis=1))
            
        
            
    noisyObjectPaths = addNoiseToPaths(originalObjectPaths, std, noiseMode)
        
    
    return (originalObjectPaths, noisyObjectPaths)





def main():
    
    print("Spline 2D test")
    
    # way points
    """
    x = [-20.04, -45.04, -50.04, -48.04, -17.31,  \
          -1.31, 20.14, 39.03, 49.8, 50.04,       \
          30.19, 24.19, -20.04, -45.04, -57.04,    \
          -56.04, -42.04, -17.04, 18.35, 24.35]
    
    y = [3.3, 3.33, 15.0, 40.0, 50.4,          \
         50.4, 50.4, 50.4, 34.3, 9.54,         \
         3.33, 3.33, 3.33, 3.33, -15.0,        \
         -40.0, -54.8, -57.0, -57.0, -57.0]
    """
    
    x = [1,3,2,8]
    y = [5,8,3,9]     
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], 0.1)
    
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))
    
    flg, ax = plt.subplots(1)
    plt.plot(x, y, "D", label="input")
    plt.plot(rx, ry, "-r", label="spline") #rx ve ry senin noktalarin
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()
    plt.show()
    """
    flg, ax = plt.subplots(1)
    plt.plot(s, [math.degrees(iyaw) for iyaw in ryaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")
    
    flg, ax = plt.subplots(1)
    plt.plot(s, rk, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")
    """


