# import the necessary packages
import numpy as np
import math, cv2

# Class: SBA             
def euclidean2D(pointA, pointB):
    return np.linalg.norm(np.array(pointA) - np.array(pointB))

def triangleArea(tri):
    x1, y1, x2, y2, x3, y3 = tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1]
    area = abs(0.5 * (((int(x2)-int(x1))*(int(y3)-int(y1)))-((int(x3)-int(x1))*(int(y2)-int(y1)))))
    return area

class SBA:
    def reprocessing(self):
        mask = cv2.threshold(self.input,127,255,cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=1)
        return mask
        
    def getContour(self):
        return max(cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0], key = cv2.contourArea)

    def getCentroid(self):
        try:
            M = cv2.moments(self.contour)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        except: 
            cx, cy = 0, 0
        return np.array([int(cx), int(cy)])

    def getCorners(self):
            # compute the first_vertex
            euclideans = []
            for point in self.contour:
                pointA = self.centroid
                pointB = point[0]
                euclideans.append(euclidean2D(pointA, pointB))
            first_vertex = self.contour[np.argsort(np.asarray(euclideans))][-1:][0][0]

            # compute the second_vertex
            # compute the furthest vertex from first_vertex 
            euclideans = []
            for point in self.contour:
                pointA = first_vertex
                pointB = point[0]
                euclideans.append(euclidean2D(pointA, pointB))
            second_vertex = self.contour[np.argsort(np.asarray(euclideans))][-1:][0][0]

            # compute the third vertex
            # compute the furthest vertex from first_vertex, second_vertex
            triareas = []
            for point in self.contour:
                tri = [first_vertex, second_vertex, point[0]]
                triareas.append(triangleArea(tri))
            triareas = np.asarray(triareas).astype(np.int64)
            sort = self.contour[np.argsort(triareas)]
            third_vertex = sort[-1:][0][0]

            # compute the fourth_vertex
            # Chose vertex that maximise the area 
            triareas = []
            for point in self.contour:
                Area1 = triangleArea([first_vertex, second_vertex, point[0]])
                Area2 = triangleArea([second_vertex, third_vertex, point[0]])
                Area3 = triangleArea([third_vertex, first_vertex, point[0]])
                triareas.append(Area1+Area2+Area3)
            triareas = np.asarray(triareas).astype(np.int64)
            sort = self.contour[np.argsort(triareas)]
            fourth_vertex = sort[-1:][0][0]

            vertice = np.asarray([first_vertex, second_vertex, third_vertex, fourth_vertex])
            vertice = vertice[np.argsort(vertice[:,0])]

            # anterior    
            A = [vertice[0], vertice[1]]
            A = np.asarray(A)
            A_y = [A[0][1], A[1][1]]
            A_y = np.asarray(A_y)
            A = A[np.argsort(A_y)]
            A1 = A[0]
            A2 = A[1]

            # posterior
            P = [vertice[2], vertice[3]]
            P = np.asarray(P)
            P_y = [P[0][1], P[1][1]]
            P_y = np.asarray(P_y)
            P = P[np.argsort(P_y)]
            P1 = P[0]
            P2 = P[1]
            
            if np.count_nonzero(np.asarray([A1, A2, P1, P2])[:,1]) < 4:
                self.log = np.append(self.log,"Vertice at border")
                self.valid = 0

            return A1, A2, P1, P2

    def getHeights(self):
        return euclidean2d(self.a1, self.a2), euclidean2d(self.p1, self.p2)
    
    def getLoss(self):
        return 1 - (np.min(np.asarray([self.ha, self.hp])) / np.max(np.asarray([self.ha, self.hp])))

    def getGrade(self):
        if self.l < 0.2: GRADE = 0
        elif self.l >= 0.2 and self.l < 0.25: GRADE = 1
        elif self.l >= 0.25 and self.l < 0.40: GRADE = 2
        elif self.l >= 0.40: GRADE = 3
        return GRADE
    
    def __init__(self, mask):
        self.input = mask
        self.valid = 1
        self.log = []
        self.mask = self.reprocessing()
        self.contour = self.getContour()
        self.centroid = self.getCentroid()
        self.a1, self.a2, self.p1, self.p2 = self.getCorners()
        self.ha, self.hp = self.getHeights()
        self.l = self.getLoss()
        self.g = self.getGrade()
