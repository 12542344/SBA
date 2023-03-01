# import the necessary packages
import numpy as np
import math, argparse, imutils, cv2

# Class: SBA             
def my_euclidean2d(pointA, pointB):
    Ax = pointA[0]
    Ay = pointA[1]    
    Bx = pointB[0]
    By = pointB[1]
    Euclidean = math.sqrt((int(By)-int(Ay))**2 + (int(Bx)-int(Ax))**2)
    return Euclidean

def my_D2score(arrayA, pointB):
    D2score = 0
    for pointA in arrayA:
        Ax = pointA[0]
        Ay = pointA[1]    
        Bx = pointB[0]
        By = pointB[1]
        D2score = D2score + (By-Ay)**2 + (Bx-Ax)**2
    return D2score

def triangle_area(tri):
    x1, y1, x2, y2, x3, y3 = tri[0][0], tri[0][1], tri[1][0], tri[1][1], tri[2][0], tri[2][1]
    area = abs(0.5 * (((int(x2)-int(x1))*(int(y3)-int(y1)))-((int(x3)-int(x1))*(int(y2)-int(y1)))))
    return area

def Loss_2_Grade(LOSS):
    if LOSS < 0.2: GRADE = 0
    elif LOSS >= 0.2 and LOSS < 0.25: GRADE = 1
    elif LOSS >= 0.25 and LOSS < 0.40: GRADE = 2
    elif LOSS >= 0.40: GRADE = 3
    return GRADE

def filter_mask_binary(mask, method = "AREA"):
    _, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    h, w = mask.shape 
    mask = (mask//255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if method == "AREA":
        centroid = [mask.shape[1]/2, mask.shape[0]/2]
        blob_centroids = [findCentroid(cnt) for cnt in cnts]
        cnt = cnts[np.argmin([my_euclidean2d(value, centroid) for value in blob_centroids])]
    elif method == "CENTROID":
        cnt = max(cnts, key = cv2.contourArea)
    out = np.zeros(mask.shape, np.uint8)
    out = cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    _, out = cv2.threshold(out,127,255,cv2.THRESH_BINARY)
    return out

def findCentroid(contour):
    try:
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except: 
        cx, cy = 0, 0
    return [int(cx), int(cy)]

class SBA:      
    # The init method or constructor
    def __init__(self, mask):
        self.value = None
        self.centroid = None
        self.a1 = None
        self.a2 = None
        self.p1 = None
        self.p2 = None
        self.ha = None
        self.hp = None
        self.l = None
        self.g = None
        self.valid = 0
        self.log = None
        self.value = cv2.morphologyEx(cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1], 
                        cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=1)

        cnts, _ = cv2.findContours(self.value, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        centroid = [self.value.shape[1]/2,self.value.shape[0]/2]
        blob_centroids = [findCentroid(cnt) for cnt in cnts]
        contour = cnts[np.argmin([my_euclidean2d(value, centroid) for value in blob_centroids])]    
        area = cv2.contourArea(contour)
        centroid = np.asarray(findCentroid(contour))

        # compute the first_vertex
        euclideans = []
        for point in contour:
            pointA = centroid
            pointB = point[0]
            euclideans.append(my_euclidean2d(pointA, pointB))
        first_vertex = contour[np.argsort(np.asarray(euclideans))][-1:][0][0]

        # compute the second_vertex
        # compute the furthest vertex from first_vertex 
        euclideans = []
        for point in contour:
            pointA = first_vertex
            pointB = point[0]
            euclideans.append(my_euclidean2d(pointA, pointB))
        second_vertex = contour[np.argsort(np.asarray(euclideans))][-1:][0][0]

        # compute the third vertex
        # compute the furthest vertex from first_vertex, second_vertex
        triareas = []
        for point in contour:
            tri = [first_vertex, second_vertex, point[0]]
            triareas.append(triangle_area(tri))
        triareas = np.asarray(triareas).astype(np.int64)
        sort = contour[np.argsort(triareas)]
        third_vertex = sort[-1:][0][0]

        # compute the fourth_vertex
        # Chose vertex that maximise the area 
        triareas = []
        for point in contour:
            triangle_area1 = triangle_area([first_vertex, second_vertex, point[0]])
            triangle_area2 = triangle_area([second_vertex, third_vertex, point[0]])
            triangle_area3 = triangle_area([third_vertex, first_vertex, point[0]])
            triareas.append(triangle_area1+triangle_area2+triangle_area3)
        triareas = np.asarray(triareas).astype(np.int64)
        sort = contour[np.argsort(triareas)]
        fourth_vertex = sort[-1:][0][0]

        vertexs = np.asarray([first_vertex, second_vertex, third_vertex, fourth_vertex])
        vertexs = vertexs[np.argsort(vertexs[:,0])]

        # anterior    
        A = [vertexs[0], vertexs[1]]
        A = np.asarray(A)
        A_y = [A[0][1], A[1][1]]
        A_y = np.asarray(A_y)
        A = A[np.argsort(A_y)]
        A1 = A[0]
        A2 = A[1]

        # posterior
        P = [vertexs[2], vertexs[3]]
        P = np.asarray(P)
        P_y = [P[0][1], P[1][1]]
        P_y = np.asarray(P_y)
        P = P[np.argsort(P_y)]
        P1 = P[0]
        P2 = P[1]

        self.centroid = centroid
        self.a1 = A1
        self.a2 = A2
        self.p1 = P1
        self.p2 = P2
        self.ha = my_euclidean2d(A1, A2)
        self.hp = my_euclidean2d(P1, P2)
        self.l = 1 - (np.min(np.asarray([self.ha, self.hp])) / np.max(np.asarray([self.ha, self.hp])))
        self.g = Loss_2_Grade(self.l)
        self.contour = contour
        self.area = area
        self.valid = 1
        self.log = "Success"

        if np.count_nonzero(np.asarray([self.a1, self.a1, self.p1, self.p2])[:,1]) < 4:
            self.valid = 0
            self.log = "Vertex at border"
