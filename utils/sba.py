# import the necessary packages
import numpy as np
import math
import argparse
import imutils
import cv2
from operator import itemgetter, attrgetter

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

def AB_Loss(A,B, type):
    if type == "i":
        if B >= A:
            R_A_B = A/B    
        elif B < A:
            R_A_B = B/A
        LOSS = (1-R_A_B)*100
    elif type == "o":
        R_A_B = A/B
        LOSS = (1-R_A_B)*100
    return LOSS

def Loss_2_Grade(LOSSs):
    GRADEs = []
    for LOSS in LOSSs:
        GRADE = []
        if LOSS < 20:
            GRADE = 0
        elif LOSS >= 20 and LOSS < 25:
            GRADE = 1
        elif LOSS >= 25 and LOSS < 40:
            GRADE = 2
        elif LOSS >= 40:
            GRADE = 3
        GRADEs.append(GRADE)
    return GRADEs

def filter_mask_binary(mask):
    _, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    h, w = mask.shape
    #kernel = np.ones((int(h/100), int(w/100)), np.uint8)
    #mask = cv2.erode(cv2.dilate(mask, kernel), kernel) 
    mask = (mask//255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    centroid = [mask.shape[1]/2, mask.shape[0]/2]
    blob_centroids = [findCentroid(cnt) for cnt in cnts]
    cnt = cnts[np.argmin([my_euclidean2d(value, centroid) for value in blob_centroids])]
    out = np.zeros(mask.shape, np.uint8)
    out = cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    _, out = cv2.threshold(out,127,255,cv2.THRESH_BINARY)
    return out

def filter_mask_binary_max(mask):
    _, mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    h, w = mask.shape
    mask = (mask//255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key = cv2.contourArea)
    out = np.zeros(mask.shape, np.uint8)
    out = cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    _, out = cv2.threshold(out,127,255,cv2.THRESH_BINARY)
    return out

def findCentroid_upgrade(contour):
    cx, cy = get_center_of_mass(contour)
    return [int(cx), int(cy)]

def findCentroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx, cy = get_center_of_half_area_line(contour)
    else:
        cx, cy = 0, 0
    return [int(cx), int(cy)]

def pointLocationInContour(point, contour):
    for count,value in enumerate(contour):
        compare = (value == point)
        compare = compare[0, 0] and compare[0,1]
        if compare == True:
            break
    return count

def partialContourBetween2Points(A,B,contour):
    counterA = pointLocationInContour(A,contour)
    counterB = pointLocationInContour(B,contour)
    return contour[counterA:counterB+1]

def point2set(p1,p2):
    set = []
    p = p1
    d = p2-p1
    N = np.max(np.abs(d))
    s = d/N
    set.append(np.rint(p).astype('int'))
    for ii in range(0,N):
        p = p+s;
        set.append(np.rint(p).astype('int'))
    return np.array(set)

def contourFull(contour):
    new_contour = []
    for count,value in enumerate(contour):
        len_contour = len(contour)
        if count < len_contour-1:
            p1 = value
            p2 = contour[count+1]
        elif count >= len_contour-1:
            p1 = contour[count]
            p2 = contour[0]
        set = point2set(p1,p2)
        set = np.delete(set, 0, 0)
        new_contour.extend(set)
    return np.array(new_contour)

def contourMerge(contourA, contourB):
    contourB = contourB[::-1]
    contourAtoB = point2set(contourA[-1],contourB[0])
    contourBtoA = point2set(contourB[-1],contourA[0])
    contour = np.concatenate((contourA, contourAtoB[1:], contourB, contourBtoA[1:-1]), axis = 0)
    return contour

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
        self.archTop = None
        self.archBot = None
        self.archLeft = None
        self.archRight = None
        self.contour = None
        self.area = None
        self.valid = 0
        
        self.value = cv2.morphologyEx(cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1], 
                        cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)), iterations=1)
        try:
            cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            centroid = [mask.shape[1]/2, mask.shape[0]/2]
            blob_centroids = [findCentroid(cnt) for cnt in cnts]
            contour = cnts[np.argmin([my_euclidean2d(value, centroid) for value in blob_centroids])]    
            if len(contour) >= 10:
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
                points = [A1, A2, P2, P1]
                location = []
                for point in points:
                    location.append(pointLocationInContour(point,contour))

                if location[0] > location[1]:
                    #print("Case 1")
                    archLeft1 = partialContourBetween2Points(A1,contour[-1],contour)
                    archLeft2 = partialContourBetween2Points(contour[0],A2,contour)
                    archLeft = np.concatenate((archLeft1, archLeft2))
                    archBot = partialContourBetween2Points(A2,P2,contour)
                    archRight = partialContourBetween2Points(P2,P1,contour)
                    archTop = partialContourBetween2Points(P1,A1,contour)
                elif location[2] > location[3]:
                    #print("Case 2")
                    archRight1 = partialContourBetween2Points(P2,contour[-1],contour)
                    archRight2 = partialContourBetween2Points(contour[0],P1,contour)
                    archRight = np.concatenate((archRight1, archRight2))
                    archTop = partialContourBetween2Points(P1,A1,contour)
                    archLeft = partialContourBetween2Points(A1,A2,contour)
                    archBot = partialContourBetween2Points(A2,P2,contour)
                elif location[1] > location[2]:
                    #print("Case 3")
                    archBot1 = partialContourBetween2Points(A2,contour[-1],contour)
                    archBot2 = partialContourBetween2Points(contour[0],P2,contour)
                    archBot = np.concatenate((archBot1, archBot2))
                    archTop = partialContourBetween2Points(P1,A1,contour)
                    archLeft = partialContourBetween2Points(A1,A2,contour)
                    archRight = partialContourBetween2Points(P2,P1,contour)
                else:
                    #print("Case 4")
                    archTop1 = partialContourBetween2Points(P1,contour[-1],contour)
                    archTop2 = partialContourBetween2Points(contour[0],A1,contour)
                    archTop = np.concatenate((archTop1, archTop2))
                    archBot = partialContourBetween2Points(A2,P2,contour)
                    archLeft = partialContourBetween2Points(A1,A2,contour)
                    archRight = partialContourBetween2Points(P2,P1,contour)

                self.centroid = centroid
                self.a1 = A1
                self.a2 = A2
                self.p1 = P1
                self.p2 = P2
                self.ha = my_euclidean2d(A1, A2)
                self.hp = my_euclidean2d(P1, P2)
                self.l = 1 - (np.min(np.asarray([self.ha, self.hp])) / np.max(np.asarray([self.ha, self.hp])))
                self.archTop = archTop[::-1]
                self.archBot = archBot
                self.archLeft = archLeft
                self.archRight = archRight
                self.contour = contour
                self.area = area
                self.valid = 1

                if np.count_nonzero(np.asarray([A1, A2, P1, P2])[:,1]) < 4:
                    print("Vertex at border")
                    self.valid = 0
                elif len(archTop) < 1 and len(archBot) < 1 and len(archLeft) < 1 and len(archRight) < 1:
                    print("Arch < 1")
                    self.valid = 0
            else:
                print("Contour < 10")
                self.valid = 0
        except:
            pass
 
# Class: SBAvertebra
def closestPoint(point, points):
    dist_2 = np.sum((points - point)**2, axis=2)
    element = np.argmin(dist_2)
    return dist_2[element], points[element]

def closestPointToModel(points, model_points):
    distance = []
    for point in points:
        distance.append(closestPoint(point, model_points)[0])
    return points[np.argmin(distance)]

def isFracture(HA, HM, HP):
    H = np.array([HA,HM,HP])
    Hmin = np.min(H)
    Hmax = np.max(H)

    R = Hmin / Hmax 
    L = 1 - R
    if L <= 0.2:
        G = 0 # Grade 0 / Normal
    elif L > 0.2 and L <= 0.25:
        G = 1 # Grade 1 / Mild
    elif L > 0.25 and L <= 0.40:
        G = 2 # Grade 2 / Moderate
    elif L > 0.40:
        G = 3 # Grade 3 / Severe

    if HA > HM and HP > HM:
        C = 3 # Crush
    elif HP > HM and HM > HA:
        C = 1 # Bi-concave
    else: 
        C = 2 # Wedge

    return L, G, C

def polarSwitchContour(a, ind):
    # Create an auxiliary array of twice size.
    n = len(a)
    b = np.concatenate((a,a), axis = 0)
    i = ind
    b = b[ind+1:(n+ind-1)]
    return b

def midPointCalibrate(a1,m1,p1,p2,m2,a2):
    simple_shape = Polygon([a1,m1,p1,p2,m2,a2])
    y_min = min(simple_shape.exterior.xy[1])
    y_max = max(simple_shape.exterior.xy[1])
    m1c, m2c = (a1+p1)/2,(a2+p2)/2
    if (m1c[0] - m2c[0]) != 0:
        a = (m1c[1] - m2c[1]) / (m1c[0] - m2c[0])
        b = m1c[1] - a * m1c[0]
        m1p = [(y_min-b)/a,y_min]
        m2p = [(y_max-b)/a,y_max]
    else:
        m1p = [m1c[0],y_min]
        m2p = [m1c[0],y_max]
    mid_line = LineString([m1p,m2p])
    m1_update,m2_update = mid_line.intersection(simple_shape).coords
    return a1,np.asarray(m1_update).astype(int),p1,p2,np.asarray(m2_update).astype(int),a2

class SBAvertebra:
    # The init method or constructor
    def __init__(self, image, masks):
        self.I = None
        self.A = None
        self.B = None
        self.log = None
        self.contourTop = None
        self.contourBot = None
        self.archTopUp = None
        self.archTopLow = None
        self.archBotUp = None
        self.archBotLow = None
        self.contourTopCentroid = None
        self.contourBotCentroid = None
        self.contourTopQuadratic = None
        self.contourBotQuadratic = None
        self.centroid = None
        self.a1 = None
        self.a2 = None
        self.p1 = None
        self.p2 = None
        self.m1 = None
        self.m2 = None
        self.ha = None
        self.hm = None
        self.hp = None
        self.l = None
        self.g = None
        self.c = None
        self.valid = 0
        try:
            SBA_A = SBA(masks[0])
            SBA_B = SBA(masks[1])
            valid = []
            log = []
            if (SBA_A.valid + SBA_B.valid) == 2:

                original_resolution = (SBA_A.value.shape[1], SBA_A.value.shape[0])

                #### top
                cnt_top = contourMerge(SBA_A.archTop, SBA_B.archTop)  
                l = cnt_top    
                x = [l1[0][0] for l1 in l]
                y = [l1[0][1] for l1 in l]
                model_top = np.poly1d(np.polyfit(x, y, 2))

                #### bot
                cnt_bot = contourMerge(SBA_A.archBot, SBA_B.archBot)
                l = cnt_bot    
                x = [l1[0][0] for l1 in l]
                y = [l1[0][1] for l1 in l]
                model_bot = np.poly1d(np.polyfit(x, y, 2))

                #### when either bot or top centroid is 0, 0
                cnt_top_centroid = findCentroid(cnt_top)
                cnt_bot_centroid = findCentroid(cnt_bot)
                if cnt_top_centroid == [0, 0]:
                    cnt_top_centroid = [cnt_bot_centroid[0], 0]
                    log.append("M1 was modified based on M2")
                elif cnt_bot_centroid == [0, 0]:
                    cnt_bot_centroid = [cnt_top_centroid[0], image.shape[1]]
                    log.append("M2 was modified based on M1")
                elif cnt_top_centroid == cnt_bot_centroid:
                    log.append("M1 was at M2")

                #### point reselection
                ##### model fitting
                #X = np.linspace(0, original_resolution[0]-1, original_resolution[0])

                #Y = model_top(X)
                #model_points = np.asarray([[[X[count], Y[count]]] for count, value in enumerate(Y) if Y[count] >= 0 and X[count] >= 0]).astype(np.int32)
                #A1 = closestPointToModel([SBA_A.a1, SBA_B.a1], model_points)
                #A2 = closestPointToModel([SBA_A.a2, SBA_B.a2], model_points)

                #Y = model_bot(X)
                #model_points = np.asarray([[[X[count], Y[count]]] for count, value in enumerate(Y) if Y[count] >= 0 and X[count] >= 0]).astype(np.int32)
                #P1 = closestPointToModel([SBA_A.p1, SBA_B.p1], model_points)
                #P2 = closestPointToModel([SBA_A.p2, SBA_B.p2], model_points)

                A1 = np.asarray((SBA_A.a1 + SBA_B.a1)/2).astype(int)
                A2 = np.asarray((SBA_A.a2 + SBA_B.a2)/2).astype(int)
                P1 = np.asarray((SBA_A.p1 + SBA_B.p1)/2).astype(int)
                P2 = np.asarray((SBA_A.p2 + SBA_B.p2)/2).astype(int)
                
                cnt_top = polarSwitchContour(cnt_top, pointLocationInContour(A1, cnt_top))
                arch_top_low = np.concatenate((np.array([[A1]]) ,cnt_top[:pointLocationInContour(P1, cnt_top)+1]), axis = 0)
                arch_top_up = np.concatenate((cnt_top[pointLocationInContour(P1, cnt_top):], np.array([[A1]])), axis = 0)

                cnt_bot = polarSwitchContour(cnt_bot, pointLocationInContour(A2, cnt_bot))
                arch_bot_low = np.concatenate((np.array([[A2]]) ,cnt_bot[:pointLocationInContour(P2, cnt_bot)+1]), axis = 0)
                arch_bot_up = np.concatenate((cnt_bot[pointLocationInContour(P2, cnt_bot):], np.array([[A2]])), axis = 0)

                try:
                    _,M1,_,_,M2,_ = midPointCalibrate(A1,cnt_top_centroid,P1,P2,cnt_bot_centroid,A2)
                except:
                    M1,M2 = cnt_top_centroid, cnt_bot_centroid

                HA = my_euclidean2d(A1, A2)
                HM = my_euclidean2d(M1, M2)
                HP = my_euclidean2d(P1, P2)

                L, G, C = isFracture(HA, HM, HP)
            else:
                if SBA_A.valid == 0:
                    log.append("A was invalid")
                if SBA_B.valid == 0:
                    log.append("B was invalid")

            if log == []:
                log.append("Successfully identified")

                self.contourTop = cnt_top
                self.contourBot = cnt_bot

                self.archTopUp = arch_top_up
                self.archTopLow = arch_top_low
                self.archBotUp = arch_bot_up
                self.archBotLow = arch_bot_low

                self.contourTopCentroid = cnt_top_centroid
                self.contourBotCentroid = cnt_bot_centroid

                self.contourTopQuadratic = model_top
                self.contourBotQuadratic = model_bot

                self.centroid = np.asarray((SBA_A.centroid + SBA_B.centroid)/2).astype(int)

                self.a1 = np.asarray(A1).astype(int)
                self.a2 = np.asarray(A2).astype(int)

                self.p1 = np.asarray(P1).astype(int)
                self.p2 = np.asarray(P2).astype(int)

                self.m1 = np.asarray(M1).astype(int)
                self.m2 = np.asarray(M2).astype(int)

                self.ha = HA
                self.hm = HM
                self.hp = HP

                self.l = L
                self.g = G
                self.c = C
                self.valid = 1

            else:
                self.valid = 0

            self.I = image
            self.A = SBA_A
            self.B = SBA_B
            self.log = log
        except:
            self.valid = 0

    def get6Points(self):
        return np.array([self.a1, self.m1, self.p1, self.a2, self.m2, self.p2])

    def getNPoints(self, nsplit):
        N_POINTS = []
        ALL_POINTS = []
        SIX_POINTS = np.array([self.a1, self.m1, self.p1, self.a2, self.m2, self.p2])
        for arch in [self.archTopUp, self.archTopLow, self.archBotUp, self.archBotLow]:
            arch_split = np.array_split(arch, nsplit)
            for i in range(1,nsplit,1):
                N_POINTS.extend(arch_split[i][0])
        ALL_POINTS = np.concatenate((SIX_POINTS, N_POINTS), axis = 0)
        return ALL_POINTS
    
    def getHeights(self):
        return np.array([self.ha, self.hm, self.hp])


#######################CENTROID#############################
import numpy as np
import cv2
from shapely.geometry import Polygon, LineString, MultiLineString, Point, MultiPoint, GeometryCollection
from skimage.morphology import skeletonize, medial_axis
from scipy.optimize import minimize_scalar
from scipy.ndimage.morphology import distance_transform_edt

H, W = 500, 500

def get_center_of_mass(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy

def split_mask_by_line(mask, centroid:tuple, theta_degrees:float, eps:float = 1e-4):
    h, w = mask.shape[:2]
    mask = mask[..., None]
    cx, cy = centroid
    # convert theta first to radians and then to line slope(s)
    theta_degrees = np.atleast_1d(theta_degrees)
    theta_degrees = np.clip(theta_degrees, -90+eps, 90-eps)
    theta_rads = np.radians(theta_degrees)
    slopes = np.tan(theta_rads)[:, None]
    # define the line(s)
    x = np.arange(w, dtype="int32")
    y = np.int32(slopes * (x - cx) + cy)
    _y = np.arange(h, dtype="int32")
    # split the input mask into two halves by line(s)
    m = (y[..., None] <= _y).T
    m1 = (m * mask).sum((0,1))
    m2 = ((1 - m) * mask).sum((0,1))
    m2 = m2 + eps if m2==0 else m2
    # calculate the resultant masks ratio
    ratio = m1/m2
    return (x.squeeze(), y.squeeze()), ratio

def get_half_area_line(mask, centroid: tuple, eps: float = 1e-4):
    # find the line that splits the input mask into two equal area halves
    minimize_fun = lambda theta: abs(1. - split_mask_by_line(mask, centroid, theta, eps=eps)[1].item())
    bounds = np.clip((-90, 90), -90 + eps, 90 - eps)
    res = minimize_scalar(minimize_fun, bounds=bounds, method='bounded')
    theta_min = res.x
    line, _ = split_mask_by_line(mask, centroid, theta_min)
    return line

def get_representative_point(cnt):
    poly = Polygon(cnt.squeeze())
    cx = poly.representative_point().x
    cy = poly.representative_point().y
    return cx, cy

def get_skeleton_center_of_mass(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    skel = medial_axis(mask//255).astype(np.uint8) #<- medial_axis wants binary masks with value 0 and 1
    skel_cnt,_ = cv2.findContours(skel,1,2)
    skel_cnt = skel_cnt[0]
    M = cv2.moments(skel_cnt) 
    if(M["m00"]==0): # this is a line
        cx = int(np.mean(skel_cnt[...,0]))
        cy = int(np.mean(skel_cnt[...,1]))
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return cx, cy

def get_furthest_point_from_edge(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    d = distance_transform_edt(mask)
    cy, cx = np.unravel_index(d.argmax(), d.shape)
    return cx, cy

def get_furthest_point_from_edge_cv2(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    dist_img = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    cy, cx = np.where(dist_img==dist_img.max())
    cx, cy = cx.mean(), cy.mean()  # there are sometimes cases where there are multiple values returned for the visual center
    return cx, cy

def get_center_of_half_area_line(cnt):
    mask = draw_contour_on_mask((H,W), cnt, color=1)
    # get half-area line that passes through centroid
    cx, cy = get_center_of_mass(mask)
    line = get_half_area_line(mask, centroid=(cx, cy))
    line = LineString(np.array(list(zip(line))).T.reshape(-1, 2))
    # find the visual center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 5]
    polys = [Polygon(c.squeeze(1)) for c in contours if len(c) >= 3]  # `Polygon` must have at least 3 points
    cpoint = Point(cx, cy)
    points = []
    for poly in polys:
        if poly.is_valid == False:
            poly = poly.buffer(0)
        isect = poly.intersection(line)
        if isect.is_empty:
            # skip when intersection is empty: this can happen for masks that consist of multiple disconnected parts
            continue
        if isinstance(isect, (MultiLineString, GeometryCollection)):
            # take the line segment intersecting with `poly` that is closest to the centroid point
            isect = isect.geoms[np.argmin([g.distance(cpoint) for g in isect.geoms])]
        if isinstance(isect, Point):
            # sometimes the intersection can be a singleton point
            points.append(isect)
            continue
        isect = isect.boundary
        if poly.intersects(cpoint):
            points = [isect]
            break
        else:
            points.append(isect)

    if len(points) == 0:
        # multiple reasons for this one:
        # - if len(polys)==0
        # - if len(polys)==1, but for some reason the line does not intersect with polygon
        # - if the above search does not match with any points

        return cx, cy

    points = points[np.argmin([p.distance(cpoint) for p in points])]
    if isinstance(points, Point):
        return np.array(points.xy)
    
    points = [np.array(p.xy).tolist() for p in points.geoms]
    visual_center = np.average(points, (0, 2))
    return visual_center

def draw_contour_on_mask(size, cnt, color:int = 255):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.drawContours(mask, [cnt], -1, color, -1)
    return mask