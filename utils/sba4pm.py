# import the necessary packages
import numpy as np
import math, argparse, imutils, cv2
from operator import itemgetter, attrgetter
from shapely.geometry import Polygon, LineString, MultiLineString, Point, MultiPoint, GeometryCollection
from skimage.morphology import skeletonize, medial_axis
from scipy.optimize import minimize_scalar
from scipy.ndimage.morphology import distance_transform_edt

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

def findCentroid(contour, method = "MASS"):
    try:
        if method == "MASS": cx, cy = get_center_of_mass(contour)  
        elif method == "HALF_AREA_LINE": cx, cy = get_center_of_half_area_line(contour)
    except: 
        cx, cy = 0, 0
    return [int(cx), int(cy)]

def pointLocationInContour(point, contour):
    for count,value in enumerate(contour):
        compare = (value == point)
        compare = compare[0, 0] and compare[0,1]
        if compare == True: break
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

        if np.count_nonzero(np.asarray([A1, A2, P1, P2])[:,1]) < 4:
            self.valid = 0
            self.log = "Vertex at border"


#######################CENTROID#############################
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