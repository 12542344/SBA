import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import reduce        

def display(handle, shape = None, figsize = None):
    n = len(handle)
    if shape is None:
        if n > 1:
            factors  = np.sort(list(set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))
            ncol = factors[len(factors)//2]
            nrow = int(n/ncol)
            print(n," | ",nrow, " x ", ncol)
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
            for i,ax in enumerate(axes.flat):
                ax.imshow(handle[i], cmap = "bone")
                ax.axis('off')
            plt.show()
        else:
            plt.figure(figsize=figsize)
            plt.imshow(handle[0])
            plt.axis('off')
            plt.show()
    else:
        ncol = shape[1]
        nrow = shape[0]
        print(n," | ",nrow, " x ", ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        for i,ax in enumerate(axes.flat):
            ax.imshow(handle[i], cmap = "bone")
            ax.axis('off')
        plt.show()
        
def drawPoints(image, points, size, colour, thickness):
    for point in points:
        image = cv2.circle(image, point, size, colour, thickness)
    return image