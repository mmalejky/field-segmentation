import sys
from glob import iglob
import os
import cv2 as cv
import numpy as np
from skimage import filters
from skimage.morphology import skeletonize

# python drawing.py ./imgs_in ./edges_in OR python demo.py ./imgs_in
# Displays images from IMAGES_PATH with edges from EDGES_PATH or just images from IMAGES_PATH
# and allows for modyfing them by adding new edges
# Saves results to IMGS_OUTPUT_PATH and EDGES_OUTPUT_PATH

#  USAGE:
#   LEFT CLICK AND HOLD SHIFT TO DRAW LINE
#   WHEN YOU REALISE SHIFT LINE WITH NEW ORIGIN WILL START TO DRAW
#   ESCAPE - NEXT IMAGE
#   LEFT CLICK AND CTRL - SAVE IMAGE AND DELETE FROM IMAGES_PATH AND EDGES_PATH
#   LEFT CLICK AND LEFT ALT - UNDO

# Image and Edges must have the same filename for this script to work
IMAGES_PATH = sys.argv[1]

if len(sys.argv) <= 2:
    EDGES_PATH = ""
else:
    EDGES_PATH = sys.argv[2]

IMGS_OUTPUT_PATH = "./imgs_out"
EDGES_OUTPUT_PATH = "./edges_out"

ix,iy,sx,sy = -1,-1,-1,-1
save_flag = False
exit_flag = False
undo_flag = False
resize_ratio = 1

edges_history = []
image_history = []


def draw_lines(event, x, y, flags, param):  
        global ix,iy,sx,sy,save_flag,exit_flag,undo_flag

        if event == cv.EVENT_LBUTTONDOWN and (flags & cv.EVENT_FLAG_CTRLKEY):
            save_flag = True
            return

        if event == cv.EVENT_LBUTTONDOWN and (flags & cv.EVENT_FLAG_ALTKEY):
            undo_flag = True
            return

        if event == cv.EVENT_LBUTTONDOWN and (flags & cv.EVENT_FLAG_SHIFTKEY):

            if ix != -1: # if ix and iy are not first points, then draw a line
                image_history.append(img.copy())
                cv.line(img, (ix, iy), (x, y), (255, 255, 255), resize_ratio, cv.LINE_4)
                edges_history.append(edges.copy())
                cv.line(edges, (ix, iy), (x, y), (255), resize_ratio, cv.LINE_4)
            else: # if ix and iy are first points, store as starting points
                sx, sy = x, y

            ix,iy = x, y
            return
        
        if event == cv.EVENT_MOUSEMOVE and not (flags & cv.EVENT_FLAG_SHIFTKEY):
            ix, iy = -1, -1
    

img_path = IMAGES_PATH + "/**/*"
edge_path = EDGES_PATH + "/**/*"
image_list = [f for f in iglob(img_path, recursive=True) if os.path.isfile(f)]


for image_path in image_list:
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    image_org = image.copy()
    image = cv.resize(image, (image.shape[1] * resize_ratio, image.shape[0] * resize_ratio))

    edge_path = os.path.join(EDGES_PATH, os.path.basename(image_path))
    print(image.shape)
    # if you want to draw edges from beggining
    if EDGES_PATH == "":
        edges = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    else:
        # if you load edges from file
        edges = cv.imread(edge_path, cv.IMREAD_GRAYSCALE)
        print(edges.shape)
        edges = np.invert(edges)
        edges = edges > 210

        edges = skeletonize(edges)
        edges = np.array(edges * 255, dtype='uint8')
        edges = cv.resize(edges, (edges.shape[0] * resize_ratio, edges.shape[1] * resize_ratio))

    edges_copy = edges.copy()

    zeros_ch = np.zeros(image.shape[0:2], dtype="uint8")
    edge_mask = cv.merge([edges, edges, edges])
    img = cv.addWeighted(image, 1, edge_mask, 1, 0)
    img_copy = img.copy()
    ix, iy = -1, -1
    
    cv.namedWindow(os.path.basename(image_path), cv.WINDOW_NORMAL) 
    cv.setMouseCallback(os.path.basename(image_path), draw_lines)
    cv.resizeWindow(os.path.basename(image_path), 1000,1000)

    while(1):
        cv.imshow(os.path.basename(image_path),img)
        if cv.waitKey(20) & 0xFF == 27:
            print("Not Saved")
            break
        if save_flag:
            print("Saved")
            save_flag = False
            
            edges = (edges > 0) * 255
            cv.imwrite(os.path.join(IMGS_OUTPUT_PATH, os.path.basename(image_path)), image_org)
            cv.imwrite(os.path.join(EDGES_OUTPUT_PATH, os.path.basename(image_path)), edges)

            os.remove(image_path)
            if EDGES_PATH != "":
                os.remove(edge_path)
            break
        if undo_flag:
            if len(edges_history) > 0:
                img = image_history.pop()
                edges = edges_history.pop()
            else:
                img = img_copy.copy()
                edges = edges_copy.copy()

            undo_flag = False

    image_history = []
    edges_history = []

    cv.destroyAllWindows()

    
