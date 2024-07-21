import os
import sys
from sqlite3 import complete_statement

from tokenize import triple_quoted

import skimage.measure
import skimage.morphology
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import rasterio
from rasterio import features
import skimage

from .model import DexiNed
from .utils import fuse_predictions
from .datasets import OneImageDataset


# Size of single tile
WIDTH = 512
HEIGHT = 512

# Size of tile given to the model, every tile before passing through the network
# is resized to this size, around two times bigger is found to work the best.
# BEAWARE when changing this,
# only certain numbers work becouse of the architecture of the network. (kernel sizes and strides)
WIDTH_MODEL_INPUT = 1024
HEIGHT_MODEL_INPUT = 1024

def model_inference(dataloader, model, device, args={'is_testing': True}):
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            image_shape = sample_batched['image_shape']
            #print(f"input tensor shape: {images.shape}")
            preds = model(images)
            images = fuse_predictions(preds,
                                    image_shape,
                                    arg=args)
            torch.cuda.empty_cache()
            return images
    
def getPrediction(image, model, mean, device):
    """
    Gives edges for image preferably of size two time smaller
    than WIDTH_MODEL_INPUT by HEIGHT_MODEL_INPUT

    :param image: image of any size,
    :param model: must be DexiNed
    :param mean: torch.tensor containing 3 floats range [0,255] example: 
    :param device: device on which to run the model
    :return: grayscale image of detected edges
    """
    # Using modified TestDataset so that original code of other utils functions
    # can be reused without modifing them
    # BEAWARE that this is not the most optimal code
    dataset = OneImageDataset(image,
                              img_width=WIDTH_MODEL_INPUT,
                              img_height=HEIGHT_MODEL_INPUT,
                              mean_bgr=mean.tolist(),
                              )

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False)

    predictions = model_inference(dataloader, model, device)
    assert len(predictions) == 1

    return predictions[0]

def getEdges(image, model, mean, device):
    """
    Gives edges detected by model from any sized image,
    It divides the image into tiles, of size padding + WIDTH + padding , padding + HEIGHT + padding,
    where inner square of size WIDTH by HEIGHT is unique for each tile,
    and border of the square of size padding overlaps other tiles
    , on which edge detection is made.
    Then the tiles are combined back together to create the output edges.
    Padding is used becouse sometimes edges are detected on the edges of the tiles.
    It ensures that those edges are cut out of the tile before combining with others.

    :param image: image of any size,
    :param model: must be DexiNed
    :param mean: torch.tensor containing 3 floats range [0,255] example: 
    :param device: device on which to run the model
    :return: grayscale image of detected edges
    """ 

    # size of padding
    padding = 10

    img_width = image.shape[1]
    img_height = image.shape[0]

    num_squares_width = (img_width // WIDTH) + 1
    num_squares_height = (img_height // HEIGHT) + 1

    padding_down = num_squares_height * HEIGHT - img_height
    padding_right = num_squares_width * WIDTH - img_width

    image = cv2.copyMakeBorder(image, 0, padding_down, 0, padding_right, cv2.BORDER_CONSTANT)

    edges = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')

    for i in range(0, num_squares_width):
        for j in range(0, num_squares_height):
            j_start = j * HEIGHT
            j_end = (j * HEIGHT) + HEIGHT
            i_start = i * WIDTH
            i_end =  (i * WIDTH) + WIDTH

            if i != 0:
                i_start = i_start - padding
            if i != num_squares_width - 1:
                i_end = i_end + padding
            if j != 0:
                j_start = j_start - padding
            if j != num_squares_height - 1:
                j_end = j_end + padding
                
            current_tile = image[j_start : j_end, i_start : i_end]
            tile_edges = getPrediction(current_tile, model, mean, device)
            
            if i != 0:
                tile_edges = tile_edges[:,padding:]
            if i != num_squares_width - 1:
                tile_edges = tile_edges[:,:-padding]
            if j != 0:
                tile_edges = tile_edges[padding:,:]
            if j != num_squares_height - 1:
                tile_edges = tile_edges[:-padding,:]

            j_start = j * HEIGHT
            j_end = (j * HEIGHT) + HEIGHT
            i_start = i * WIDTH
            i_end =  (i * WIDTH) + WIDTH
            
            edges[j_start : j_end, i_start : i_end] = tile_edges

    edges = edges[:img_height, :img_width]
    return edges


def filterSmallAreas(segmented, skelet):
    regions_id = np.unique(segmented)

    for region_id in regions_id:
        if region_id == 0:
            continue
        region = (segmented == region_id)
        c, _ = cv2.findContours(np.array(region, dtype='uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ((x, y), radius) = cv2.minEnclosingCircle(c[0])
        if radius <= 5: # fields of area aprox 154 pixels = 15400 m^2
            skelet[region] = True
            continue

        area_in_pixels = region[region].size
        if area_in_pixels < 10: # that is 10 * 10 * 10 = 1000 m^2
            skelet[region] = True
    
    return skelet


def tiffToVector(path_to_tiff, low, high, model, mean, device, geo_reference=True):
    """
    Gives a list of polygons, if image is geo referenced the polygons are also, if not 
    returns polygons in pixel coordfinates. 
    :param path_to_tiff: path to image of any size,
    :param model: must be DexiNed
    :param mean: torch.tensor containing 3 floats range [0,255] example: 
    :param device: device on which to run the model
    :param geo_reference: if True and path_to_tiff contains georefrence
                          polygons will be georeferenced, otherwise polygons
                          in pixel coordinates are returned
    :return: list of polygons
    """ 

    image_src = rasterio.open(path_to_tiff)
    image_cv = cv2.imread(path_to_tiff)

    edges = getEdges(image_cv, model, mean, device)
    edges = np.invert(edges)

    thresh_img = skimage.filters.apply_hysteresis_threshold(edges, low, high)

    skelet = skimage.morphology.skeletonize(thresh_img)

    segmented = skimage.measure.label(skelet == 0, connectivity=1)

    filterSmallAreas(segmented, skelet)
    skelet = skimage.morphology.skeletonize(skelet)
    connected_edges = skimage.morphology.closing(skelet, skimage.morphology.disk(1))
    skelet = skimage.morphology.skeletonize(connected_edges)
    segmented = skimage.measure.label(skelet == 0, connectivity=1)
    region = skimage.morphology.closing(segmented, skimage.morphology.disk(1))

    if geo_reference:
        shapes = features.shapes(region, transform=image_src.transform)
    else:
        shapes = features.shapes(region)

    org = image_cv.copy()
    list_of_polygons = []
    for s in shapes:
        list_of_polygons.append(s)
    
    return list_of_polygons

def main():
    IMG_PATH = sys.argv[1]
    MODEL_PATH = sys.argv[2]
    
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    model = DexiNed().to(device)

    model.load_state_dict(torch.load(MODEL_PATH,
                                     map_location=device))

    if not os.path.exists("./mean"):
        print("Need to get mean of dataset the data was !!! trained !!! on, main.py generetes such file.")
        return
    else:
        mean_std = torch.load("./mean")
        print("Mean and std loaded from file.")

    mean, std = mean_std
    print(mean)

    image = cv2.imread(IMG_PATH)

    edges = getEdges(image, model, mean, device)
    cv2.imwrite("test_image.png", image)
    cv2.imwrite("test_edges.png", edges)

    return

if __name__ == "__main__":
    main()