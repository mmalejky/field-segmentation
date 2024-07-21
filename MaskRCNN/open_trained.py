import code
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import transforms as T
import os
import numpy as np
import cv2
import sys
from numpy.random import default_rng
from tv_training_code import FieldsCocoDataset, get_model_instance_segmentation_nowe

DATA = "data"
MASKS = "masks"
RED_FILL = True

class FieldsDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, DATA))))
        self.masks = list(sorted(os.listdir(os.path.join(root, MASKS))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, DATA, self.imgs[idx])
        mask_path = os.path.join(self.root, MASKS, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors in range from 1 to 255
        obj_ids = np.unique(mask)

        # If first id is 0 then it is a background
        if obj_ids[0] == 0:
       	    obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks

        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        num_good_objs = 0
        good_masks = []
        boxes = []
        for i in range(num_objs):
            if i >= masks.shape[0]:
                break

            pos = np.where(masks[i])

            if len(pos) < 2 or len(pos) > 2:
                print("Degenerated Mask Detected")
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmax - xmin <= 1 or ymax - ymin <=1:
                print("Degenerated Mask Detected")
            else:
                # Saving good masks
                good_masks.append(masks[i])
                num_good_objs += 1
                boxes.append([xmin, ymin, xmax, ymax])



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_good_objs,), dtype=torch.int64)

        masks = torch.as_tensor(np.array(good_masks), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

MODEL_PATH = sys.argv[1]
def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = get_model_instance_segmentation_nowe(2, False)
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        device = torch.device('cpu')
        model = get_model_instance_segmentation_nowe(2, False)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    datapath = "../our-data/DATA_DENMARK_2016_50PRC/"
    dataset_test = FieldsCocoDataset(datapath + "annotations/val2016.json", datapath + "images/val2016", "DENMARK2016", get_transform(train=False))
    img, target = dataset_test[3]

    with torch.no_grad():
        prediction = model([img.to(device)])

    rng = default_rng()
    org = img.mul(255).permute(1, 2, 0).byte().numpy()
    org = cv2.cvtColor(org, cv2.COLOR_RGB2BGR)

    org2 = org.copy()
    for i in range(prediction[0]['masks'].shape[0]):
        gray_image = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
        ret, thresh = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
        if RED_FILL:
            zeros_ch = np.zeros(gray_image.shape[0:2], dtype="uint8")
            red_mask = cv2.merge([zeros_ch, zeros_ch, thresh])
            org = cv2.addWeighted(org, 1, red_mask, 0.3, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_color = (rng.choice(255, size = 3, replace=False))
        new_color = (int (new_color[0]), int (new_color[1]), int (new_color[2]))
        cv2.drawContours(org, contours, -1, new_color, 1)

    for i in range(target['masks'].shape[0]):
        gray_image = target['masks'][i].mul(255).byte().cpu().numpy()
        ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        if RED_FILL:
            zeros_ch = np.zeros(gray_image.shape[0:2], dtype="uint8")
            red_mask = cv2.merge([zeros_ch, zeros_ch, thresh])
            org2 = cv2.addWeighted(org2, 1, red_mask, 0.3, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new_color = (rng.choice(255, size = 3, replace=False))
        new_color = (int (new_color[0]), int (new_color[1]), int (new_color[2]))
        cv2.drawContours(org2, contours, -1, new_color, 1)

    output_dir = "./visualisation/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_dir + "result.png", org)
    cv2.imwrite(output_dir + "truth.png", org2)

if __name__ == "__main__":
    main()
