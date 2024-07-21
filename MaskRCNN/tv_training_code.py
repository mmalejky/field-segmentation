# Modified from: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# USAGE: python ./PATH_TO_OLD_MODEL ./PATH_TO_SAVE_NEW_MODEL
#  OR    python ./PATH_TO_SAVE_NEW_MODEL
from dataclasses import dataclass
import json
import os
import pickle
import numpy as np
import torch
from PIL import Image
import sys
import cv2 as cv

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import torchvision.transforms as T2
import torch.nn.functional as F
import cv2 as cv
import albumentations as A

DATA = "data"
MASKS = "masks"

class FieldsCocoDataset(object):
    def __init__(self, json_file_path, images_path, name, transforms=None):
        dataset_path = "./coco_dataset_" + name
        if not os.path.exists(dataset_path):

            id_to_image = {}
            id_to_annotations = {}

            json_file = open(json_file_path)
            json_file = json.load(json_file)

            new_id = 0
            for image in json_file['images']:
                image_id = image['id']

                id_to_image[new_id] = image
                annotations = []
                for annotation in json_file['annotations']:
                    field_image_id = annotation['image_id']
                    if(image_id == field_image_id):
                        annotations.append(annotation)

                id_to_annotations[new_id] = annotations
                new_id += 1

            self.id_to_image = id_to_image
            self.id_to_annotations = id_to_annotations

            print("Coco precomputing done!")
            save_dict = {}
            save_dict['id_to_img'] = id_to_image
            save_dict['id_to_ann'] = id_to_annotations

            f = open(dataset_path,'wb+')
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(dataset_path, 'rb') as handle:
                save_dict = pickle.load(handle)
                self.id_to_image = save_dict['id_to_img']
                self.id_to_annotations = save_dict['id_to_ann']

        self.images_folder_path = images_path
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = self.id_to_image[idx]['file_name']
        annotations = self.id_to_annotations[idx]

        img = Image.open(os.path.join(self.images_folder_path, img_path)).convert("RGB")

        boxes = []
        labels = []
        masks = []
        areas = []
        is_crowd = []

        for annotation in annotations:
            bbox = np.array(annotation['bbox'], dtype='float32')
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            boxes.append(bbox)
            labels.append(annotation['category_id'])
            width = self.id_to_image[idx]['width']
            height = self.id_to_image[idx]['height']
            mask = np.zeros((width, height), dtype = 'uint8')

            poly = np.array(annotation['segmentation'], dtype='int32')
            new_poly = []
            for in_poly in poly:
                in_poly = np.split(in_poly,in_poly.shape[0] // 2)
                new_poly.append(in_poly)
            new_poly = np.array(new_poly, dtype = 'int32')
            mask = cv.fillPoly(mask, new_poly, color = 1)
            #mask_bool = mask.astype('bool')

            masks.append(mask)
            areas.append(annotation['area'])
            is_crowd.append(annotation['iscrowd'])

        boxes = torch.as_tensor(np.asarray(boxes), dtype=torch.float32)

        # there is only one class tak lub tak obydwa powinny byc ok
        #labels = torch.as_tensor(labels, dtype=torch.int64)
        labels = torch.ones((len(annotations),), dtype=torch.int64)

        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = torch.as_tensor(np.array(areas), dtype=torch.float32)
        # suppose all instances are not crowd
        #iscrowd = torch.as_tensor(np.array(is_crowd), dtype=torch.int64)
        iscrowd = torch.zeros((len(annotations),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        """
       transformPIL = T2.ToPILImage()
        img2 = transformPIL(img)

        import cv2

        cv2.imshow("trans", np.array(img2, dtype='uint8'))
        cv2.imshow("org", cv2.imread(os.path.join(self.images_folder_path, img_path)))
        cv2.waitKey(0)"""

        return img, target

    def __len__(self):
        return len(self.id_to_image)

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
        #print(img_path)
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
                #print("Degenerated Mask Detected")
                #print(pos)
                #print(img_path)
                #cv.imshow("maska", np.array(masks[i] * 255, dtype = 'uint8'))
                #cv.waitKey(0)
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmax - xmin <= 1 or ymax - ymin <=1:
                #print("Degenerated Mask Detected")
                #print(pos)
                #print(img_path)
                #cv.imshow("maska", np.array(masks[i] * 255, dtype = 'uint8'))
                #cv.waitKey(0)
                continue
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

        if num_good_objs == 0:
            area = torch.empty(0, 4)
            boxes = torch.empty(0, 4)
            masks = torch.empty(0, mask.shape[0], mask.shape[1])
        else:
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


from torchvision.models.detection.roi_heads import project_masks_on_boxes, maskrcnn_inference

def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    # type: (Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tensor
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])
    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    for gt_label, idxs in zip(gt_labels, mask_matched_idxs):
        print(gt_label[idxs])

    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size) for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss

import cv2 as cv

def maskrcnn_loss_wraper(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    mask_loss = maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
    #print(mask_logits)
    #print("DUPA")
    #print(proposals)
    #print(mask_matched_idxs[0].shape)
    #print(gt_masks)
    #print(gt_masks[0].shape)
    #print(mask_logits.shape)
    proposals = [p.long() for p in proposals]
    masks = maskrcnn_inference(mask_logits, proposals)
    masks_0_1 = torch.sigmoid(mask_logits).detach().numpy()
    masks_binary = np.where(np.array(masks_0_1) > 0.5, 1, 0 )
    #print(masks_0_1)
    #print(masks_binary)
    #cv.imshow("bin", np.array(masks_binary[0][1] * 255, dtype='uint8'))
    #cv.waitKey(0)
    #masks_sum =
    return mask_loss

def get_model_instance_segmentation_stare(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    #vision.torchvision.models.detection.roi_heads.maskrcnn_loss = maskrcnn_loss_wraper
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #model.roi_heads.maskrcnn_loss = maskrcnn_loss_wraper
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_model_instance_segmentation_nowe(num_classes, pretrained, mean=None, std=None):
    backbone  = resnet_fpn_backbone("resnet101", pretrained=pretrained)
    model = MaskRCNN(backbone=backbone,num_classes=num_classes, image_mean=mean, image_std=std)
    return model

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
# This approach wasn't tested yet
def get_different_backbone(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    backbone.out_channels = 1280

    backbone  = resnet_fpn_backbone("resnet101", pretrained=True, trainable_layers=3)
    print(backbone.eval())
    model = MaskRCNN(backbone=backbone, num_classes=2)
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    # Commented becouse I wrote stand alone augmentaion script
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomPhotometricDistort(p=0.5, p_all=0.5))
        transforms.append(T.GaussianNoise(0.5))
        #transforms.append(T.RandomRotate90(1.0)) cos nie dziala
    return T.Compose(transforms)

if len(sys.argv) == 2:
    MODEL_PATH = ""
    MODEL_SAVE_PATH = sys.argv[1]
else:
    MODEL_PATH = sys.argv[1]
    MODEL_SAVE_PATH = sys.argv[2]

def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        for img in data:
            channels_sum += torch.mean(img, dim=[1,2])
            channels_squared_sum += torch.mean(img**2, dim=[1,2])
            num_batches += 1
    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    #dataset = FieldsDataset('./data_generating/ruscy/37UDU_TCI', get_transform(train=True))

    #dataset_opole = FieldsDataset('./fields_aug', get_transform(train=True))

    path_2016 = "../our-data/DATA_DENMARK_2016_50PRC/"
    dataset_2016 = FieldsCocoDataset(path_2016 + "annotations/train2016.json",
                   path_2016 + "images/train2016", "DENMARK2016", get_transform(train=True))
    dataset_test_2016 = FieldsCocoDataset(path_2016 + "annotations/val2016.json",
                        path_2016 + "images/val2016", "DENMARK2016test", get_transform(train=False))
    path_2021 = "../our-data/DATA_DENMARK_2021_50PRC/"
    dataset_2021 = FieldsCocoDataset(path_2021 + "annotations/train2016.json",
                   path_2021 + "images/train2016", "DENMARK2021", get_transform(train=True))
    dataset_test_2021 = FieldsCocoDataset(path_2021 + "annotations/val2016.json",
                        path_2021 + "images/val2016", "DENMARK2021test", get_transform(train=False))

    dataset_comb = torch.utils.data.ConcatDataset([dataset_2016, dataset_2021])
    dataset_comb_test = torch.utils.data.ConcatDataset([dataset_test_2016, dataset_test_2021])

    # permute indices to permute input data
    indices = torch.randperm(len(dataset_comb)).tolist()
    indices_test = torch.randperm(len(dataset_comb_test)).tolist()
    # split the dataset in train and test set
    #size_of_training = (len(dataset) * 90) // 100
    dataset_comb = torch.utils.data.Subset(dataset_comb, indices)
    dataset_comb_test = torch.utils.data.Subset(dataset_comb_test, indices_test)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_comb, batch_size=2, shuffle=True,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_comb_test, batch_size=2, shuffle=False,
        collate_fn=utils.collate_fn)

    if not os.path.exists("./mean"):
        mean_std = get_mean_and_std(data_loader)
        torch.save(mean_std, "./mean")
    else:
        mean_std = torch.load("./mean")

    mean, std = mean_std

    # get the model using our helper function
    model = get_model_instance_segmentation_nowe(num_classes, pretrained=False, mean=mean, std=std)

    if MODEL_PATH != "":
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(MODEL_PATH))
        else:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)

    #optimizer = torch.optim.SGD(params, lr=0.0005,
    #                            momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                               step_size=3,
    #                                               gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 2

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate

        # Use this only with SGD
        #lr_scheduler.step()

        # evaluate on the test dataset
        if MODEL_SAVE_PATH != "":
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        evaluate(model, data_loader_test, device=device)


    print("That's it!")

    if MODEL_SAVE_PATH != "":
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
