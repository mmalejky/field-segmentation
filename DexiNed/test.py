
from __future__ import print_function

import os
import time, platform

from torch.utils.data import DataLoader

from datasets import TestDataset
from losses import *
from model import DexiNed
from utils import (save_image_batch_to_disk,
                   count_parameters)

# Loads model from MODEL_DIR and runs it on INPUT_DIR and saves the results to OUTPUT_DIR
# Size of input images should be aprox 2 times smaller then the img_width and img_height
# specified in TestDataset

IS_LINUX = True if platform.system()=="Linux" else False

def test(checkpoint_path, dataloader, model, device, output_dir, args={'is_testing': True}):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            # images = images[:, [2, 1, 0], :, :]
            start_time = time.time()
            preds = model(images)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk(preds,
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     arg=args)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished *****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

def testPich(checkpoint_path, dataloader, model, device, output_dir, args={'is_testing': True}):
    # a test model plus the interganged channels
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device))

    # Put model in evaluation mode
    model.eval()

    with torch.no_grad():
        total_duration = []
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)

            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']
            print(f"input tensor shape: {images.shape}")
            start_time = time.time()
            # images2 = images[:, [1, 0, 2], :, :]  #GBR
            images2 = images[:, [2, 1, 0], :, :] # RGB
            preds = model(images)
            preds2 = model(images2)
            tmp_duration = time.time() - start_time
            total_duration.append(tmp_duration)
            save_image_batch_to_disk([preds,preds2],
                                     output_dir,
                                     file_names,
                                     image_shape,
                                     args, is_inchannel=True)
            torch.cuda.empty_cache()

    total_duration = np.array(total_duration)
    print("******** Testing finished ****")
    print("Average time per image: %f.4" % total_duration.mean(), "seconds")
    print("Time spend in the Dataset: %f.4" % total_duration.sum(), "seconds")

OUTPUT_DIR = "./results_of_tests"
MODEL_PATH = "./checkpoints/19/19_model.pth"
INPUT_DIR = "./data_to_test"


def main():
    """Main function."""

    print(f"Number of GPU's available: {torch.cuda.device_count()}")
    print(f"Pytorch version: {torch.__version__}")

    # path to saved model weights
    checkpoint_path = os.path.join(MODEL_PATH)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')

    # Instantiate model and move it to the computing device
    model = DexiNed().to(device)
    # model = nn.DataParallel(model)

    if not os.path.exists("./mean"):
        #mean_std = get_mean_and_std(dataset_path)
        #torch.save(mean_std, "./mean")
        print("Need to get mean of dataset the data was !!! trained !!! on train.py generetes such file.")
        return
    else:
        mean_std = torch.load("./mean")
        print("Mean and std loaded from file.")
    mean, std = mean_std
    output_dir = os.path.join(OUTPUT_DIR)
    print(f"output_dir: {output_dir}")

    dataset_val = TestDataset(INPUT_DIR,
                              img_width=128,
                              img_height=128,
                              #mean_bgr = dania_mean, 
                              mean_bgr=mean.tolist(), # default mean
                              )
    dataloader_val = DataLoader(dataset_val,
                                batch_size=1,
                                shuffle=False,
                                num_workers=1)#args.workers)

    # better result with normal test
    if False:
        # predict twice an image changing channels, then mix those results
        testPich(checkpoint_path, dataloader_val, model, device, output_dir)
    else:
        test(checkpoint_path, dataloader_val, model, device, output_dir)

    num_param = count_parameters(model)
    print('-------------------------------------------------------')
    print('Number of parameters of current DexiNed model:')
    print(num_param)
    print('-------------------------------------------------------')
    return

if __name__ == '__main__':
    main()
