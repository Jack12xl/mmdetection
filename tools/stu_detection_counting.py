from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import glob
import os
import scipy.io
import numpy as np
import argparse



def getCount(matdir):
    mat = scipy.io.loadmat(matdir)
    return mat["image_info"][0][0][0][0][1][0][0]


def inferCount(imgdir, model):
    result = inference_detector(model, imgdir)
    return len(result[0])




def parse_args():
    parser = argparse.ArgumentParser(description="test counting model performance")

    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    data_root_dir = "./data/detection_data/detection_real_test_data/"
    gt_dirs = sorted(glob.glob(os.path.join(data_root_dir, "*.mat")))
    img_dirs = sorted(glob.glob(os.path.join(data_root_dir, "*.jpg")))
    assert (len(gt_dirs) == len(img_dirs))

    config_file = args.config
    checkpoint_file = args.checkpoint
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    gts = np.array(list(map(lambda x: getCount(x), gt_dirs)))
    res = np.array(list(map(lambda x: inferCount(x, model), img_dirs)))

    MSE = np.mean(np.power((gts - res), 2))
    MAE = np.mean(np.abs(gts - res))

    print("GT:")
    print(gts)

    print('Predict:')
    print(res)

    print("MSE: {}".format(MSE))
    print("MAE: {}".format(MAE))


if __name__ == '__main__':
    main()