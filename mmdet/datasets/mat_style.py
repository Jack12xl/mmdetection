# STU dataset
import os.path as osp
import mmcv
import numpy as np
import natsort
import glob

from .custom import CustomDataset

class MatSTUDataset(CustomDataset):
    '''
    # custom dataset for .mat
    '''
    CLASSES = ('person',)

    def __init__(self, **kwargs):
        super(MatSTUDataset, self).__init__(**kwargs)

    def load_annotations(self, ann_file):
        # store the annnotation
        self.bboxes = mmcv.load( ann_file )

        img_names = natsort.natsorted(glob.glob( osp.join(self.data_root, "*.jpg") ))

        def dir2dict(filedir):
            # all 1024 * 768 * 3
            return dict(filename=filedir, width=1024, height=768)

        img_infos = list(map(lambda x: dir2dict(x), img_names))

        assert( len(self.data) == len(img_infos) )

        return img_infos

    def get_ann_info(self, idx):
        bboxes = np.array(self.bboxes[idx], ndmin=2) - 1
        labels = np.ones(len(bboxes))
        # no bboxes_ignore
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0,))

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64)
        )

        return ann

