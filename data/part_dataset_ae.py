import os
import numpy as np
import data_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PartDatasetAE():
    def __init__(self, root, npoints=400, class_choice='Chair', split='train', part_label=None):
        if part_label is None:
            print 'Wrong part label - part_dataset_ae'
            exit(1)

        self.npoints = npoints
        self.part_label = part_label
        self.cache = {}  # caching the loaded parts

        cat, meta = data_utils.data_parse(os.path.join(root, 'synsetoffset2category.txt'), class_choice, root, split)

        self.datapath = []
        for item in cat:
            for fn in meta[item]:
                # discard missing parts
                seg = np.loadtxt(fn[1]).astype(np.int64) - 1
                part_points = np.where(seg == self.part_label)
                if len(part_points[0]) > 1:
                    self.datapath.append((item, fn[0], fn[1]))

    def __getitem__(self, index):
        if index in self.cache:
            point_set = self.cache[index]
        else:
            point_set, seg = data_utils.get_point_set_and_seg(self.datapath, index)
            part_points = np.where(seg == self.part_label)
            point_set = data_utils.pc_normalize(point_set[part_points])
            self.cache[index] = point_set

        # choose the right number of point by
        # randomly picking, if there are too many
        # or re-sampling, if there are less than needed
        point_set_length = len(point_set)
        if point_set_length >= self.npoints:
            point_set, _ = data_utils.choose_points(point_set, self.npoints)
        else:
            extra_point_set, choice = data_utils.choose_points(point_set, self.npoints - point_set_length)
            point_set = np.append(point_set, extra_point_set, axis=0)

        return point_set

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    from utils import show3d_balls

    d = PartDatasetAE(root=os.path.join(BASE_DIR, '../data/shapenetcore_partanno_segmentation_benchmark_v0'),
                      class_choice='Chair', split='test', part_label=1)
    i = 27
    ps = d[i]
    show3d_balls.showpoints(ps, ballradius=8)
