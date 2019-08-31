import os
import numpy as np
import data_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PartDatasetPCN:
    def __init__(self, root, npoints=400, class_choice='Chair', split='train'):
        self.npoints = npoints
        self.cache = {}  # caching the loaded parts

        cat, meta = data_utils.data_parse(os.path.join(root, 'synsetoffset2category.txt'), class_choice, root, split)

        self.datapath = []
        for item in cat:
            for fn in meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.num_parts = data_utils.compute_num_of_parts(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            parts_point_sets = self.cache[index]
        else:
            point_set, seg = data_utils.get_point_set_and_seg(self.datapath, index)
            point_set = data_utils.pc_normalize(point_set)
            parts_point_sets = []
            for p in xrange(self.num_parts):
                part_points = np.where(seg == p)
                if len(part_points[0]) > 1:
                    part_point_set = point_set[part_points]
                    is_part_exist = True
                else:
                    part_point_set = np.zeros((self.npoints, 3))
                    is_part_exist = False
                # normalized each part on its own
                if is_part_exist:
                    norm_part_point_set = data_utils.pc_normalize(part_point_set)
                else:
                    norm_part_point_set = part_point_set
                parts_point_sets.append((part_point_set, norm_part_point_set, is_part_exist))
            self.cache[index] = parts_point_sets

        point_sets = []
        for point_set, norm_part_point_set, is_full in parts_point_sets:
            # choose the right number of point by
            # randomly picking, if there are too many
            # or re-sampling, if there are less than needed
            point_set_length = len(point_set)
            if point_set_length >= self.npoints:
                point_set, choice = data_utils.choose_points(point_set, self.npoints)
                norm_part_point_set = norm_part_point_set[choice]
            else:
                extra_point_set, choice = data_utils.choose_points(point_set, self.npoints - point_set_length)
                point_set = np.append(point_set, extra_point_set, axis=0)
                norm_part_point_set = np.append(norm_part_point_set, norm_part_point_set[choice], axis=0)
            point_sets.append((point_set, norm_part_point_set, is_full))

        return point_sets

    def __len__(self):
        return len(self.datapath)

    def get_number_of_parts(self):
        return self.num_parts


if __name__ == '__main__':
    from utils import show3d_balls

    d = PartDatasetPCN(root=os.path.join(BASE_DIR, '../data/shapenetcore_partanno_segmentation_benchmark_v0'),
                       class_choice='Chair', split='test')
    i = 27
    point_sets = d[i]
    for p in xrange(d.get_number_of_parts()):
        ps, _, _ = point_sets[p]
        show3d_balls.showpoints(ps, ballradius=8)
