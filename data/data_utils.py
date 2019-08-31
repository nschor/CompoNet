import os
import json
import numpy as np
import part_dataset_ae
import part_dataset_pcn


def load_data(data_path, num_point, category, seen_split, unseen_split):
    pcn_train_dataset, pcn_test_dataset, num_parts = load_pcn_data(data_path, num_point, category, seen_split,
                                                                   unseen_split)
    ae_train_dataset, ae_test_dataset = load_aes_data(data_path, num_point, category, seen_split, unseen_split,
                                                      num_parts)

    return pcn_train_dataset, pcn_test_dataset, ae_train_dataset, ae_test_dataset, num_parts


def load_pcn_data(data_path, num_point, category, seen_split, unseen_split):
    pcn_train_dataset = part_dataset_pcn.PartDatasetPCN(root=data_path, npoints=num_point, class_choice=category,
                                                        split=seen_split)
    pcn_test_dataset = part_dataset_pcn.PartDatasetPCN(root=data_path, npoints=num_point, class_choice=category,
                                                       split=unseen_split)
    num_parts = pcn_train_dataset.get_number_of_parts()

    return pcn_train_dataset, pcn_test_dataset, num_parts


def load_aes_data(data_path, num_point, category, seen_split, unseen_split, num_parts):
    ae_train_dataset = []
    ae_test_dataset = []
    for i in xrange(num_parts):
        print 'Loading part ' + str(i)
        ae_train_dataset.append(
            part_dataset_ae.PartDatasetAE(root=data_path, npoints=num_point, class_choice=category, split=seen_split,
                                          part_label=i))
        ae_test_dataset.append(
            part_dataset_ae.PartDatasetAE(root=data_path, npoints=num_point, class_choice=category, split=unseen_split,
                                          part_label=i))

    return ae_train_dataset, ae_test_dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def data_parse(catfile, class_choice, root, split):
    cat = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    cat = {k: v for k, v in cat.items() if k in class_choice}

    meta = {}
    with open(os.path.join(root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    with open(os.path.join(root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    with open(os.path.join(root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
    for item in cat:
        meta[item] = []
        dir_point = os.path.join(root, cat[item], 'points')
        dir_seg = os.path.join(root, cat[item], 'points_label')
        fns = sorted(os.listdir(dir_point))
        if split == 'trainval':
            fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
        elif split == 'train':
            fns = [fn for fn in fns if fn[0:-4] in train_ids]
        elif split == 'val':
            fns = [fn for fn in fns if fn[0:-4] in val_ids]
        elif split == 'test':
            fns = [fn for fn in fns if fn[0:-4] in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

    return cat, meta


def compute_num_of_parts(path):
    max_num_parts = 0
    for i in range(len(path) / 50):
        num_parts = len(np.unique(np.loadtxt(path[i][-1]).astype(np.uint8)))
        if num_parts > max_num_parts:
            max_num_parts = num_parts
    return max_num_parts


def get_point_set_and_seg(path, index):
    fn = path[index]
    point_set = np.loadtxt(fn[1]).astype(np.float32)
    seg = np.loadtxt(fn[2]).astype(np.int64) - 1

    return point_set, seg


def choose_points(point_set, npoints):
    choice = np.random.choice(len(point_set), npoints, replace=True)
    point_set = point_set[choice]

    return point_set, choice
