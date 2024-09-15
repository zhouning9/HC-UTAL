import os
import json
import numpy as np
import torch
import random
import core.utils as utils
# import utils as utils
import torch.utils.data as data


class NpyFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, class_dict, seed=-1, supervision='weak'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(
                    data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(
                data_path, 'features', self.mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()
        print('=> {} set has {} videos'.format(mode, len(self.vid_list)))

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = class_dict
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

        self.action2video, self.video2action = self.mapper(self.anno)

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                               vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                                vid_name + '.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                           vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        # classwise_anno = [[]] * self.num_classes
        classwise_anno = []
        for i in range(self.num_classes):
            classwise_anno.append([])
        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(
                _anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_seg, self.num_classes])
            t_factor = self.feature_fps / 16

            for class_idx in range(self.num_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]

            return label, torch.from_numpy(temp_anno)

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

    def mapper(self, data):
        # 获得和聚类相关的数据
        action2video = {}
        video2action = {}

        data = data['database']

        for file_name in data:
            # in th14 is "train" ,and in anet is "training"
            if 'train' not in data[file_name]['subset']:
                continue
            ############ video 2 action ##########
            s = set()
            for i in range(len(data[file_name]['annotations'])):
                label = data[file_name]['annotations'][i]['label']
                s.add(label)

                action_s = action2video.get(label, set())
                action_s.add(file_name)
                action2video[label] = action_s

            video2action[file_name] = list(s)

        for action_name in action2video:
            s = action2video[action_name]
            action2video[action_name] = list(s)

        return action2video, video2action

    def load_data(self, n_similar=3, batch_size=16):
        features = []
        labels = []
        idx = []

        # Load similar pairs
        rand_classid = np.random.choice(
            self.num_classes, size=n_similar)  # 选择3个不同类，分别为这三个类挑选pair
        for rid in rand_classid:
            class_name = list(self.action2video.keys())[rid]
            rand_sampleid = np.random.choice(
                len(self.action2video[class_name]), size=2)
            name0 = self.action2video[class_name][rand_sampleid[0]]
            name1 = self.action2video[class_name][rand_sampleid[1]]
            idx.append(self.vid_list.index(name0))
            idx.append(self.vid_list.index(name1))

        # Load rest pairs
        rand_sampleid = np.random.choice(
            len(self.vid_list), size=batch_size - 2 * n_similar)
        for r in rand_sampleid:
            idx.append(r)

        for i in idx:
            data, vid_num_seg, sample_idx = self.get_data(i)
            label, temp_anno = self.get_label(i, vid_num_seg, sample_idx)
            features.append(data.cpu().numpy())
            labels.append(label)

        return torch.tensor(features), torch.tensor(labels)


class NpyFeature_UTAL(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, class_dict, seed=-1, supervision='weak'):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(
                    data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(
                data_path, 'features', self.mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()
        print('=> {} set has {} videos'.format(mode, len(self.vid_list)))

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()

        self.class_name_to_idx = class_dict
        self.num_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling

        self.action2video, self.video2action = self.mapper(self.anno)

    def mapper(self, data):
        # 获得和聚类相关的数据
        action2video = {}
        video2action = {}

        data = data['database']

        for file_name in data:
            if data[file_name]['subset'] == 'test':
                continue
            ############ video 2 action ##########
            s = set()
            for i in range(len(data[file_name]['annotations'])):
                label = data[file_name]['annotations'][i]['label']
                s.add(label)

                action_s = action2video.get(label, set())
                action_s.add(file_name)
                action2video[label] = action_s

            video2action[file_name] = list(s)

        for action_name in action2video:
            s = action2video[action_name]
            action2video[action_name] = list(s)

        return action2video, video2action

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, temp_anno = self.get_label(index, vid_num_seg, sample_idx)

        return data, label, temp_anno, self.vid_list[index], vid_num_seg

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                               vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                                vid_name + '.npy')).astype(np.float32)

            vid_num_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            flow_feature = flow_feature[sample_idx]

            feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        else:
            feature = np.load(os.path.join(self.feature_path,
                                           vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        if self.supervision == 'unsupervision':
            vid_name = self.vid_list[index]
            anno_list = self.anno[vid_name]
            label = np.zeros([self.num_classes], dtype=np.float32)

            for _anno in anno_list:
                label[self.class_name_to_idx[_anno]] = 1

            return label, torch.Tensor(0)

        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)

        classwise_anno = [[]] * self.num_classes

        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(
                _anno)

        if self.supervision == 'weak':
            return label, torch.Tensor(0)
        else:
            temp_anno = np.zeros([vid_num_seg, self.num_classes])
            t_factor = self.feature_fps / 16

            for class_idx in range(self.num_classes):
                if label[class_idx] != 1:
                    continue

                for _anno in classwise_anno[class_idx]:
                    tmp_start_sec = float(_anno['segment'][0])
                    tmp_end_sec = float(_anno['segment'][1])

                    tmp_start = round(tmp_start_sec * t_factor)
                    tmp_end = round(tmp_end_sec * t_factor)

                    temp_anno[tmp_start:tmp_end+1, class_idx] = 1

            temp_anno = temp_anno[sample_idx, :]

            return label, torch.from_numpy(temp_anno)

    def update_label(self, vid_name_list, label_list):
        assert len(vid_name_list) == len(
            label_list), 'the number of vid list is not equal label list !'
        self.anno = {}
        cluster_action2video = {}
        for i in range(len(vid_name_list)):
            vid_name = vid_name_list[i]
            label_tmp = label_list[i]
            self.anno[vid_name] = label_tmp

            for _label in label_tmp:
                action_s = cluster_action2video.get(_label, set())
                action_s.add(vid_name)
                cluster_action2video[_label] = action_s

        for action_name in cluster_action2video:
            s = cluster_action2video[action_name]
            cluster_action2video[action_name] = list(s)

        self.init_flag = True
        self.cluster_action2video = cluster_action2video

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

    def load_data(self, n_similar=3, batch_size=16):
        features = []
        labels = []
        idx = []

        # Load similar pairs
        rand_classid = np.random.choice(
            len(self.cluster_action2video), size=n_similar)  # 选择3个不同类，分别为这三个类挑选pair
        for rid in rand_classid:
            class_name = list(self.cluster_action2video.keys())[rid]
            rand_sampleid = np.random.choice(
                len(self.cluster_action2video[class_name]), size=2)
            name0 = self.cluster_action2video[class_name][rand_sampleid[0]]
            name1 = self.cluster_action2video[class_name][rand_sampleid[1]]
            idx.append(self.vid_list.index(name0))
            idx.append(self.vid_list.index(name1))

        # Load rest pairs
        rand_sampleid = np.random.choice(
            len(self.vid_list), size=batch_size - 2 * n_similar)
        for r in rand_sampleid:
            idx.append(r)

        for i in idx:
            data, vid_num_seg, sample_idx = self.get_data(i)
            label, temp_anno = self.get_label(i, vid_num_seg, sample_idx)
            features.append(data.cpu().numpy())
            labels.append(label)

        return torch.tensor(features), torch.tensor(labels)


class pairDataset(data.Dataset):
    def __init__(self, cfg, phase="train", sample="random"):
        self.phase = phase
        self.sample = sample
        self.data_dir = cfg.DATA_PATH
        self.sample_segments_num = cfg.NUM_SEGMENTS

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]

        self.feature_path = []
        for _modal in ['rgb', 'flow']:
            self.feature_path.append(os.path.join(
                self.data_dir, 'features', self.phase, _modal))
        self.data_list = list(
            open(os.path.join(self.data_dir, "split_train.txt")))
        self.data_list = [item.strip() for item in self.data_list]

        # self.class_name_lst = cfg.class_name_lst
        # self.action_class_idx_dict = {action_cls: idx for idx, action_cls in enumerate(self.class_name_lst)}
        self.action_class_idx_dict = cfg.CLASS_DICT

        self.action_class_num = len(self.action_class_idx_dict)

        self.label_dict = {}
        self.get_label()

        self.pos_pair_list = []
        self.neg_pair_list = []
        self.pair_list = []
        self.get_pair()

        self.feature_list = []
        self.get_feature()

    def get_label(self):

        for vid_name in self.data_list:

            vid_anns_list = self.gt_dict[vid_name]["annotations"]
            vid_label = np.zeros(self.action_class_num)
            for ann in vid_anns_list:
                ann_label = ann["label"]
                vid_label[self.action_class_idx_dict[ann_label]] = 1.0

            self.label_dict[vid_name] = vid_label

    def get_group(self):

        for idx in range(len(self.data_list)):

            vid_name = self.data_list[idx]
            group_idx = np.argwhere(self.label_dict[vid_name] == 1)
            for class_idx in group_idx:
                self.group_list[class_idx[0]].append(idx)

    def get_pair(self):

        for i in range(len(self.data_list)):
            label_0 = np.array(self.label_dict[self.data_list[i]])
            for j in range(i + 1, len(self.data_list)):
                label_1 = np.array(self.label_dict[self.data_list[j]])
                if (label_0 == label_1).all():
                    self.pos_pair_list.append([i, j])
                elif np.sum(label_0 * label_1) == 0:
                    self.neg_pair_list.append([i, j])

        self.neg_pair_list = random.sample(
            self.neg_pair_list, len(self.pos_pair_list))
        self.neg_pair_list.extend(self.pos_pair_list)
        self.pair_list = self.neg_pair_list

    def get_feature(self):

        for vid_name in self.data_list:
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                               vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                                vid_name + '.npy')).astype(np.float32)

            con_vid_feature = np.concatenate(
                (rgb_feature, flow_feature), axis=1)

            if self.sample == "random":
                con_vid_spd_feature = self.random_sample(
                    con_vid_feature, self.sample_segments_num)
            else:
                con_vid_spd_feature = self.uniform_sample(
                    con_vid_feature, self.sample_segments_num)

            self.feature_list.append(con_vid_spd_feature)

    def __len__(self):
        if self.phase == 'train':
            return len(self.pair_list)
        else:
            return len(self.data_list)

    def __getitem__(self, idx):

        if self.phase == 'train':
            idx_1, idx_2 = self.pair_list[idx]

            feature_1 = torch.as_tensor(
                self.feature_list[idx_1].astype(np.float32))
            feature_2 = torch.as_tensor(
                self.feature_list[idx_2].astype(np.float32))

            label_1 = torch.as_tensor(
                self.label_dict[self.data_list[idx_1]].astype(np.float32))
            label_2 = torch.as_tensor(
                self.label_dict[self.data_list[idx_2]].astype(np.float32))

            return feature_1, feature_2, label_1, label_2

        else:
            vid_name = self.data_list[idx]
            vid_label = self.label_dict[vid_name]
            vid_duration = self.gt_dict[vid_name]["duration"]
            con_vid_feature = self.feature_list[idx]
            vid_len = con_vid_feature.shape[0]

            if self.sample == "random":
                con_vid_spd_feature = self.random_sample(
                    con_vid_feature, self.sample_segments_num)
            else:
                con_vid_spd_feature = self.uniform_sample(
                    con_vid_feature, self.sample_segments_num)

            con_vid_spd_feature = torch.as_tensor(
                con_vid_spd_feature.astype(np.float32))

            vid_label_t = torch.as_tensor(vid_label.astype(np.float32))

            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration

    def random_sample(self, input_feature, sample_len):
        input_len = input_feature.shape[0]
        assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(
            sample_len)

        if input_len < sample_len:
            sample_idxs = np.random.choice(input_len, sample_len, replace=True)
            sample_idxs = np.sort(sample_idxs)
        elif input_len > sample_len:
            sample_idxs = np.arange(sample_len) * input_len / sample_len
            for i in range(sample_len - 1):
                sample_idxs[i] = np.random.choice(
                    range(np.int(sample_idxs[i]), np.int(sample_idxs[i + 1] + 1)))
            sample_idxs[-1] = np.random.choice(
                np.arange(sample_idxs[-2], input_len))
        else:
            sample_idxs = np.arange(input_len)

        return input_feature[sample_idxs.astype(np.int), :]

    def uniform_sample(self, input_feature, sample_len):
        input_len = input_feature.shape[0]
        assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(
            sample_len)

        if sample_len >= input_len > 1:
            sample_idxs = np.arange(input_len)
        else:
            if input_len == 1:
                sample_len = 2
            sample_scale = input_len / sample_len
            sample_idxs = np.arange(sample_len) * sample_scale
            sample_idxs = np.floor(sample_idxs)

        return input_feature[sample_idxs.astype(np.int), :]


class pairDatasetUTAL(data.Dataset):
    def __init__(self, cfg, phase="train", sample="random"):
        self.phase = phase
        self.sample = sample
        self.data_dir = cfg.DATA_PATH
        self.sample_segments_num = cfg.NUM_SEGMENTS

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]

        self.feature_path = []
        for _modal in ['rgb', 'flow']:
            self.feature_path.append(os.path.join(
                self.data_dir, 'features', self.phase, _modal))
        self.data_list = list(
            open(os.path.join(self.data_dir, "split_train.txt")))
        self.data_list = [item.strip() for item in self.data_list]

        # self.class_name_lst = cfg.class_name_lst
        # self.action_class_idx_dict = {action_cls: idx for idx, action_cls in enumerate(self.class_name_lst)}
        self.action_class_idx_dict = cfg.CLASS_DICT

        self.action_class_num = len(self.action_class_idx_dict)

        self.label_dict = {}
        self.get_label()

        self.pos_pair_list = []
        self.neg_pair_list = []
        self.pair_list = []
        self.get_pair()

        self.feature_list = []
        self.get_feature()

        self.action2video, self.video2action = self.mapper(self.gt_dict)

    def mapper(self, data):
        # 获得和聚类相关的数据
        action2video = {}
        video2action = {}

        # data = data['database']

        for file_name in data:
            if data[file_name]['subset'] == 'test':
                continue
            ############ video 2 action ##########
            s = set()
            for i in range(len(data[file_name]['annotations'])):
                label = data[file_name]['annotations'][i]['label']
                s.add(label)

                action_s = action2video.get(label, set())
                action_s.add(file_name)
                action2video[label] = action_s

            video2action[file_name] = list(s)

        for action_name in action2video:
            s = action2video[action_name]
            action2video[action_name] = list(s)

        return action2video, video2action

    def update_label(self, vid_name_list, label_list):
        assert len(vid_name_list) == len(
            label_list), 'the number of vid list is not equal label list !'
        self.label_dict = {}
        cluster_action2video = {}
        for i in range(len(vid_name_list)):
            vid_name = vid_name_list[i]
            label_tmp = label_list[i]
            vid_label = np.zeros(self.action_class_num)
            for _label in label_tmp:
                action_s = cluster_action2video.get(_label, set())
                action_s.add(vid_name)
                cluster_action2video[_label] = action_s
                vid_label[self.action_class_idx_dict[_label]] = 1.0
            self.label_dict[vid_name] = vid_label

        for action_name in cluster_action2video:
            s = cluster_action2video[action_name]
            cluster_action2video[action_name] = list(s)

        self.init_flag = True
        self.cluster_action2video = cluster_action2video

        self.pos_pair_list = []
        self.neg_pair_list = []
        self.pair_list = []
        self.get_pair()

    def get_label(self):
        for vid_name in self.data_list:
            vid_anns_list = self.gt_dict[vid_name]["annotations"]
            vid_label = np.zeros(self.action_class_num)
            for ann in vid_anns_list:
                ann_label = ann["label"]
                vid_label[self.action_class_idx_dict[ann_label]] = 1.0
            self.label_dict[vid_name] = vid_label

    def get_group(self):

        for idx in range(len(self.data_list)):

            vid_name = self.data_list[idx]
            group_idx = np.argwhere(self.label_dict[vid_name] == 1)
            for class_idx in group_idx:
                self.group_list[class_idx[0]].append(idx)

    def get_pair(self):

        for i in range(len(self.data_list)):
            label_0 = np.array(self.label_dict[self.data_list[i]])
            for j in range(i + 1, len(self.data_list)):
                label_1 = np.array(self.label_dict[self.data_list[j]])
                if (label_0 == label_1).all():
                    self.pos_pair_list.append([i, j])
                elif np.sum(label_0 * label_1) == 0:
                    self.neg_pair_list.append([i, j])

        self.neg_pair_list = random.sample(
            self.neg_pair_list, len(self.pos_pair_list))
        self.neg_pair_list.extend(self.pos_pair_list)
        self.pair_list = self.neg_pair_list

    def get_feature(self):

        for vid_name in self.data_list:
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                               vid_name + '.npy')).astype(np.float32)
            flow_feature = np.load(os.path.join(self.feature_path[1],
                                                vid_name + '.npy')).astype(np.float32)

            con_vid_feature = np.concatenate(
                (rgb_feature, flow_feature), axis=1)

            if self.sample == "random":
                con_vid_spd_feature = self.random_sample(
                    con_vid_feature, self.sample_segments_num)
            else:
                con_vid_spd_feature = self.uniform_sample(
                    con_vid_feature, self.sample_segments_num)

            self.feature_list.append(con_vid_spd_feature)

    def __len__(self):
        if self.phase == 'train':
            return len(self.pair_list)
        else:
            return len(self.data_list)

    def __getitem__(self, idx):

        if self.phase == 'train':
            idx_1, idx_2 = self.pair_list[idx]

            feature_1 = torch.as_tensor(
                self.feature_list[idx_1].astype(np.float32))
            feature_2 = torch.as_tensor(
                self.feature_list[idx_2].astype(np.float32))

            label_1 = torch.as_tensor(
                self.label_dict[self.data_list[idx_1]].astype(np.float32))
            label_2 = torch.as_tensor(
                self.label_dict[self.data_list[idx_2]].astype(np.float32))

            return feature_1, feature_2, label_1, label_2

        else:
            vid_name = self.data_list[idx]
            vid_label = self.label_dict[vid_name]
            vid_duration = self.gt_dict[vid_name]["duration"]
            con_vid_feature = self.feature_list[idx]
            vid_len = con_vid_feature.shape[0]

            if self.sample == "random":
                con_vid_spd_feature = self.random_sample(
                    con_vid_feature, self.sample_segments_num)
            else:
                con_vid_spd_feature = self.uniform_sample(
                    con_vid_feature, self.sample_segments_num)

            con_vid_spd_feature = torch.as_tensor(
                con_vid_spd_feature.astype(np.float32))

            vid_label_t = torch.as_tensor(vid_label.astype(np.float32))

            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration

    def random_sample(self, input_feature, sample_len):
        input_len = input_feature.shape[0]
        assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(
            sample_len)

        if input_len < sample_len:
            sample_idxs = np.random.choice(input_len, sample_len, replace=True)
            sample_idxs = np.sort(sample_idxs)
        elif input_len > sample_len:
            sample_idxs = np.arange(sample_len) * input_len / sample_len
            for i in range(sample_len - 1):
                sample_idxs[i] = np.random.choice(
                    range(np.int(sample_idxs[i]), np.int(sample_idxs[i + 1] + 1)))
            sample_idxs[-1] = np.random.choice(
                np.arange(sample_idxs[-2], input_len))
        else:
            sample_idxs = np.arange(input_len)

        return input_feature[sample_idxs.astype(np.int), :]

    def uniform_sample(self, input_feature, sample_len):
        input_len = input_feature.shape[0]
        assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(
            sample_len)

        if sample_len >= input_len > 1:
            sample_idxs = np.arange(input_len)
        else:
            if input_len == 1:
                sample_len = 2
            sample_scale = input_len / sample_len
            sample_idxs = np.arange(sample_len) * sample_scale
            sample_idxs = np.floor(sample_idxs)

        return input_feature[sample_idxs.astype(np.int), :]


if __name__ == "__main__":
    from config import cfg
    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='test',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='uniform'),
        batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS)
