import os
import torch
import numpy as np
from core.dataset import NpyFeature, NpyFeature_UTAL
from core.config import *
# from dataset import NpyFeature, NpyFeature_UTAL
# from config import *

from core.model import *
# from model import *
from sklearn.cluster import SpectralClustering


def get_affinity(video_index, video_feature, action_2_video):
    sorted_video_index = []
    sorted_video_feat = []
    cnt = 0
    for tmp_act in action_2_video:
        for tmp_vid in action_2_video[tmp_act]:
            if tmp_vid in sorted_video_index:
                continue
            if tmp_vid not in video_index:
                continue
            sorted_video_index.append(tmp_vid)
            tmp_vid_index = video_index.index(tmp_vid)
            sorted_video_feat.append(video_feature[tmp_vid_index])

    num_video = len(sorted_video_index)
    weight = np.zeros((num_video, num_video))
    beta = 0
    cnt = 0
    for i in range(num_video):

        for j in range(i, num_video):
            # calculate gamma
            cnt += 1
            beta += np.linalg.norm(sorted_video_feat[i] - sorted_video_feat[j])

            dis = np.square(np.linalg.norm(sorted_video_feat[i] - sorted_video_feat[j]))
            weight[i][j] = dis
            weight[j][i] = dis

    beta = beta / cnt
    gamma = - 1.0 / (2 * beta * beta)
    # print("gamma is %f " % gamma)

    weight = np.exp(gamma * weight)

    return weight, sorted_video_index, sorted_video_feat

def clustering(fea_vid_array, num_classes,weight=0.3):
    estimator = SpectralClustering(n_clusters=num_classes,random_state=0, affinity='precomputed')
    estimator.fit_predict(fea_vid_array)
    y_pred = estimator.labels_
    return y_pred

def get_cluster_performance(num_of_cluster, label_pred, subset_index, action_2_video, video_2_action,logger,step):
    '''
    :param num_of_cluster: 聚类的蔟数
    :param label_pred:  聚类预测的结果
    :param subset_index: 列表[file_name1,file_name2,file_name3]
    :param action_2_video:  {'label1':[file_name1,file_name2,...,file_namen]}
    :param video_2_action: {'file_name1':[label1,label2]}
    :return:
    '''
    cluster_res = {}
    for tmp_cls in range(num_of_cluster):
        cluster_res[tmp_cls] = []
    for i in range(len(label_pred)):
        tmp_cls = label_pred[i]
        file_name = subset_index[i]
        cluster_res[tmp_cls].append(file_name)

    all_precision = []
    all_recall = []
    all_cluster_label = []
    # add cluster index to action class mapping
    cluser_2_action = {}
    soft_cluster_2_action = {}

    total_true_cnt = 0

    # all class
    for label_index in range(num_of_cluster):
        all_class = list(action_2_video.keys())
        # print(all_class)
        action_cnt = {}
        for tc in all_class:
            action_cnt[tc] = 0
        # print(label_index)
        for tv in cluster_res[label_index]:
            tv_label = video_2_action[tv]
            for sig_label in tv_label:
                action_cnt[sig_label] += 1

            # set the label of cluster as the class which appear most
            max_cnt = 0
            cluster_label = ''
            for tmp_label in action_cnt:
                if action_cnt[tmp_label] > max_cnt:
                    max_cnt = action_cnt[tmp_label]
                    cluster_label = tmp_label
        all_cluster_label.append(cluster_label)

        # add cluster index to action class mapping
        cluser_2_action[label_index] = cluster_label

        soft_cluster_2_action[label_index] = []
        cluster_video_num = len(cluster_res[label_index])

        for tmp_label in action_cnt:
            if action_cnt[tmp_label] == 0:
                continue
            if action_cnt[tmp_label] == (max_cnt):
                # tmp_label_weight = 1.0 * action_cnt[tmp_label] / cluster_video_num
                tmp_label_weight = 1.0
                soft_cluster_2_action[label_index].append([tmp_label, tmp_label_weight])
            elif action_cnt[tmp_label] >= 0.5 * max_cnt:
                tmp_label_weight = 0.5
                soft_cluster_2_action[label_index].append([tmp_label, tmp_label_weight])

        precision = 1.0 * max_cnt / len(cluster_res[label_index])
        recall = 1.0 * max_cnt / len(action_2_video[cluster_label])
        total_true_cnt += max_cnt

        # print("********")
        # print("cluster label %d" % label_index)
        # print("num of video in cluster %d" % (len(cluster_res[label_index])))
        # print("match class %s" % (cluster_label))
        # print("num of all gt video %d" % (len(action_2_video[cluster_label])))
        # print("num of cluster gt video %d" % (max_cnt))
        # print("precision %.4f" % precision)
        # print("recall %.4f\n" % recall)
        action_cnt = sorted(action_cnt.items(), key=lambda e: e[1], reverse=True)
        # print(action_cnt[0:10])
        # print("\n")
        # print(soft_cluster_2_action[label_index])

        all_precision.append(precision)
        all_recall.append(recall)

    average_prec = np.mean(np.array(all_precision))
    average_recall = np.mean(np.array(all_recall))
    print("avg prec")
    print(average_prec)
    print("avg recall")
    print(average_recall)
    print("all prec %.4f" % (1.0 * total_true_cnt / len(subset_index)))

    all_cluster_label = list(set(all_cluster_label))
    print("num of cluster label %d" % (len(all_cluster_label)))

    video_index = subset_index
    gt_label = []
    for tv in video_index:
        gt_label.append(video_2_action[tv])
    #     non_over_label = []
    #     for i in range(label_pred.shape[0]):
    #         non_over_label.append(int(label_pred[i]))
    gt_label = np.array(gt_label)
    gt_label = np.squeeze(gt_label)
    # 记录评价标准 ars、nmi
    from sklearn import metrics
    ars = metrics.adjusted_rand_score(gt_label, label_pred)
    nmi = metrics.normalized_mutual_info_score(gt_label, label_pred)
    print("Adjusted rand score %.4f" % ars)
    print("NMI %.4f" % nmi)
    if logger!=None:
        logger.add_scalar('Adjusted rand score',ars,step)
        logger.add_scalar('NMI',nmi,step)

    return cluser_2_action, soft_cluster_2_action

def cluster_label(model, cfg,action2video,video2action, logger=None,init=False,step=0):
    model.eval()
    data_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
        batch_size=1,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=None)

    fuse_fea_list = []
    fea_name_list = []
    with torch.no_grad():
        for data, _, _, vid_name, _ in data_loader:
            data = data.cuda()
            if not init:
                video_scores, contrast_pairs, actionness, cas, embeddings = model(data)
                positive_act = contrast_pairs['PA']
                fuse_fea = torch.mean(positive_act,dim=1).squeeze(dim=0)
                fuse_fea = torch.nn.functional.normalize(fuse_fea,dim=0)
                fuse_fea = fuse_fea.cpu().detach().numpy()

            else:
                fuse_fea = torch.mean(data,dim=1).squeeze(dim=0)
                fuse_fea = torch.nn.functional.normalize(fuse_fea,dim=0)
                fuse_fea = fuse_fea.cpu().numpy()
            fuse_fea_list.append(fuse_fea)
            fea_name_list.append(vid_name[0])
    # todo 可能需要更改
    fuse_fea_array = np.array(fuse_fea_list)

    affinity_matrix,sorted_video_index,sorted_video_fea = get_affinity(fea_name_list,fuse_fea_array,action2video)
    fea_name_list = sorted_video_index
    fuse_fea_array = sorted_video_fea
    label_pred = clustering(affinity_matrix, cfg.NUM_CLASSES ,fea_name_list)
    cluster_2_action,soft_cluster_2_action = get_cluster_performance(cfg.NUM_CLASSES,label_pred,fea_name_list,action2video,video2action,logger,step)

    persudo_label_list = []
    for index in label_pred:
        label_action = soft_cluster_2_action[index]
        final_label_action = []
        for action in label_action:
            if action[1] == 1.0 or action[1]==0.5:
                final_label_action.append(action[0])
        persudo_label_list.append(final_label_action)

    return fea_name_list,persudo_label_list
    # train_data.update_label(fea_name_list, persudo_label_list)


if __name__=="__main__":
    from config import cfg
    net = HC_UTAL(cfg)
    net = net.cuda()
    cfg.MODEL_FILE = ''
    net.load_state_dict(torch.load(cfg.MODEL_FILE))
    net = net.cuda()
    train_dataset = NpyFeature_UTAL(data_path=cfg.DATA_PATH, mode='train',
                        modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                        num_segments=cfg.NUM_SEGMENTS, supervision='unsupervision',
                        class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random')
    cluster_label(net, cfg,train_dataset.action2video,train_dataset.video2action, logger=None,init = False,step=0)