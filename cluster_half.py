
from core.contrastive_loss import ClusterLoss, InstanceLoss
from core.network import Network
from terminaltables import AsciiTable
from torch.utils.tensorboard import SummaryWriter
from core.dataset import NpyFeature
from core.utils import AverageMeter
from core.config import cfg
import core.utils as utils
import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings

from eval.evaluation import evaluate
warnings.filterwarnings("ignore")

#DWA部分
use_dwa=True
if use_dwa:
    avg_cost = torch.zeros([12004,2]).float()
    dwa_t = 4.0


def train(cfg, writter, localization_net=None, init=False):
    worker_init_fn = np.random.seed(cfg.SEED)

    # utils.set_path(cfg)
    # utils.save_config(cfg)

    model = Network(cfg.FEATS_DIM*2, cfg.feature_dim, cfg.num_classes)
    model = model.to('cuda')

    train_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
        batch_size=cfg.bs,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
        batch_size=cfg.bs,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    criterion_instance = InstanceLoss(cfg.bs, cfg.instance_temperature).cuda()
    criterion_cluster = ClusterLoss(
        cfg.num_classes, cfg.cluster_temperature).cuda()

    min_loss = 10000000000
    best_nmi = 0
    epoch = 0
    for step in range(1, cfg.steps + 1):
        lr = optimizer.param_groups[0]["lr"]

        if (step - 1) % len(train_loader) == 0:
            loader_iter = iter(train_loader)
            epoch+=1

        batch_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        cost = train_one_step(model, loader_iter, optimizer,
                              criterion_instance, criterion_cluster, writter, step, cfg, localization_net, init, epoch)
        losses.update(cost.item(), cfg.bs)
        batch_time.update(time.time() - end)
        end = time.time()

        # if step == 1 or step % cfg.PRINT_FREQ == 0:
        #     print(('Step: [{0:04d}/{1}]\t'
        #            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
        #                step, cfg.steps, batch_time=batch_time, loss=losses)))

        if step > -1 and step % cfg.TEST_FREQ == 0:
            nmi, ari = test_all(model, cfg, test_loader,
                                localization_net, model_file=None, init=init)

            if nmi > best_nmi:
                best_nmi = nmi
                print('save model NMI = {:.4f} ARI = {:.4f} '.format(nmi, ari))
                torch.save(model.state_dict(), os.path.join(cfg.MODEL_PATH,
                                                            "model_best_cluster.pth.tar"))
                min_loss = losses.val

    cfg.MODEL_FILE = os.path.join(cfg.MODEL_PATH, "model_best_cluster.pth.tar")
    nmi, ari = test_all(model, cfg, test_loader,
                        localization_net, cfg.MODEL_FILE, init)
    file_path = os.path.join(cfg.OUTPUT_PATH, "best_cluster_results.txt")
    fo = open(file_path, "w")
    fo.write('NMI = {:.4f}  ARI = {:.4f} '.format(nmi, ari))
    fo.write('loss = {:.4f}'.format(min_loss))
    fo.close()

def train_one_step(net, loader_iter, optimizer, criterion_instance, criterion_cluster, writter, step, cfg, localization_net=None, init=False,epoch=0):
    net.train()
    data, label, _, _, _ = next(loader_iter)  # data[8,750,2048]
    optimizer.zero_grad()
    
    if init == False:
        data = data.cuda()
        localization_net.eval()
        video_scores, contrast_pairs, actionness, cas, embeddings = localization_net(
            data)
        # easy_act = contrast_pairs['EA']
        # half = easy_act.size(1)//2
        # x_i_act = easy_act[:, 0:half, :].cuda()
        # x_j_act = easy_act[:, half:, :].cuda()
        # x_i = x_i_act
        # x_j = x_j_act

        # easy_bkg = contrast_pairs['EB']
        # half = easy_bkg.size(1)//2
        # x_i_bkg = easy_bkg[:, 0:half, :].cuda()
        # x_j_bkg = easy_bkg[:, half:, :].cuda()

        # x_i = torch.cat((x_i_act, x_i_bkg), dim=1)
        # x_j = torch.cat((x_j_act, x_j_bkg), dim=1)
        # hard_act = contrast_pairs['HA']
        # half = hard_act.size(1)//2
        # x_i = hard_act[:, 0:half, :].cuda()
        # x_j = hard_act[:, half:, :].cuda()

        # half = embeddings.size(1)//2
        # x_i = embeddings[:, 0:half, :].cuda()
        # x_j = embeddings[:, half:, :].cuda()

        # half = embeddings.size(1)//2
        # actionness = torch.nn.functional.normalize(actionness, dim=1)
        # actionness = torch.unsqueeze(actionness, dim=2)
        # embeddings = embeddings*actionness
        # x_i = embeddings[:, 0:half, :].cuda()
        # x_j = embeddings[:, half:, :].cuda()

        # actionness = torch.nn.functional.normalize(actionness, dim=1)
        # actionness = torch.unsqueeze(actionness, dim=2)
        # data = data * actionness
        # half = data.size(1)//2
        # x_i = data[:, 0:half, :].cuda()
        # x_j = data[:, half:, :].cuda()

        idx = contrast_pairs['IDX']  # [8,150]
        half = idx.size(1)//2
        sort_idx, _ = torch.sort(idx)
        idx_topk = sort_idx.unsqueeze(2).expand(
            [-1, -1, data.shape[2]])  # [8,150,2048]
        idx_data = torch.gather(data, 1, idx_topk)  # []
        x_i = idx_data[:, 0:half, :].cuda()
        x_j = idx_data[:, half:, :].cuda()

    else:
        half = cfg.NUM_SEGMENTS//2
        x_i = data[:, 0:half, :].cuda()
        x_j = data[:, half:, :].cuda()

    z_i, z_j, c_i, c_j = net(x_i, x_j)


    loss_instance = criterion_instance(z_i, z_j)
    loss_cluster = criterion_cluster(c_i, c_j)


    ins_weight = torch.FloatTensor([1.0])
    clu_weight = torch.FloatTensor([1.0])

    if use_dwa:
        # if epoch > 2:
        #     ins_w = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
        #     clu_w = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
        #     ins_weight = 2 * torch.exp(ins_w / dwa_t) / (torch.exp(ins_w / dwa_t) + torch.exp(clu_w / dwa_t))
        #     clu_weight = 2 * torch.exp(clu_w / dwa_t) / (torch.exp(ins_w / dwa_t) + torch.exp(clu_w / dwa_t))
        
        if step > 2:
            ins_w = avg_cost[step - 1, 0] / avg_cost[step - 2, 0]
            clu_w = avg_cost[step - 1, 1] / avg_cost[step - 2, 1]
            ins_weight = 2 * torch.exp(ins_w / dwa_t) / (torch.exp(ins_w / dwa_t) + torch.exp(clu_w / dwa_t))
            clu_weight = 2 * torch.exp(clu_w / dwa_t) / (torch.exp(ins_w / dwa_t) + torch.exp(clu_w / dwa_t))

    if use_dwa:
        # avg_cost[epoch, 0] = loss_instance.detach()
        # avg_cost[epoch, 1] = loss_cluster.detach()
        avg_cost[step, 0] = loss_instance.detach()
        avg_cost[step, 1] = loss_cluster.detach()


    loss = ins_weight.cuda()*loss_instance + clu_weight.cuda()*loss_cluster
    loss.backward()
    optimizer.step()

    loss_dict = {
        'Loss/instance': loss_instance,
        'Loss/cluster': loss_cluster,
    }

    for key in loss_dict.keys():
        writter.add_scalar(key, loss_dict[key].cpu().item(), step)

    # if step % 50 == 0:
        # print(
        #     f"Step [{step}/{len(loader_iter)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
    return loss


def test(cfg, writter, localization_net=None, init=False):
    worker_init_fn = np.random.seed(cfg.SEED)
    model = Network(cfg.FEATS_DIM*2, cfg.feature_dim, cfg.num_classes)
    model = model.to('cuda')

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
        batch_size=cfg.bs,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)
    cfg.MODEL_FILE = os.path.join(cfg.MODEL_PATH, "model_best_cluster.pth.tar")
    nmi, ari = test_all(model, cfg, test_loader,
                        localization_net, cfg.MODEL_FILE, init)
    print('NMI = {:.4f}  ARI = {:.4f} '.format(nmi, ari))


def test_all(model, cfg, test_loader, localization_net=None, model_file=None, init=False):
    model.eval()

    if model_file:
        print('=> loading model: {}'.format(model_file))
        model.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    feature_vector = []
    labels_vector = []
    for step, (data, y, _, _, _) in enumerate(test_loader):
        if init == False:
            data = data.cuda()
            localization_net.eval()
            video_scores, contrast_pairs, actionness, cas, embeddings = localization_net(
                data)
            # easy_data = contrast_pairs['EA']
            # x_1 = easy_data

            idx = contrast_pairs['IDX']  # [8,150]
            sort_idx, _ = torch.sort(idx)
            idx_topk = sort_idx.unsqueeze(2).expand(
                [-1, -1, data.shape[2]])  # [8,150,2048]
            x_1 = torch.gather(data, 1, idx_topk).cuda()  # []

            # x_1 = embeddings.cuda()

            # actionness = torch.nn.functional.normalize(actionness, dim=1)
            # actionness = torch.unsqueeze(actionness, dim=2)
            # embeddings = embeddings*actionness
            # x_1 = embeddings.cuda()

            # actionness = torch.nn.functional.normalize(actionness, dim=1)
            # actionness = torch.unsqueeze(actionness, dim=2)
            # data = data * actionness
            # x_1 = data.cuda()

            # easy_act = contrast_pairs['EA']
            # easy_bkg = contrast_pairs['EB']
            # x_1 = torch.cat((easy_act, easy_bkg), dim=1)

        else:
            x_1 = data.cuda()
        with torch.no_grad():
            c = model.forward_cluster(x_1)
        c = c.detach()
        y = torch.argmax(y, dim=1)
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    # print("Features shape {}".format(feature_vector.shape))
    nmi, ari = evaluate(feature_vector, labels_vector)
    # print('NMI = {:.4f} ARI = {:.4f} '.format(nmi, ari))
    return nmi, ari


def cluster(cfg, localization_net=None, init=False):
    worker_init_fn = np.random.seed(cfg.SEED)

    model = Network(cfg.FEATS_DIM*2, cfg.feature_dim, cfg.num_classes)
    model = model.to('cuda')

    # train_loader = torch.utils.data.DataLoader(
    #     NpyFeature(data_path=cfg.DATA_PATH, mode='train',
    #                modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
    #                num_segments=cfg.NUM_SEGMENTS, supervision='weak',
    #                class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
    #     batch_size=cfg.bs,
    #     shuffle=True, num_workers=cfg.NUM_WORKERS,
    #     worker_jjinit_fn=worker_init_fn)

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='train',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random'),
        batch_size=cfg.bs,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    # cfg.MODEL_FILE = 'experiments/cluster_freq_1_instance_temperature_0.6_seed_42/model/model_best.pth.tar'
    vid_name_list,persudo_label_list,feature_vector = cluster_map_label(
        model, cfg, test_loader, cfg.MODEL_CLUSTER_FILE, localization_net, init)
    return vid_name_list, persudo_label_list,feature_vector


def cluster_map_label(model, cfg, test_loader, model_file=None, localization_net=None, init=False):
    model.eval()
    if model_file:
        print('=> loading model: {}'.format(model_file))
        model.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    feature_vector = []
    labels_vector = []
    vid_name_list = []
    for step, (data, y, _, vid_name, _) in enumerate(test_loader):
        if init == False:
            data = data.cuda()
            localization_net.eval()
            video_scores, contrast_pairs, actionness, cas, embeddings = localization_net(
                data)
            # easy_data = contrast_pairs['EA']
            # x_1 = easy_data

            idx = contrast_pairs['IDX']  # [8,150]
            sort_idx, _ = torch.sort(idx)
            idx_topk = sort_idx.unsqueeze(2).expand(
                [-1, -1, data.shape[2]])  # [8,150,2048]
            x_1 = torch.gather(data, 1, idx_topk).cuda()  # []

            # actionness = torch.nn.functional.normalize(actionness, dim=1)
            # actionness = torch.unsqueeze(actionness, dim=2)
            # data = data * actionness
            # x_1 = data.cuda()
        else:
            x_1 = data.cuda()

        # x = x.cuda()
        with torch.no_grad():
            c = model.forward_cluster(x_1)
        c = c.detach()
        y = torch.argmax(y, dim=1)
        feature_vector.extend(c.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        vid_name_list.extend(vid_name)
        # if step % 20 == 0:
        #     print(f"Step [{step}/{len(test_loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)

    # print("Features shape {}".format(feature_vector.shape))
    nmi, ari = evaluate(feature_vector, labels_vector)
    print('NMI = {:.4f} ARI = {:.4f} '.format(nmi, ari))
    persudo_label_list = get_label(cfg.num_classes, feature_vector, vid_name_list,
                                   test_loader.dataset.action2video, test_loader.dataset.video2action)
    return vid_name_list, persudo_label_list, feature_vector


def get_label(num_of_cluster, label_pred, vid_name_list, action_2_video, video_2_action):
    cluster_res = {}
    for tmp_cls in range(num_of_cluster):
        cluster_res[tmp_cls] = []
    for i in range(len(label_pred)):
        tmp_cls = label_pred[i]
        vid_name = vid_name_list[i]
        cluster_res[tmp_cls].append(vid_name)
    # print(cluster_res)
    all_precision = []
    all_recall = []
    all_cluster_label = []
    # add cluster index to action class mapping
    cluser_2_action = {}
    soft_cluster_2_action = {}
    total_true_cnt = 0

    for label_index in range(num_of_cluster):
        all_class = list(action_2_video.keys())
        # print(all_class)
        action_cnt = {}
        for tc in all_class:
            action_cnt[tc] = 0
            
        # print(label_index)
        cluster_label = ''
        for tv in cluster_res[label_index]:
            tv_label = video_2_action[tv]
            for sig_label in tv_label:
                action_cnt[sig_label] += 1

            # set the label of cluster as the class which appear most
            max_cnt = 0
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
                soft_cluster_2_action[label_index].append(
                    [tmp_label, tmp_label_weight])
            elif action_cnt[tmp_label] >= 0.5 * max_cnt:
                tmp_label_weight = 0.5
                soft_cluster_2_action[label_index].append(
                    [tmp_label, tmp_label_weight])

        # precision = 1.0 * max_cnt / len(cluster_res[label_index])
        # recall = 1.0 * max_cnt / len(action_2_video[cluster_label])
        # total_true_cnt += max_cnt

        action_cnt = sorted(action_cnt.items(),
                            key=lambda e: e[1], reverse=True)

        # all_precision.append(precision)
        # all_recall.append(recall)

    # average_prec = np.mean(np.array(all_precision))
    # average_recall = np.mean(np.array(all_recall))
    # print("avg prec")
    # print(average_prec)
    # print("avg recall")
    # print(average_recall)
    # print("all prec %.4f" % (1.0 * total_true_cnt / len(vid_name_list)))

    # all_cluster_label = list(set(all_cluster_label))
    # print("num of cluster label %d" % (len(all_cluster_label)))

    video_index = vid_name_list
    gt_label = []
    for tv in video_index:
        gt_label.append(video_2_action[tv][0])
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

    persudo_label_list = []
    for index in label_pred:
        label_action = soft_cluster_2_action[index]
        final_label_action = []
        for action in label_action:
            if action[1] == 1.0 or action[1] == 0.5:
                final_label_action.append(action[0])
        persudo_label_list.append(final_label_action)

    return persudo_label_list


if __name__ == "__main__":
    from core.model import CoLA
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    cfg.MODE = 'train'
    utils.set_path(cfg)
    writter = SummaryWriter(cfg.LOG_PATH)
    localization_net = CoLA(cfg)
    localization_net = localization_net.cuda()
    # model_file = 'experiments/have_trained/easy_5_hard_20_m_3_M_6_freq_5_seed_0/model/model_best.pth.tar'
    # localization_net.load_state_dict(torch.load(model_file))
    train(cfg, writter, localization_net, init=True)
    # test(cfg, writter, localization_net, init=False)
