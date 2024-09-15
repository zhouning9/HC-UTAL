from terminaltables import AsciiTable
from eval.eval_detection import ANETdetection
from torch.utils.tensorboard import SummaryWriter
from core.dataset import NpyFeature, NpyFeature_UTAL
from core.utils import AverageMeter
from core.config import cfg
from core.model import HC_UTAL
import core.utils as utils
import os
import sys
import time
import copy
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_ID
    worker_init_fn = None
    if cfg.SEED >= 0:
        utils.set_seed(cfg.SEED)
        worker_init_fn = np.random.seed(cfg.SEED)

    # 更改参数
    cfg.CAS_THRESH = np.arange(0.325, 0.375, 0.025)
    cfg.ANESS_THRESH = np.arange(0.025, 0.95, 0.025)
    cfg.MAGNITUDES_THRESH = np.arange(0.325, 0.925, 0.025)  # 0.4 0.8
    cfg.NMS_THRESH = 0.55
    
    utils.set_path(cfg)
    utils.save_config(cfg)

    localization_net = HC_UTAL(cfg)
    localization_net = localization_net.cuda()

    train_dataset = NpyFeature_UTAL(data_path=cfg.DATA_PATH, mode='train',
                                    modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                                    num_segments=cfg.NUM_SEGMENTS, supervision='unsupervision',
                                    class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='random')

    test_loader = torch.utils.data.DataLoader(
        NpyFeature(data_path=cfg.DATA_PATH, mode='test',
                   modal=cfg.MODAL, feature_fps=cfg.FEATS_FPS,
                   num_segments=cfg.NUM_SEGMENTS, supervision='weak',
                   class_dict=cfg.CLASS_DICT, seed=cfg.SEED, sampling='uniform'),
        batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        worker_init_fn=worker_init_fn)

    test_info = {"step": [], "test_acc": [], "average_mAP": [],
                 "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [],
                 "mAP@0.4": [], "mAP@0.5": [], "mAP@0.6": [],
                 "mAP@0.7": []}

    best_mAP = -1




    cfg.MODEL_FILE = 'file/us/model/model_best.pth.tar'
    # cfg.MODEL_FILE = 'file/ws/model/model_best.pth.tar'
    _, _ = test_all(localization_net, cfg, test_loader,
                    test_info, 0, None, cfg.MODEL_FILE)
    utils.save_best_record_thumos(test_info,
                                    os.path.join(cfg.OUTPUT_PATH, "best_results.txt"))
    print(utils.table_format(test_info, cfg.TIOU_THRESH,
            'THUMOS\'14 Performance'))

   

@torch.no_grad()
def test_all(net, cfg, test_loader, test_info, step, writter=None, model_file=None):
    net.eval()

    if model_file:
        print('=> loading model: {}'.format(model_file))
        net.load_state_dict(torch.load(model_file))
        print('=> tesing model...')

    final_res = {'method': '', 'results': {}}

    acc = AverageMeter()

    for data, label, _, vid, vid_num_seg in test_loader:
        data, label = data.cuda(), label.cuda()
        vid_num_seg = vid_num_seg[0].cpu().item()

        video_scores, contrast_pairs, actionness, cas, features = net(data)
        feat_act = contrast_pairs["EA"]
        feat_bkg = contrast_pairs['EB']

        feat_magnitudes_act = torch.mean(
            torch.norm(feat_act, dim=2), dim=1)
        feat_magnitudes_bkg = torch.mean(
            torch.norm(feat_bkg, dim=2), dim=1)
        # feat_magnitudes:[1,52]
        feat_magnitudes = torch.norm(features, p=2, dim=2)

        feat_magnitudes = utils.minmax_norm(feat_magnitudes, max_val=feat_magnitudes_act,
                                            min_val=feat_magnitudes_bkg)
        # feat_magnitudes:[1:52,20]
        feat_magnitudes = feat_magnitudes.repeat(
            (cfg.NUM_CLASSES, 1, 1)).permute(1, 2, 0)

        label_np = label.cpu().data.numpy()
        score_np = video_scores[0].cpu().data.numpy()

        pred_np = np.where(score_np < cfg.CLASS_THRESH, 0, 1)
        correct_pred = np.sum(label_np == pred_np, axis=1)
        acc.update(float(np.sum((correct_pred == cfg.NUM_CLASSES))),
                   correct_pred.shape[0])

        pred = np.where(score_np >= cfg.CLASS_THRESH)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(score_np)])

        cfg.CAS_THRESH = np.arange(0.325, 0.375, 0.025)
        cas_pred = utils.get_pred_activations(cas, pred, cfg)

        aness_pred = utils.get_pred_activations(actionness, pred, cfg)

        # feat_magnitudes_np 代表被选中的类别当中的片段
        feat_magnitudes_np = feat_magnitudes[0].cpu().data.numpy()[:, pred]
        # feat_magnitudes_np:[52,2,1] 52代表片段数，2代表被选择的类别
        feat_magnitudes_np = np.reshape(
            feat_magnitudes_np, (feat_magnitudes.size(1), -1, 1))
        feat_magnitudes_np = utils.upgrade_resolution(
            feat_magnitudes_np, cfg.UP_SCALE)

        proposal_dict = utils.get_proposal_dict_um(
            cas_pred, aness_pred, feat_magnitudes_np, pred, score_np, vid_num_seg, cfg)

        final_proposals = [utils.nms(v, cfg.NMS_THRESH)
                           for _, v in proposal_dict.items()]
        final_res['results'][vid[0]] = utils.result2json(
            final_proposals, cfg.CLASS_DICT)

    json_path = os.path.join(cfg.OUTPUT_PATH, 'result.json')
    json.dump(final_res, open(json_path, 'w'))

    anet_detection = ANETdetection(cfg.GT_PATH, json_path,
                                   subset='test', tiou_thresholds=cfg.TIOU_THRESH,
                                   verbose=False, check_status=False)
    mAP, average_mAP = anet_detection.evaluate()

    if writter:
        writter.add_scalar('Test Performance/Accuracy', acc.avg, step)
        writter.add_scalar('Test Performance/mAP@AVG', average_mAP, step)
        for i in range(cfg.TIOU_THRESH.shape[0]):
            writter.add_scalar(
                'mAP@tIOU/mAP@{:.1f}'.format(cfg.TIOU_THRESH[i]), mAP[i], step)

    test_info["step"].append(step)
    test_info["test_acc"].append(acc.avg)
    test_info["average_mAP"].append(average_mAP)

    for i in range(cfg.TIOU_THRESH.shape[0]):
        test_info["mAP@{:.1f}".format(cfg.TIOU_THRESH[i])].append(mAP[i])
    return test_info['mAP@0.5'][-1], average_mAP


if __name__ == "__main__":
    cfg.MODE = 'test'
    main()
