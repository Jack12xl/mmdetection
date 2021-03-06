import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from ..registry import HEADS

from mmcv.cnn import xavier_init
from mmcv.runner import load_checkpoint

from mmdet.models.utils import YoloConvLayer

from mmdet.core import (multiclass_nms, force_fp32)

from mmdet.ops.nms import nms_wrapper


_EPSILON = 1e-6


@HEADS.register_module
class YoloHead(nn.Module):

    num_scales = 3
    num_classes_no_bkg = 1
    num_classes_w_bkg = num_classes_no_bkg + 1
    num_classes = num_classes_w_bkg
    num_anchors_per_scale = 3
    num_attrib = num_classes_no_bkg + 5
    last_layer_dim = num_anchors_per_scale * num_attrib
    in_channels = [512, 256, 128]
    out_channels = [1024, 512, 256]
    scales = ['l', 'm', 's']
    strides = [32, 16, 8]
    # scale_params = [(1024, 'l', 32), (512, 'm', 16), (256, 's', 8)]  #out_channel, scale, stride

    anchor_base_sizes = [[(116, 90), (156, 198), (373, 326)],
                         [(30, 61), (62, 45), (59, 119)],
                         [(10, 13), (16, 30), (33, 23)],
                         ]

    def __init__(self):
        super(YoloHead, self).__init__()
        self.convs_bridge = nn.ModuleList()
        self.convs_final = nn.ModuleList()
        for i_scale in range(self.num_scales):
            in_c = self.in_channels[i_scale]
            out_c = self.out_channels[i_scale]
            conv_bridge = YoloConvLayer(in_c, out_c, 3)
            conv_final = nn.Conv2d(out_c, self.last_layer_dim, 1, bias=True)

            self.convs_bridge.append(conv_bridge)
            self.convs_final.append(conv_final)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, feats):
        assert len(feats) == self.num_scales
        results = []
        for i in range(self.num_scales):
            x = feats[i]
            x = self.convs_bridge[i](x)
            out = self.convs_final[i](x)
            results.append(out)

        return tuple(results),

    @force_fp32(apply_to=('results_raw',))
    def get_bboxes(self, results_raw, img_metas, cfg, rescale=False):
        result_list = []
        for img_id in range(len(img_metas)):
            result_raw_list = [
                results_raw[i][img_id].detach() for i in range(self.num_scales)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(result_raw_list, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    @staticmethod
    def _get_anchors_grid_xy(num_grid_h, num_grid_w, stride, device='cpu'):
        grid_x = torch.arange(num_grid_w, dtype=torch.float, device=device).repeat(num_grid_h, 1)
        grid_y = torch.arange(num_grid_h, dtype=torch.float, device=device).repeat(num_grid_w, 1)

        grid_x = grid_x.unsqueeze(0) * stride
        grid_y = grid_y.t().unsqueeze(0) * stride

        return grid_x, grid_y

    def get_bboxes_single(self, results_raw, scale_factor, cfg, rescale=False):
        assert len(results_raw) == self.num_scales
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        for i_scale in range(self.num_scales):
            result_raw = results_raw[i_scale]
            num_grid_h = result_raw.size(1)
            num_grid_w = result_raw.size(2)

            prediction_raw = result_raw.view(self.num_anchors_per_scale,
                                             self.num_attrib,
                                             num_grid_h,
                                             num_grid_w).permute(0, 2, 3, 1).contiguous()

            # grid x y offset, with stride step included

            stride = self.strides[i_scale]

            grid_x, grid_y = self._get_anchors_grid_xy(num_grid_h, num_grid_w, stride, result_raw.device)

            # Get outputs x, y

            x_center_pred = torch.sigmoid(prediction_raw[..., 0]) * stride + grid_x # Center x
            y_center_pred = torch.sigmoid(prediction_raw[..., 1]) * stride + grid_y # Center y

            anchors = torch.tensor(self.anchor_base_sizes[i_scale], device=result_raw.device, dtype=torch.float32)

            anchor_w = anchors[:, 0:1].view((-1, 1, 1))
            anchor_h = anchors[:, 1:2].view((-1, 1, 1))

            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w  # Width
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h  # Height

            x1_pred = x_center_pred - w_pred / 2
            y1_pred = y_center_pred - h_pred / 2

            x2_pred = x_center_pred + w_pred / 2
            y2_pred = y_center_pred + h_pred / 2

            bbox_pred = torch.stack((x1_pred, y1_pred, x2_pred, y2_pred), dim=3).view(
                (-1, 4))  # cxcywh
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(-1)  # Conf
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(-1, self.num_classes_no_bkg)  # Cls pred one-hot.

            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if multi_lvl_conf_scores.size(0) == 0:
            return torch.zeros((0, 5)), torch.zeros((0,))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0], 1)
        multi_lvl_cls_scores = torch.cat([padding, multi_lvl_cls_scores], dim=1)

        det_bboxes, det_labels = multiclass_nms(multi_lvl_bboxes, multi_lvl_cls_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img, score_factors=multi_lvl_conf_scores)

        return det_bboxes, det_labels

    @force_fp32(apply_to=('preds_raw',))
    def loss(self,
             preds_raw,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        losses = {'loss_xy': 0, 'loss_wh': 0, 'loss_conf': 0, 'loss_cls': 0}

        for img_id in range(len(img_metas)):
            pred_raw_list = []
            anchor_grids = []
            for i_scale in range(self.num_scales):
                pred_raw = preds_raw[i_scale][img_id]
                num_grid_h = pred_raw.size(1)
                num_grid_w = pred_raw.size(2)
                pred_raw = pred_raw.view(self.num_anchors_per_scale,
                                         self.num_attrib,
                                         num_grid_h,
                                         num_grid_w).permute(0, 2, 3, 1).contiguous()
                anchor_grid = self.get_anchors(num_grid_h, num_grid_w, i_scale, device=pred_raw.device)

                pred_raw_list.append(pred_raw)
                anchor_grids.append(anchor_grid)

            gt_bboxes_per_img = gt_bboxes[img_id]
            gt_labels_per_img = gt_labels[img_id]

            ignore_thresh = cfg.get('ignore_thresh', 0.5)
            one_hot_smoother = cfg.get('one_hot_smoother', 0.)
            xy_use_logit = cfg.get('xy_use_logit', False)
            balance_conf = cfg.get('balance_conf', False)

            gt_t_across_scale, negative_mask_across_scale = \
                self._preprocess_target_single_img(gt_bboxes_per_img,
                                                   gt_labels_per_img,
                                                   anchor_grids,
                                                   ignore_thresh=ignore_thresh,
                                                   one_hot_smoother=one_hot_smoother,
                                                   xy_use_logit=xy_use_logit)

            losses_per_img = self.loss_single(pred_raw_list,
                                              gt_t_across_scale,
                                              negative_mask_across_scale,
                                              xy_use_logit=xy_use_logit,
                                              balance_conf=balance_conf)

            for loss_term in losses:
                term_no_loss = loss_term[5:]
                losses[loss_term] += losses_per_img[term_no_loss]

        return losses

    def loss_single(self, preds_raw, gts_t, neg_masks, xy_use_logit=False, balance_conf=False):

        losses = {'xy': 0, 'wh': 0, 'conf': 0, 'cls': 0}

        for i_scale in range(self.num_scales):
            pred_raw = preds_raw[i_scale]
            gt_t = gts_t[i_scale]
            neg_mask = neg_masks[i_scale].float()
            pos_mask = gt_t[..., 4]
            pos_and_neg_mask = neg_mask + pos_mask
            pos_mask = pos_mask.unsqueeze(dim=-1)
            if torch.max(pos_and_neg_mask) > 1.:
                print("Warning: pos_and_neg_mask gives max of more than 1. Some bugs in the program.")
                pos_and_neg_mask = pos_and_neg_mask.clamp(min=0., max=1.)
            # ignore_mask = (1. - pos_and_neg_mask).clamp(min=0)

            pred_t_xy = pred_raw[..., :2]
            pred_t_wh = pred_raw[..., 2:4]
            pred_conf = pred_raw[..., 4]
            pred_label = pred_raw[..., 5:]

            gt_t_xy = gt_t[..., :2]
            gt_t_wh = gt_t[..., 2:4]
            gt_conf = gt_t[..., 4]
            gt_label = gt_t[..., 5:]

            if balance_conf:
                num_pos_gt = max(int(torch.sum(gt_conf)), 1)
                grid_size = list(gt_conf.size())
                num_total_grids = 1
                for s in grid_size:
                    num_total_grids *= s
                pos_weight = num_total_grids / num_pos_gt
                conf_loss_weight = 1 / pos_weight
            else:
                pos_weight = 1
                conf_loss_weight = 1

            pos_weight = gt_label.new_tensor(pos_weight)

            losses_cls = F.binary_cross_entropy_with_logits(pred_label, gt_label, reduction='none')

            losses_cls *= pos_mask

            losses_conf = F.binary_cross_entropy_with_logits(pred_conf,
                                                             gt_conf,
                                                             reduction='none',
                                                             pos_weight=pos_weight
                                                             ) * pos_and_neg_mask * conf_loss_weight

            if xy_use_logit:
                losses_xy = F.mse_loss(pred_t_xy, gt_t_xy, reduction='none') * pos_mask * 2
            else:
                losses_xy = F.binary_cross_entropy_with_logits(pred_t_xy, gt_t_xy, reduction='none') * pos_mask * 2

            losses_wh = F.mse_loss(pred_t_wh, gt_t_wh, reduction='none') * pos_mask * 2

            losses['cls'] += torch.sum(losses_cls)
            losses['conf'] += torch.sum(losses_conf)
            losses['xy'] += torch.sum(losses_xy)
            losses['wh'] += torch.sum(losses_wh)

        return losses

    def _preprocess_target_single_img(self, gt_bboxes, gt_labels, anchor_grids, ignore_thresh, one_hot_smoother=0, xy_use_logit=False):
        """Generate matching bounding box prior and converted GT."""
        assert gt_bboxes.size(1) == 4
        assert gt_bboxes.size(0) == gt_labels.size(0)
        assert len(anchor_grids) == self.num_scales

        # iou_to_match_across_scale is a list of 3D tensors (in uint8).
        # each tensor has dimension of AxWxH
        # where A is the number of anchors in this scale,
        # W and H is the size of the grid in this scale
        # each element of the tensor represents whether the prediction should have generate non-objectness
        negative_mask_across_scale = []

        gt_t_across_scale = []

        for anchor_grid in anchor_grids:
            negative_mask_size = list(anchor_grid.size())[:-1]
            negative_mask = anchor_grid.new_ones(negative_mask_size, dtype=torch.uint8)
            negative_mask_across_scale.append(negative_mask)
            gt_t_size = negative_mask_size + [self.num_attrib]
            gt_t = anchor_grid.new_zeros(gt_t_size)
            gt_t_across_scale.append(gt_t)

        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):

            # convert to cxywh
            gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
            gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
            gt_w = gt_bbox[2] - gt_bbox[0]
            gt_h = gt_bbox[3] - gt_bbox[1]
            gt_bbox_cxywh = torch.stack((gt_cx, gt_cy, gt_w, gt_h))

            iou_to_match_across_scale = []

            grid_coord_across_scale = []

            for i_scale in range(self.num_scales):
                stride = self.strides[i_scale]
                anchor_grid = anchor_grids[i_scale]
                iou_gt_anchor = iou_multiple_to_one(anchor_grid, gt_bbox_cxywh, center=True)
                negative_mask = (iou_gt_anchor <= ignore_thresh)
                w_grid = int(gt_cx // stride)
                h_grid = int(gt_cy // stride)
                iou_to_match = list(iou_gt_anchor[:, h_grid, w_grid])

                # AND operation, only negative when all are negative
                negative_mask_across_scale[i_scale] *= negative_mask
                iou_to_match_across_scale.extend(iou_to_match)
                grid_coord_across_scale.append((h_grid, w_grid))

            itmas = iou_to_match_across_scale  # make the name shorter
            max_match_iou_idx = max(range(len(itmas)), key=lambda x: itmas[x])  # get idx of max iou
            match_scale = max_match_iou_idx // self.num_anchors_per_scale
            match_anchor_in_scale = max_match_iou_idx - match_scale * self.num_anchors_per_scale
            match_grid_h, match_grid_w = grid_coord_across_scale[match_scale]

            match_anchor_w, match_anchor_h = self.anchor_base_sizes[match_scale][match_anchor_in_scale]

            gt_tw = torch.log((gt_w / match_anchor_w).clamp(min=_EPSILON))
            gt_th = torch.log((gt_h / match_anchor_h).clamp(min=_EPSILON))

            gt_tcx = (gt_cx / self.strides[match_scale] - match_grid_w).clamp(_EPSILON, 1 - _EPSILON)
            gt_tcy = (gt_cy / self.strides[match_scale] - match_grid_h).clamp(_EPSILON, 1 - _EPSILON)

            if xy_use_logit:
                gt_tcx = torch.log(gt_tcx / (1. - gt_tcx))  # inverse of sigmoid
                gt_tcy = torch.log(gt_tcy / (1. - gt_tcy))  # inverse of sigmoid

            gt_t_bbox = torch.stack((gt_tcx, gt_tcy, gt_tw, gt_th))

            # raw label start from 1, need to minus 1 to compensate that
            gt_label_one_hot = F.one_hot(gt_label - 1, num_classes=self.num_classes_no_bkg).float()

            gt_label_one_hot = gt_label_one_hot * (1 - one_hot_smoother) + one_hot_smoother / self.num_classes_no_bkg

            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h, match_grid_w, :4] = gt_t_bbox
            # if gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h, match_grid_w, 4] == 1.:
            #     print("Warning: Target confidence has been assigned to 1 already.")
            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h, match_grid_w, 4] = 1.
            gt_t_across_scale[match_scale][match_anchor_in_scale, match_grid_h, match_grid_w, 5:] = gt_label_one_hot

            # although iou fall under a certain thres, since it has max iou, still positive
            negative_mask_across_scale[match_scale][match_anchor_in_scale, match_grid_h, match_grid_w] = 0

        return gt_t_across_scale, negative_mask_across_scale

    def get_anchors(self, num_grid_h, num_grid_w, scale, device='cpu'):
        assert scale in range(self.num_scales)
        anchors = torch.tensor(self.anchor_base_sizes[scale], device=device, dtype=torch.float32)
        num_anchors = anchors.size(0)
        stride = self.strides[scale]

        grid_x, grid_y = self._get_anchors_grid_xy(num_grid_h, num_grid_w, stride, device)

        grid_x += stride / 2  # convert to center of the grid, that is, making the raw prediction 0, not -inf
        grid_y += stride / 2
        grid_x = grid_x.expand((num_anchors, -1, -1))
        grid_y = grid_y.expand((num_anchors, -1, -1))

        anchor_w = anchors[:, 0:1].view((-1, 1, 1))
        anchor_h = anchors[:, 1:2].view((-1, 1, 1))
        anchor_w = anchor_w.expand((-1, num_grid_h, num_grid_w))
        anchor_h = anchor_h.expand((-1, num_grid_h, num_grid_w))

        anchor_cxywh = torch.stack((grid_x, grid_y, anchor_w, anchor_h), dim=3)

        return anchor_cxywh


def iou_multiple_to_one(bboxes1, bbox2, center=False, zero_center=False):
    """Calculate the IOUs between bboxes1 (multiple) and bbox2 (one).
    :param
      bboxes1: (Tensor) A n-D tensor representing first group of bboxes.
        The dimension is (..., 4).
        The lst dimension represent the bbox, with coordinate (x, y, w, h) or (cx, cy, w, h).
      bbox2: (Tensor) A 1D tensor representing the second bbox.
        The dimension is (4,).
      center: (bool). Whether the bboxes are in format (cx, cy, w, h).
      zero_center: (bool). Whether to align two bboxes so their center is aligned.
    :return
      iou_: (Tensor) A (n-1)-D tensor representing the IOUs.It has one less dim than bboxes1"""

    epsilon = 1e-6

    x1 = bboxes1[..., 0]
    y1 = bboxes1[..., 1]
    w1 = bboxes1[..., 2]
    h1 = bboxes1[..., 3]

    x2 = bbox2[0]
    y2 = bbox2[1]
    w2 = bbox2[2]
    h2 = bbox2[3]

    area1 = w1 * h1
    area2 = w2 * h2

    if zero_center:
        w_intersect = torch.min(w1, w2).clamp(min=0)
        h_intersect = torch.min(h1, h2).clamp(min=0)
    else:
        if center:
            x1 = x1 - w1 / 2
            y1 = y1 - h1 / 2
            x2 = x2 - w2 / 2
            y2 = y2 - h2 / 2
        right1 = (x1 + w1)
        right2 = (x2 + w2)
        top1 = (y1 + h1)
        top2 = (y2 + h2)
        left1 = x1
        left2 = x2
        bottom1 = y1
        bottom2 = y2
        w_intersect = (torch.min(right1, right2) - torch.max(left1, left2)).clamp(min=0)
        h_intersect = (torch.min(top1, top2) - torch.max(bottom1, bottom2)).clamp(min=0)
    area_intersect = h_intersect * w_intersect

    iou_ = area_intersect / (area1 + area2 - area_intersect + epsilon)

    return iou_


def multiclass_nms_custom(multi_bboxes,
                          multi_scores,
                          score_thr,
                          nms_cfg,
                          max_num=-1,
                          score_factors=None):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    multi_scores_no_bkg = multi_scores[:, 1:]
    _, class_ids = torch.max(multi_scores_no_bkg, dim=1)
    for i in range(0, num_classes - 1):
        cls_inds = (class_ids == i)
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        # _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            selected_score_factors = score_factors[cls_inds]
            _scores = selected_score_factors
        else:
            raise ValueError("score factor should be assigned with conf score")
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i,
                                           dtype=torch.long)
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels