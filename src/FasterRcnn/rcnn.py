# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FasterRcnn Rcnn network."""
import pickle as pkl
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class DenseNoTranpose(nn.Cell):
    """Dense method"""

    def __init__(self, input_channels, output_channels, weight_init):
        super(DenseNoTranpose, self).__init__()
        self.weight = Parameter(ms.common.initializer.initializer(weight_init, \
                                                                  [input_channels, output_channels], ms.float32))
        self.bias = Parameter(ms.common.initializer.initializer("zeros", \
                                                                [output_channels], ms.float32))

        self.matmul = ops.MatMul(transpose_b=False)
        self.bias_add = ops.BiasAdd()
        self.cast = ops.Cast()
        self.device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"

    def construct(self, x):
        if self.device_type == "Ascend":
            x = self.cast(x, ms.float16)
            weight = self.cast(self.weight, ms.float16)
            output = self.bias_add(self.matmul(x, weight), self.bias)
        else:
            output = self.bias_add(self.matmul(x, self.weight), self.bias)
        return output


class Rcnn(nn.Cell):
    """
    Rcnn subnet.

    Args:
        config (dict) - Config.
        representation_size (int) - Channels of shared dense.
        batch_size (int) - Batchsize.
        num_classes (int) - Class number.
        target_means (list) - Means for encode function. Default: (.0, .0, .0, .0]).
        target_stds (list) - Stds for encode function. Default: (0.1, 0.1, 0.2, 0.2).

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        Rcnn(config=config, representation_size = 1024, batch_size=2, num_classes = 81, \
             target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    """

    def __init__(self,
                 config,
                 representation_size,
                 batch_size,
                 num_classes,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 proto=1,
                 ranking=0.97,
                 ):
        super(Rcnn, self).__init__()
        cfg = config
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.rcnn_loss_cls_weight = Tensor(np.array(cfg.rcnn_loss_cls_weight).astype(self.dtype))
        self.rcnn_loss_reg_weight = Tensor(np.array(cfg.rcnn_loss_reg_weight).astype(self.dtype))
        self.rcnn_fc_out_channels = cfg.rcnn_fc_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.without_bg_loss = config.without_bg_loss
        self.num_classes = num_classes
        self.num_classes_fronted = num_classes
        if self.without_bg_loss:
            self.num_classes_fronted = num_classes - 1
        self.in_channels = cfg.rcnn_in_channels
        self.train_batch_size = batch_size
        self.test_batch_size = cfg.test_batch_size

        self.proto = proto
        self.ranking = ranking
        if self.proto == 1:
            prototype = pkl.load(open('../prototype/prototype.pkl', 'rb'))
        elif self.proto == 2:
            prototype = pkl.load(open('../prototype/prototype_val.pkl', 'rb'))
        prototype = ms.Tensor.from_numpy(prototype[:num_classes-1, :self.rcnn_fc_out_channels])
        bg_proto = prototype.mean(0, True)
        self.prototype = msnp.concatenate((prototype, bg_proto), 0)
        self.proto_mlp = nn.SequentialCell([
            nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels),
            nn.ReLU(),
            nn.Dense(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels),
        ])
        self.topk = ops.TopK()
        self.bitwise_and = ops.BitwiseAnd()
        self.concat = ops.Concat(0)
        self.norm = ops.LpNorm(1)
        self.relu = ops.ReLU()
        self.avgpool = ops.AdaptiveAvgPool2D(1)

        shape_0 = (self.rcnn_fc_out_channels, representation_size)
        weights_0 = ms.common.initializer.initializer("XavierUniform", shape=shape_0[::-1], \
                                                      dtype=self.ms_type).init_data()
        shape_1 = (self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        weights_1 = ms.common.initializer.initializer("XavierUniform", shape=shape_1[::-1], \
                                                      dtype=self.ms_type).init_data()
        self.shared_fc_0 = DenseNoTranpose(representation_size, self.rcnn_fc_out_channels, weights_0)
        self.shared_fc_1 = DenseNoTranpose(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels, weights_1)

        cls_weight = ms.common.initializer.initializer('Normal', shape=[num_classes, self.rcnn_fc_out_channels][::-1],
                                                       dtype=self.ms_type).init_data()
        reg_weight = ms.common.initializer.initializer('Normal', shape=[self.num_classes_fronted * 4,
                                                                        self.rcnn_fc_out_channels][::-1],
                                                       dtype=self.ms_type).init_data()
        self.cls_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes, cls_weight)
        self.reg_scores = DenseNoTranpose(self.rcnn_fc_out_channels, self.num_classes_fronted * 4, reg_weight)

        self.flatten = ops.Flatten()
        self.relu = ops.ReLU()
        self.logicaland = ops.LogicalAnd()
        self.loss_cls = ops.SoftmaxCrossEntropyWithLogits()
        self.loss_bbox = ops.SmoothL1Loss(beta=1.0)
        self.reshape = ops.Reshape()
        self.onehot = ops.OneHot()
        self.greater = ops.Greater()
        self.cast = ops.Cast()
        self.sum_loss = ops.ReduceSum()
        self.tile = ops.Tile()
        self.expandims = ops.ExpandDims()

        self.gather = ops.GatherNd()
        self.argmax = ops.ArgMaxWithValue(axis=1)

        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.value = Tensor(1.0, self.ms_type)

        self.num_bboxes = (cfg.num_expected_pos_stage2 + cfg.num_expected_neg_stage2) * batch_size

        rmv_first = np.ones((self.num_bboxes, self.num_classes_fronted))
        self.rmv_first_tensor = Tensor(rmv_first.astype(self.dtype))

        self.num_bboxes_test = cfg.rpn_max_num * cfg.test_batch_size

        range_max = np.arange(self.num_bboxes_test).astype(np.int32)
        self.range_max = Tensor(range_max)
        self.delta = 0.0001  # Avoid to produce 0

    def construct(self, featuremap, bbox_targets, labels, mask):
        x = self.flatten(featuremap)

        x = self.relu(self.shared_fc_0(x))
        x = self.relu(self.shared_fc_1(x))

        if self.proto:
            proto_w = self.proto_mlp(self.prototype)
            x_cls = msnp.matmul(x, proto_w.T)
        else:
            x_cls = self.cls_scores(x)
        x_reg = self.reg_scores(x)

        if self.training:
            if self.ranking:
                val, idx = self.topk(x_cls, 2)
                y = val.mean(1) if self.ranking == 1 else val[:, 0] * self.ranking
                bg_cls = msnp.where(idx[:, 0] == x_cls.shape[1] - 1, x_cls[:, -1], y)
                x_cls = msnp.concatenate([x_cls[:, :-1], self.expandims(bg_cls, 1)], axis=1)
            bbox_weights = self.cast(self.logicaland(self.greater(labels, 0), mask), ms.int32) * labels
            labels = self.onehot(labels, self.num_classes, self.on_value, self.off_value)
            bbox_targets = self.tile(self.expandims(bbox_targets, 1), (1, self.num_classes_fronted, 1))

            loss, loss_cls, loss_reg, loss_print = self.loss(x_cls, x_reg, bbox_targets, bbox_weights, labels, mask)
            proposal_feats = x
            confi_score = ops.softmax(x_cls)
            loss_inter, loss_triplet = self.domain_adaptive_loss(proposal_feats, confi_score, self.prototype)
            loss = loss + loss_inter + loss_triplet
            out = (loss, loss_cls, loss_reg, loss_print)
        else:
            out = (x_cls, (x_cls / self.value), x_reg, x_cls)

        return out

    def loss(self, cls_score, bbox_pred, bbox_targets, bbox_weights, labels, weights):
        """Loss method."""
        loss_print = ()
        loss_cls, _ = self.loss_cls(cls_score, labels)

        weights = self.cast(weights, self.ms_type)
        loss_cls = loss_cls * weights
        loss_cls = self.sum_loss(loss_cls, (0,)) / self.sum_loss(weights, (0,))

        bbox_weights = self.cast(self.onehot(bbox_weights, self.num_classes, self.on_value, self.off_value),
                                 self.ms_type)
        if self.without_bg_loss:
            bbox_weights = bbox_weights[:, 1:] * self.rmv_first_tensor
        else:
            bbox_weights = bbox_weights * self.rmv_first_tensor
        pos_bbox_pred = self.reshape(bbox_pred, (self.num_bboxes, -1, 4))
        loss_reg = self.loss_bbox(pos_bbox_pred, bbox_targets)
        loss_reg = self.sum_loss(loss_reg, (2,))
        loss_reg = loss_reg * bbox_weights
        if self.without_bg_loss:
            loss_reg = loss_reg / (self.sum_loss(weights, (0,)) + self.delta)
        else:
            loss_reg = loss_reg / (self.sum_loss(weights, (0,)))
        loss_reg = self.sum_loss(loss_reg, (0, 1))

        loss = self.rcnn_loss_cls_weight * loss_cls + self.rcnn_loss_reg_weight * loss_reg
        loss_print += (loss_cls, loss_reg)

        return loss, loss_cls, loss_reg, loss_print

    def domain_adaptive_loss(self, proposal_feats, confi_score, prototype, epsilon=1e-6):

        proposal_proto = (self.expandims(confi_score, 2) * self.expandims(proposal_feats, 1)).sum(0) / self.expandims(confi_score.sum(0) + epsilon, 1)

        ### prototype alignment
        ### inter loss
        loss_inter = inter_loss(confi_score)

        d_an = ops.norm(self.expandims(proposal_proto, 1) - self.expandims(self.concat((proposal_proto, prototype)), 0), 2)
        indices = self.expandims(msnp.arange(d_an.shape[0]), 1)
        d_ap = ops.gather_elements(d_an, 1, indices)
        mat = self.relu(d_ap - d_an + 1)
        mask = ops.ones_like(mat)
        mask = ops.tensor_scatter_elements(mask, indices, self.cast(ops.zeros_like(indices), ms.float32), 1)
        mask = ops.tensor_scatter_elements(mask, indices + mat.shape[0], self.cast(ops.zeros_like(indices), ms.float32), 1)
        mat = mat * mask
        triple_loss = mat.mean()
        '''for i in range(proposal_proto.shape[0]):
            c_proto1 = self.unsqueeze(proposal_proto[i, :], 0)
            s_proto1 = self.unsqueeze(prototype[i, :], 0)
            if i == 0:
                neg_proto = self.concat((proposal_proto[1:], prototype[1:]))
            elif i == proposal_proto.shape[0]-1:
                neg_proto = self.concat((proposal_proto[:-1], prototype[:-1]))
            else:
                neg_proto = self.concat((self.concat((proposal_proto[0:i], proposal_proto[i+1:i+2])),
                                       self.concat((prototype[0:i], prototype[i+1:i+2]))))
            anchor_proto = c_proto1
            pos_proto = s_proto1
            d_ap = self.norm(anchor_proto - pos_proto)
            d_an = self.norm(anchor_proto - neg_proto)
            triple_loss = triple_loss + self.relu(d_ap - d_an + 1).mean()

        triple_loss /= proposal_proto.shape[0]'''

        return loss_inter, triple_loss

def inter_loss(p):
    epsilon = 1e-5
    return -1 * (p * msnp.log(p + epsilon)).sum() / p.shape[0]