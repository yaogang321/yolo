import torch.nn as nn
import torch


def calculate_iou(bbox1, bbox2):
    ''' bbox1和bbox2的格式x1,y1,x2,y2和x3,y3,x4,y4'''

    # 排除没有交集的情况
    if bbox1[0] > bbox2[2] or bbox1[1] > bbox2[3] or bbox2[0] > bbox1[2] or bbox2[1] > bbox1[3]:
        pass
    else:
        # 对坐标排序，使用中间的两个数
        List1 = [bbox1[0], bbox1[2], bbox2[0], bbox2[2]]
        List2 = [bbox1[1], bbox1[3], bbox2[1], bbox2[3]]
        List1.sort()
        List2.sort()
        # 计算交集Intersection和area1,area2的面积，交集的面积等于中间的坐标差的乘积
        Intersection = (List1[2] - List1[1]) * (List2[2] - List1[1])
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        # 计算iou
        if Intersection > 0:
            return Intersection / (area1 + area2 - Intersection)
        else:
            return 0


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, pred, labels):
        """

        :param pred:(batchsize,30,7,7)的网络输出数据
        :param labels:(batchsize,30,7,7)的样本标签数据
        :return:当前批次样本的平均损失
        """

        grid_x, grid_y = 7, 7
        n_batch = labels.size()[0]
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失

        for i in n_batch:
            for m in range(grid_x):
                for n in range(grid_y):
                    if labels[i, 4, m, n] == 1:
                        # 对于labels而言,0-4和5-9的数据是一样的
                        # 转换数据格式，从px,py,w,h,转换成xyxy,px和py是m,n的一个格子里的相对坐标，计算过后是相对于全图的相对坐标
                        pred_box1 = ((pred[i, 0, m, n] + m) / grid_x - pred[i, 2, m, n] / 2,
                                     (pred[i, 1, m, n] + n) / grid_y - pred[i, 3, m, n] / 2,
                                     (pred[i, 0, m, n] + m) / grid_x + pred[i, 2, m, n] / 2,
                                     (pred[i, 1, m, n] + n) / grid_y + pred[i, 3, m, n] / 2)
                        pred_box2 = ((pred[i, 5, m, n] + m) / grid_x - pred[i, 7, m, n] / 2,
                                     (pred[i, 6, m, n] + n) / grid_y - pred[i, 8, m, n] / 2,
                                     (pred[i, 5, m, n] + m) / grid_x + pred[i, 7, m, n] / 2,
                                     (pred[i, 6, m, n] + n) / grid_y + pred[i, 8, m, n] / 2)
                        ground_box = ((labels[i, 0, m, n] + m) / grid_x - labels[i, 2, m, n] / 2,
                                      (labels[i, 1, m, n] + m) / grid_y - labels[i, 3, m, n] / 2,
                                      (labels[i, 0, m, n] + m) / grid_x + labels[i, 2, m, n] / 2,
                                      (labels[i, 1, m, n] + m) / grid_x + labels[i, 3, m, n] / 2,)

                        iou1 = calculate_iou(pred_box1, ground_box)
                        iou2 = calculate_iou(pred_box2, ground_box)

                        if(iou1 >= iou2):
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2) \
                                                         + torch.sum(
                                        (pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (pred[i, 4, m, n] - iou1) ** 2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 9, m, n] - iou2) ** 2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2) \
                                                         + torch.sum(
                                        (pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2))
                            obj_confi_loss = obj_confi_loss + (pred[i, 9, m, n] - iou2) ** 2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n] - iou1) ** 2)
                        class_loss = class_loss + torch.sum((pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)
                    else: # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i, [4, 9], m, n] ** 2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss / n_batch

