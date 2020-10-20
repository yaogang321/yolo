def calculate_iou(bbox1, bbox2):
    ''' bbox1和bbox2的格式x1,y1,x2,y2和x3,y3,x4,y4'''

    # 排除没有交集的情况
    if bbox1[0]>bbox2[2] or bbox1[1]>bbox2[3] or bbox2[0]>bbox1[2] or bbox2[1]>bbox1[3]:
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
            return Intersection/(area1+area2-Intersection)
        else:
            return 0


def