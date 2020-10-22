from torch.utils.data import Dataset

DATASET_PATH = "/mnt/data/yg"


class VOC2012(Dataset):
    def __init__(self, is_train=True, is_aug=True):
        """

        :param is_train: 是训练集还是测试集
        :param is_aug: 是否进行数据增广
        """
        self.filenames = [] #存储数据集的文件名字
        if is_train:
            with open(DATASET_PATH+"ImageSets/Main/train.txt", "r") as f: # 调用包含数据集名称的txt文件
                self.filenames = [x.strip() for x in f]
        else:
            with open(DATASET_PATH + "ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]




