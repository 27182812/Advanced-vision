# Advanced-vision
注：本课程报告主要来源于本科参与的大创项目和本科实验室项目

代码中的./调过参数的py-faster-rcnn/py-faster-rcnn/data/faster_rcnn_models/下的模型较大未上传，请自行下载
或者参考完整版的百度网盘：
https://pan.baidu.com/s/1uRe7tg_S5IL4AzA3lp9Ilw 
提取码：nicm 
复制这段内容后打开百度网盘手机App，操作更方便哦

# 系统硬件环境
1）服务器：酷睿i7-6700HQ、NVIDIA GTX 1070 8G、内存8G，硬盘60G以上配置的高性能计算机。
2）操作系统：Ubuntu Linux 16.04 LTS 。
# 系统软件环境
1）操作系统：Ubuntu Linux 16.04 LTS；
2）语言环境：python2.7、G++5.4.0；
3）深度学习框架：Caffe；
4）其他主要软件运行库：CUDA toolkit 8.0、cuDNN 5.1、OpenCV 3.1等

### 运行代码前需要配置好环境，包括：
1）Ubuntu系统安装
2）CUDA8.0安装
3）cuDNN5.1安装
4）OpenCV3.1安装
5）python2.7安装
6）caffe安装
7）faster-rcnn安装
8）labelImg-master安装

# faster-rcnn配置
下载ImageNet数据集下预训练得到的模型参数（用来初始化）解压，然后将该文件放在py-faster-rcnn\data下
（1）修改训练相关参数
修改训练文件stage1_fast_rcnn_solver30k40k.pt等4个solver文件，修改base_lr: 0.0001，使训练第一阶段学习率减小至0.0001。
修改文件config.py，修改的部分代码如下：
__C.TRAIN.SCALES = (300,)
# 训练时图片的缩放参数，将最短边缩放至300
__C.TRAIN.MAX_SIZE = 500
# 缩放时如果最长边大于500，则以最大边为基准将最大边缩放至500
__C.TRAIN.BATCH_SIZE = 128
# 同时处理的ROI(regions of interest)个数为128
__C.TEST.SCALES = (3600,)
# 测试时图片的缩放参数，将最短边缩放至3600
__C.TEST.MAX_SIZE = 3600
# 缩放时如果最长边大于3600，则以最大边为基准将最大边缩放至3600
__C.TEST.RPN_POST_NMS_TOP_N = 6000
# 将最可能是识别目标的前6000个区域交给后续识别网络处理，由于一张图片棉蚜基数大，所以此参数需要尽可能大
（2）修改代码
修改训练配置文件里的代码stage1_fast_rcnn_train.pt、stage1_rpn_train.pt、stage2_fast_rcnn_train.pt、stage2_rpn_train.pt，将类别数量
param_str: "'num_classes': "数字设为2（蚜虫类和背景类）
修改faster_rcnn_test.pt，修改部分如下
layer {
  name: "cls_score"
  type: "InnerProduct"
  bottom: "fc7"
  top: "cls_score"
  inner_product_param {
    num_output: 2 #蚜虫和背景类
  }
}
layer {
  name: "bbox_pred"
  type: "InnerProduct"
  bottom: "fc7"
  top: "bbox_pred"
  inner_product_param {
    num_output: 8 #类别数乘与4
  }
}
修改lib/datasets/imdb.py，append_flipped_images()函数在一行代码为 boxes[:, 2] = widths[i] - oldx1 - 1下加入代码：
for b in range(len(boxes)):
  if boxes[b][2]< boxes[b][0]:
    boxes[b][0] = 0              
然后进入py-faster-rcnn/lib/datasets/pascal_voc.py修改
class pascal_voc(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'voc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'VOC' + self._year)
        self._classes = ('__background__', # always index 0
                         'bug')
然后进入py-faster-rcnn/lib/datasets/imdb.py修改append_flipped_images(self)函数代码：
def append_flipped_images(self):
        num_images = self.num_images
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            print boxes[:, 0]
            boxes[:, 2] = widths[i] - oldx1 - 1
            print boxes[:, 0]
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes' : boxes,
                     'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     'gt_classes' : self.roidb[i]['gt_classes'],
                     'flipped' : True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2

