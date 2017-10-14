#-*- coding:utf8 -*-
import glob
import numpy
import cv2
from PIL import Image
import numpy as np
import time
import pdb
import unicodedata
import re
import chars

__all__ = (
    'CHARSET',
    'sigmoid',
    'softmax',
)

OUTPUT_SHAPE = (100, 180)#60:一张图片的高度，180:一张图片最大字符长度
CHARSET = ['', ' ', '~', '', u'!', u'"', u'#', u'$', u'%', u'&', u"'", u'(', u')', u'*', u'+', u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':', u';', u'<', u'=', u'>', u'?', u'@', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'[', u'\\', u']', u'^', u'_', u'`', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'{', u'|', u'}', u'\xa1', u'\xa2', u'\xa3', u'\xa7', u'\xa9', u'\xab', u'\xae', u'\xb0', u'\xb6', u'\xbb', u'\xbf', u'\xc0', u'\xc1', u'\xc2', u'\xc3', u'\xc4', u'\xc6', u'\xc7', u'\xc8', u'\xc9', u'\xca', u'\xcb', u'\xcc', u'\xcd', u'\xce', u'\xcf', u'\xd2', u'\xd3', u'\xd4', u'\xd5', u'\xd6', u'\xd9', u'\xda', u'\xdb', u'\xdc', u'\xdf', u'\xe0', u'\xe1', u'\xe2', u'\xe3', u'\xe4', u'\xe6', u'\xe7', u'\xe8', u'\xe9', u'\xea', u'\xeb', u'\xec', u'\xed', u'\xee', u'\xef', u'\xf2', u'\xf3', u'\xf4', u'\xf5', u'\xf6', u'\xf7', u'\xf9', u'\xfa', u'\xfb', u'\xfc', u'\xff', u'\u0152', u'\u0153', u'\u0178', u'\u2020', u'\u2021', u'\u2022', u'\u2023', u'\u2039', u'\u203a', u'\u20ac', u'\u2219', u'\u25aa', u'\u25ab', u'\u25e6', u'\u4f4f', u'\u9662', u'\u53f7', u'\u5e74', u'\u9f84', u'\u6027', u'\u522b', u'\u6837', u'\u672c', u'\u68c0', u'\u9a8c', u'\u76ee', u'\u7684', u'\u9879', u'\u4ee3', u'\u6d4b', u'\u5b9a', u'\u503c', u'\u5355', u'\u4f4d', u'\u53c2', u'\u8003', u'\u65e5', u'\u671f', u'\u7537', u'\u672f', u'\u524d', u'\u56db', u'\u6297', u'\u6885', u'\u6bd2', u'\u87ba', u'\u65cb', u'\u4f53', u'\u7279', u'\u5f02', u'\u5f31', u'\u9633', u'\u9634', u'\u4e19', u'\u578b', u'\u809d', u'\u708e', u'\u4e59', u'\u8868', u'\u9762', u'\u539f', u'\u4eba', u'\u514d', u'\u75ab', u'\u7f3a', u'\u9677', u'\u75c5', u'\u7caa', u'\u4fbf', u'\u5e38', u'\u89c4', u'\u5bc4', u'\u751f', u'\u866b', u'\u5375', u'\u955c', u'\u8ba1', u'\u6570', u'\u989c', u'\u8272', u'\u9ec4', u'\u72b6', u'\u8f6f', u'\u86d4', u'\u672a', u'\u67e5', u'\u89c1', u'\u94a9', u'\u97ad', u'\u541e', u'\u566c', u'\u7ec6', u'\u80de', u'\u6cb9', u'\u6ef4', u'\u590f', u'\u96f7', u'\u767b', u'\u6c0f', u'\u8113', u'\u9709', u'\u83cc', u'\u5b62', u'\u5b50', u'\u7ea2', u'\u767d', u'\u4e0d', u'\u6d88', u'\u5316', u'\u98df', u'\u7269', u'\u9690', u'\u8840', u'\u8bd5', u'\u8111', u'\u810a', u'\u6db2', u'\u623f', u'\u6709', u'\u6838', u'\u6e05', u'\u6670', u'\u5ea6', u'\u900f', u'\u660e', u'\u5916', u'\u89c2', u'\u65e0', u'\u6f58', u'\u808c', u'\u9499', u'\u86cb', u'\u91cf', u'\u5c3f', u'\u4e9a', u'\u785d', u'\u9178', u'\u76d0', u'\u8d28', u'\u916e', u'\u6d4a', u'\u80c6', u'\u7d20', u'\u8461', u'\u8404', u'\u7cd6', u'\u7ba1', u'\u7c7b', u'\u9175', u'\u6bcd', u'\u7c98', u'\u4e1d', u'\u7ed3', u'\u6676', u'\u6bd4', u'\u91cd', u'\u4e0a', u'\u76ae', u'\u7535', u'\u5bfc', u'\u7387', u'\u6de1', u'\u8102', u'\u9176', u'\u6b63', u'\u80bf', u'\u7624', u'\u6807', u'\u5fd7', u'\u94c1', u'\u7532', u'\u80ce', u'\u764c', u'\u80da', u'\u94fe', u'\u4e09', u'\u7cfb', u'\u5fc3', u'\u80be', u'\u89e3', u'\u6781', u'\u4f4e', u'\u5bc6', u'\u56fa', u'\u9187', u'\u7518', u'\u916f', u'\u673a', u'\u78f7', u'\u9ad8', u'\u7403', u'\u817a', u'\u82f7', u'\u8131', u'\u6c28', u'\u6c2f', u'\u94a0', u'\u5929', u'\u95e8', u'\u51ac', u'\u57fa', u'\u8f6c', u'\u79fb', u'\u603b', u'\u5ca9', u'\u85fb', u'\u76f4', u'\u63a5', u'\u95f4', u'\u94be', u'\u6c41', u'\u6c2e', u'\u78b1', u'\u8c37', u'\u9170', u'\u812f', u'\u4e8c', u'\u80bd', u'\u9150', u'\u53cd', u'\u5e94', u'\u8c31', u'\u6fc0', u'\u540c', u'\u5de5', u'\u6d3b', u'\u7f9f', u'\u4e01', u'\u6c22', u'\u4e73', u'\u4e94', u'\u5206', u'\u55dc', u'\u7c92', u'\u5c0f', u'\u677f', u'\u538b', u'\u79ef', u'\u6dcb', u'\u5df4', u'\u5e73', u'\u5747', u'\u5e03', u'\u5bbd', u'\u4e2d', u'\u6d53', u'\u542b', u'\u51dd', u'\u529f', u'\u80fd', u'\u805a', u'\u56fd', u'\u9645', u'\u51c6', u'\u65f6', u'\u5bf9', u'\u7167', u'\u79d2', u'\u7ea4', u'\u7ef4', u'\u90e8', u'\u8d85', u'\u654f', u'\u5feb', u'\u901f', u'\u6cd5', u'\u80f1', u'\u6291', u'\u6ee4', u'\u8fc7', u'\u6697', u'\u6c14', u'\u6790', u'\u5168', u'\u5b9e', u'\u5269', u'\u4f59', u'\u5438', u'\u5165', u'\u6c27', u'\u6d41', u'\u78b3', u'\u6839', u'\u6e29', u'\u7ea0', u'\u540e', u'\u79bb', u'\u9699', u'\u9971', u'\u548c', u'\u6e17', u'\u8f93', u'\u68d5', u'\u4e8e', u'\u9650', u'\u9541', u'\u5957', u'\u7898', u'\u6e38', u'\u4fc3', u'\u8bf4', u'\u5973', u'\u81ea', u'\u8d39', u'\u8910', u'\u6c89', u'\u964d', u'\u52a8', u'\u8109', u'\u6dc0', u'\u7c89', u'\u5f15', u'\u6025', u'\u8bca', u'\u73af', u'\u74dc', u'\u98ce', u'\u6e7f', u'\u56e0', u'\u6eb6', u'\u7740', u'\u70b9', u'\u53cc', u'\u5217', u'\u9ed1', u'\u7cca', u'\u70c2', u'\u7eff', u'\u6cf3', u'\u7f51', u'\u7ec7', u'\u767e', u'\u7eb3', u'\u7aef', u'\u5229', u'\u80f8', u'\u6c34', u'\u674e', u'\u51e1', u'\u4ed6', u'\u5757', u'\u6d51', u'\u6216', u'\u8349', u'\u7fa4', u'\u4f20', u'\u7814', u'\u9cde', u'\u76f8', u'\u5173', u'\u795e', u'\u7ecf', u'\u5143', u'\u70ef', u'\u89d2', u'\u7247', u'\u6bb5', u'\u679d', u'\u6746', u'\u9759', u'\u9910', u'\u8179', u'\u8865', u'\u968f', u'\u5fae', u'\u89c6', u'\u5408', u'\u5185', u'\u79d1', u'\u4e24', u'\u5468', u'\u9897', u'\u529b', u'\u65c1', u'\u62f7', u'\u8d1d', u'\u8f7d', u'\u534a', u'\u7d27', u'\u5f20', u'\u6d46', u'\u996e', u'\u7acb', u'\u5367', u'\u919b', u'\u666e', u'\u901a', u'\u7a00', u'\u5efa', u'\u8bae', u'\u590d', u'\u6210', u'\u80fa', u'\u4e00', u'\u7a7a', u'\u5de8', u'\u8fa9', u'\u679c', u'\u8be6', u'\u80f0', u'\u5c9b', u'\u91ca', u'\u653e', u'\u4eae', u'\u7ed2', u'\u6bdb', u'\u819c', u'\u7ebf', u'\u7ec4', u'\u589e', u'\u6b96', u'\u7eba', u'\u9524', u'\u6ce2', u'\u5f62', u'\u5c14', u'\u9ad3', u'\u5907', u'\u6ce8', u'\u5455', u'\u5410', u'\u7edd', u'\u5e95', u'\u53ca', u'\u620a', u'\u5e9a', u'\u975e', u'\u8f7b', u'\u6f88', u'\u836f', u'\u514b', u'\u83ab', u'\u53f8', u'\u9aa8', u'\u8c22', u'\u80f6', u'\u6b8a', u'\u5e8f', u'\u5ef6', u'\u957f', u'\u4f1a', u'\u611f', u'\u67d3', u'\u8367', u'\u5149', u'\u591a', u'\u5ba4', u'\u8f85', u'\u52a9', u'\u5236', u'\u4e2a', u'\u5171', u'\u8bfb', u'\u690d', u'\u7528', u'\u786c', u'\u7fa7', u'\u96c6', u'\u9732', u'\u771f', u'\u5e72', u'\u6270', u'\u4e25', u'\u6b8b', u'\u7559', u'\u7076', u'\u5f0f', u'\u56fe', u'\u6587', u'\u62a5', u'\u544a', u'\u6d82', u'\u5176', u'\u5b66', u'\u7070', u'\u8154', u'\u4ec5', u'\u4f9b', u'\u7cdc', u'\u7c73', u'\u53ef', u'\u6d45', u'\u80a0', u'\u9053', u'\u79cd', u'\u4f18', u'\u52bf', u'\u6b67', u'\u916a', u'\u68ad', u'\u62c9', u'\u67d4', u'\u5ae9', u'\u8bba', u'\u547c', u'\u949f', u'\u5e15', u'\u6df1', u'\u7965', u'\u9686', u'\u5b55', u'\u6ce1', u'\u6ccc', u'\u6000', u'\u96cc', u'\u6392', u'\u523a', u'\u8eab', u'\u6ed1', u'\u777e', u'\u5c11', u'\u4e07', u'\u9274', u'\u8bf7', u'\u7c07', u'\u6240', u'\u5e7c', u'\u7a1a', u'\u5076', u'\u5e26', u'\u57f9', u'\u517b', u'\u53f6', u'\u5c42', u'\u6df7', u'\u5df2', u'\u6a59', u'\u5730', u'\u8f9b', u'\u5c81', u'\u4ee5', u'\u6813', u'\u5f39', u'\u53d1', u'\u9c8e', u'\u4f8b', u'\u915a', u'\u8fde', u'\u94dc', u'\u84dd', u'\u80ba', u'\u652f', u'\u60c5', u'\u77f3', u'\u4efd', u'\u6742', u'\u4ea4', u'\u5e7d', u'\u73b0', u'\u75c7', u'\u670d', u'\u9644', u'\u7eaf', u'\u75b1', u'\u75b9', u'\u5f13', u'\u884c', u'\u51fa', u'\u70ed', u'\u7efc', u'\u505c', u'\u534e', u'\u6797', u'\u513f', u'\u8336', u'\u53bb', u'\u80aa', u'\u553e', u'\u6d01', u'\u5b63', u'\u8282', u'\u65b0', u'\u5496', u'\u5561', u'\u6548', u'\u4ef7', u'\u5668', u'\u5b98', u'\u5e93', u'\u878d', u'\u4e34', u'\u5e8a', u'\u6f84', u'\u53e4', u'\u5cf0', u'\u6307', u'\u53d7', u'\u4eca', u'\u65e9', u'\u62bd', u'\u6628', u'\u665a', u'\u5219', u'\u7b5b', u'\u8010', u'\u4e0b', u'\u5927', u'\u9ecf', u'\u6bcf', u'\u5207', u'\u8ff9', u'\u505a', u'\u9664', u'\u6b21', u'\u59cb', u'\u4ecb', u'\u574f', u'\u6b7b', u'\u7075', u'\u829d', u'\u754c', u'\u82ef', u'\u52a0', u'\u7a81', u'\u53d8', u'\u8863', u'\u519b', u'\u56e2', u'\u526f', u'\u4e0e', u'\u6c9f']
TEST_SIZE = 2000
ADD_BLANK = True   # if add a blank between digits
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 20000
# parameters for bdlstm ctc
BATCH_SIZE = 30
BATCHES = 20
TRAIN_SIZE = BATCH_SIZE*BATCHES #2560
MOMENTUM = 0.9
REPORT_STEPS = 120
# Hyper-parameters
num_epochs = 1000
num_hidden = 200
num_layers = 1
num_classes = len(CHARSET) + 1   # 781


'''
encode 
'''
char2code = {}
for code ,char in enumerate(CHARSET):
    char2code[char] = code 

def str2code(s):
    dflt = char2code["~"]
    return [char2code.get(c,dflt) for c in s]#dict.get(key, default=None)

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
    return 1. / (1. + numpy.exp(-a))


"""
    {"dirname":{"fname":(im,code)}}
"""
data_set = {}


def load_data_set(dirname):
    fname_list = glob.glob(dirname + "/*.jpg") # ['test/010009.bin.png',..., 'test/010063.bin.png', ]
    result = dict()
    for fname in sorted(fname_list):
        print ("loading", fname)
        im = cv2.imread(fname)[:, :, 0].astype(numpy.float32) / 255. #shape (59,3068)
        im = cv2.resize(im,(3000,100))#统一到同一尺度(60,3000)
        with open(fname.split('.')[0]+'.txt','r') as f:
            strs = f.read()#content
        index = fname.split("/")[1]#图片代号 '010000.bin.png'
        result[index] = (im, strs)
    data_set[dirname] = result
'''
data_set={'test': {'010001.bin.png': (array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
       ..., 
       [ 1.,  1.,  1., ...,  1.,  1.,  1.],
       [ 1.,  1.,  1., ...,  1.,  1.,  1.]], dtype=float32), '60001470 87 女 A12000067994 肝肾脂糖电解质测定 SH003 球蛋白(GLB) 22.3 g/L 20.0-35.0 2016/5/10 12:07\n'), '010000.bin.png': (array([[ 1.,  1.,  1., ...,  1.,  1.,  1.],
 
       ..., }   

'''


def read_data_for_lstm_ctc(dirname, start_index=None, end_index=None):

    fname_list = []
    if dirname not in data_set.keys():
        load_data_set(dirname) #exec dirname:'test'
    f_list = glob.glob(dirname + "/*.jpg") 
    if start_index is None:#exec 
        fname_list = [fname.split("/")[1].split("_")[0] for fname in f_list] 
    else: #exec while training
        fname_list = [fname.split("/")[1] for fname in f_list][start_index:end_index]#['010009.bin.png',....] shape:64
    dir_data_set = data_set.get(dirname)

    for fname in sorted(fname_list):
        im, strs = dir_data_set.get(fname)
        codes = numpy.asarray(str2code(strs))
        yield im , codes  


def unzip(b):
    xs, ys = zip(*b) #unzip
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys
    
def normalize_text(s):
    """Apply standard Unicode normalizations for OCR.
    This eliminates common ambiguities and weird unicode
    characters."""
    #s = unicode(s)
    s = unicodedata.normalize('NFC',s)
    s = re.sub(r'\s+(?u)',' ',s)
    s = re.sub(r'\n(?u)','',s)
    s = re.sub(r'^\s+(?u)','',s)
    s = re.sub(r'\s+$(?u)','',s)
    for m,r in chars.replacements:
        s = re.sub((m),(r),s)
    return s

if __name__ == '__main__':
    train_inputs, train_codes = unzip(list(read_data_for_lstm_ctc("test"))[:2])
    print("train_codes", train_codes)
    targets = np.asarray(train_codes).flat[:]
