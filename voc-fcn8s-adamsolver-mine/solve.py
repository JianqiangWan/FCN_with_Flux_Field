import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

vgg_weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
vgg_proto = 'VGG_ILSVRC_16_layers_deploy.prototxt'


# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.AdamSolver('solver.prototxt')
# solver.net.copy_from('snapshot_bk/train_iter_156000.caffemodel')
vgg_net = caffe.Net(vgg_proto, vgg_weights, caffe.TRAIN)

surgery.transplant(solver.net, vgg_net)
del vgg_net
# solver.net.copy_from('snapshot_bk/train_iter_120000.caffemodel')
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/pascal/seg11valid.txt', dtype=str)

# score.seg_tests(solver, False, val, layer='score')
for _ in range(75):
    solver.step(8000)
    score.seg_tests(solver, False, val, layer='score')
