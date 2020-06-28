import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        weight_filler={"type": "xavier"}, bias_filler={"type": "constant"},
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def conv_relu_10x(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        weight_filler={"type": "xavier"}, bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            seed=1337)
    if split == 'train':
        pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
        pylayer = 'SBDDSegDataLayer'
    else:
        pydata_params['voc_dir'] = '../data/pascal/VOC2012'
        pylayer = 'VOCSegDataLayer'
    n.data, n.label, n.flux, n.flux_weight = L.Python(module='voc_layers', layer=pylayer,
            ntop=4, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    flux_hidden_channels = 256

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score_fr = L.Convolution(n.drop7, num_output=21, kernel_size=1, pad=0,
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.upscore2 = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=4, stride=2, pad=1,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # fully flux conv
    n.fc6_flux, n.relu6_flux = conv_relu(n.relu5_3, 4096, ks=7, pad=0)
    n.drop6_flux = L.Dropout(n.relu6_flux, dropout_ratio=0.5, in_place=True)
    n.fc7_flux, n.relu7_flux = conv_relu(n.drop6_flux, 4096, ks=1, pad=0)
    n.drop7_flux = L.Dropout(n.relu7_flux, dropout_ratio=0.5, in_place=True)

    n.score_fr_flux = L.Convolution(n.drop7_flux, num_output=flux_hidden_channels, kernel_size=1, pad=0,
        weight_filler={"type": "xavier"}, bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.upscore2_flux = L.Deconvolution(n.score_fr_flux,
        convolution_param=dict(num_output=flux_hidden_channels, kernel_size=4, stride=2, pad=1,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # scale pool4 skip for compatibility
    n.scale_pool4 = L.Scale(n.pool4, filler=dict(type='constant',
        value=0.01), param=[dict(lr_mult=0)])
    n.score_pool4 = L.Convolution(n.scale_pool4, num_output=21, kernel_size=1, pad=0,
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
            operation=P.Eltwise.SUM)
    n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
        convolution_param=dict(num_output=21, kernel_size=4, stride=2, pad=1,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # scale pool4 skip for compatibility flux
    n.scale_pool4_flux = L.Scale(n.relu4_3, filler=dict(type='constant',
        value=0.01), param=[dict(lr_mult=0)])
    n.score_pool4_flux = L.Convolution(n.scale_pool4_flux, num_output=flux_hidden_channels, kernel_size=1, pad=0,
        weight_filler={"type": "xavier"}, bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.score_pool4c_flux = crop(n.score_pool4_flux, n.upscore2_flux)
    n.fuse_pool4_flux = L.Eltwise(n.upscore2_flux, n.score_pool4c_flux,
            operation=P.Eltwise.SUM)
    n.upscore_pool4_flux = L.Deconvolution(n.fuse_pool4_flux,
        convolution_param=dict(num_output=flux_hidden_channels, kernel_size=4, stride=2, pad=1,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # scale pool3 skip for compatibility
    n.scale_pool3 = L.Scale(n.pool3, filler=dict(type='constant',
        value=0.0001), param=[dict(lr_mult=0)])
    n.score_pool3 = L.Convolution(n.scale_pool3, num_output=21, kernel_size=1, pad=0,
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
    n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c,
            operation=P.Eltwise.SUM)
    n.upscore8 = L.Deconvolution(n.fuse_pool3,
        convolution_param=dict(num_output=21, kernel_size=16, stride=8, pad=4,
            bias_term=False),
        param=[dict(lr_mult=0)])

    # scale pool3 skip for compatibility flux
    n.scale_pool3_flux = L.Scale(n.relu3_3, filler=dict(type='constant',
        value=0.0001), param=[dict(lr_mult=0)])
    n.score_pool3_flux = L.Convolution(n.scale_pool3_flux, num_output=flux_hidden_channels, kernel_size=1, pad=0,
        weight_filler={"type": "xavier"}, bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
    n.score_pool3c_flux = crop(n.score_pool3_flux, n.upscore_pool4_flux)
    n.fuse_pool3_flux = L.Eltwise(n.upscore_pool4_flux, n.score_pool3c_flux,
            operation=P.Eltwise.SUM)
    
    n.final_conv1, n.final_relu1 = conv_relu_10x(n.fuse_pool3_flux, 64)
    n.final_conv2, n.final_relu2 = conv_relu_10x(n.final_relu1, 64)
    n.final_conv3 = L.Convolution(n.final_relu2, num_output=2, kernel_size=3, stride=1, pad=1,
        weight_filler={"type": "xavier"}, bias_filler={"type": "constant"},
        param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])

    n.upscore4_final_conv3 = L.Deconvolution(n.final_conv3,
        convolution_param=dict(num_output=2, kernel_size=8, stride=4, pad=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score = crop(n.upscore8, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))
    
    n.flux_score = crop(n.upscore4_final_conv3, n.data)
    n.norm_loss = L.Python(n.flux_score, n.flux, n.flux_weight, module='utils', layer='WeightedEuclideanLossLayer', loss_weight=1)
    n.angle_loss = L.Python(n.flux_score, n.flux, n.flux_weight, module='utils', layer='WeightedAngleLossLayer', loss_weight=1)

    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('seg11valid')))

if __name__ == '__main__':
    make_net()
