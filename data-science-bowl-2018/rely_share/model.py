import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Residual(nn.HybridBlock):
    def __init__(self, nkernel, **kwargs):
        super(Residual, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(nkernel, kernel_size=(3,3), padding=(1,1))
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(nkernel, kernel_size=(3,3), padding=(1,1))
            self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

class DenseDilation(nn.HybridBlock):
    def __init__(self, nkernel, **kwargs):
        super(DenseDilation, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            self.net.add(
                nn.Conv2D(nkernel, kernel_size=(3,3), padding=(1,1), dilation=(1,1), activation='relu'),
                nn.Conv2D(nkernel, kernel_size=(3,3), padding=(2,2), dilation=(2,2), activation='relu'),
                nn.Conv2D(nkernel, kernel_size=(3,3), padding=(4,4), dilation=(4,4), activation='relu')
            )

    def hybrid_forward(self, F, x):
        return F.concat(x, self.net(x), dim=1)

def ndinput(x, ctx):
    return nd.expand_dims(nd.array(x, ctx=ctx), 0)

def half(x):
    return nd.Pooling(x, pool_type='avg', kernel=(2,2), stride=(2,2), pooling_convention='full')

def get_net(name, layers, ctx, path):
    net = nn.HybridSequential()
    net.add(*layers)

    params = net.collect_params()

    try:
        params.load(path+name+'.model', ctx=ctx)
    except:
        net.initialize(ctx=ctx, init=mx.init.Xavier())

    net.hybridize()

    return net, mx.gluon.Trainer(params, 'adam'), lambda path: params.save(path+name+'.model')

def position_mask(r):
    r *= 2
    return [[[(x, y)[i] / (r - 1) for x in range(r)] for y in range(r)] for i in range(2)]

class Model(object):
    def __init__(self, ctx='cpu', path=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()

        self.feature, self.feature_trainer, self.feature_saver = get_net('feature', [
            nn.Conv2D(16, kernel_size=(3,3), padding=(1,1), activation='relu'),
            Residual(16),
            Residual(16),
            DenseDilation(16),
            Residual(32)
        ], self.ctx, path)

        self.detector, self.detector_trainer, self.detector_saver = get_net('detector', [
            nn.Conv2D(32, kernel_size=(3,3), padding=(1,1), activation='relu'),
            nn.Conv2D(1, kernel_size=(1,1))
        ], self.ctx, path)

        self.masker, self.masker_trainer, self.masker_saver = get_net('masker', [
            nn.Conv2D(32, kernel_size=(3,3), padding=(1,1), activation='relu'),
            nn.Conv2D(1, kernel_size=(1,1))
        ], self.ctx, path)

    def learn_detect(self, image, cover, mask_levels):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

        mask_levels = [ndinput(level, self.ctx) for level in mask_levels]
        image = ndinput(image, self.ctx)
        cover = nd.expand_dims(ndinput(cover, self.ctx), 0)

        L = 0
        with mx.autograd.record():
            for mask in mask_levels:
                feat = self.feature(image)
                feat = feat * (1 - cover)
                pred = self.detector(feat)
                L = L + loss(pred, mask).sum()
                image, cover = half(image), half(cover)

        L.backward()
        self.feature_trainer.step(1)
        self.detector_trainer.step(1)

        return L.asscalar()

    def learn_mask(self, image, mask, cover, centers, r):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

        image = ndinput(image, self.ctx)
        mask  = nd.expand_dims(ndinput(mask, self.ctx), 0)
        cover = nd.expand_dims(ndinput(cover, self.ctx), 0)
        feat  = self.feature(image) * (1 - cover)

        image = image.pad(mode='constant', constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))
        mask  = mask.pad(mode='constant',  constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))
        cover = cover.pad(mode='constant', constant_value=1, pad_width=(0,0,0,0,r-1,r,r-1,r))
        feat  = feat.pad(mode='constant',  constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))

        position = ndinput(position_mask(r), self.ctx)

        L = 0
        with mx.autograd.record():
            for x, y in centers:
                slice = lambda c: c[:,:,x-1:x+2*r-1,y-1:y+2*r-1]
                f = nd.concat(slice(image), slice(feat), position, dim=1)
                p = self.masker(f) * (1 - slice(cover))
                L = L + loss(p, slice(mask)).sum() / len(centers)

        L.backward()
        self.masker_trainer.step(1)

        return L.asscalar()

    def get_feature(self, x):
        x = ndinput(x, self.ctx)
        f = self.feature(x)
        return x, f

    def detect(self, f, cover=None):
        if cover != None:
            c = nd.expand_dims(ndinput(cover, self.ctx), 0)
            f = f * (1 - c)
        return self.detector(f).asnumpy()[0,0,:,:]

    def mask(self, image, feat, x, y, r, cover):
        cover = nd.expand_dims(ndinput(cover, self.ctx), 0)
        feat  = feat * (1 - cover)

        image = image.pad(mode='constant', constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))
        cover = cover.pad(mode='constant', constant_value=1, pad_width=(0,0,0,0,r-1,r,r-1,r))
        feat  = feat.pad(mode='constant',  constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))

        position = ndinput(position_mask(r), self.ctx)

        slice = lambda c: c[:,:,x-1:x+2*r-1,y-1:y+2*r-1]
        f = nd.concat(slice(image), slice(feat), position, dim=1)
        p = self.masker(f) * (1 - slice(cover))

        return p.asnumpy()[0,0,:,:]

    def save(self, path):
        self.feature_saver(path)
        self.detector_saver(path)
        self.masker_saver(path)
