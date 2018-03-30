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

def ndinput4(x, ctx):
    x = nd.array(x, ctx=ctx)
    while x.ndim < 4:
        x = nd.expand_dims(x, 0)
    return x

def half(x, t='avg'):
    return nd.Pooling(x, pool_type=t, kernel=(2,2), stride=(2,2), pooling_convention='full')

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

        mask_levels = [ndinput4(level, self.ctx) for level in mask_levels]
        image = ndinput4(image, self.ctx)
        cover = ndinput4(cover, self.ctx)

        L = 0
        with mx.autograd.record():
            for mask in mask_levels:
                feat = self.feature(image)
                feat = feat * (1 - cover)
                pred = self.detector(feat)
                L = L + loss(pred, mask).sum()
                image, cover = half(image), half(cover)

        L.backward()
        self.feature_trainer.step(2)
        self.detector_trainer.step(1)

        return L.asscalar()

    def learn_mask(self, image, mask, cover, centers, r):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

        image = ndinput4(image, self.ctx)
        mask  = ndinput4(mask, self.ctx)
        cover = ndinput4(cover, self.ctx)

        position = ndinput4(position_mask(r), self.ctx)

        L = 0
        with mx.autograd.record():
            feat = self.feature(image) * (1 - cover)

            mask  = mask.pad(mode='constant',  constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))
            cover = cover.pad(mode='constant', constant_value=1, pad_width=(0,0,0,0,r-1,r,r-1,r))
            feat  = feat.pad(mode='constant',  constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))

            for x, y in centers:
                slice = lambda c: c[:,:,x-1:x+2*r-1,y-1:y+2*r-1]
                f = nd.concat(slice(feat), position, dim=1)
                p = self.masker(f).sigmoid() * (1 - slice(cover))
                L = L + loss(p, slice(mask)).sum() / len(centers)

        L.backward()
        self.feature_trainer.step(4)
        self.masker_trainer.step(1)

        return L.asscalar()

    def get_feature(self, x):
        x = ndinput4(x, self.ctx)
        return self.feature(x)

    def half(self, x):
        return half(ndinput4(x, self.ctx))

    def detect(self, f, cover=None):
        if cover is not None:
            c = ndinput4(cover, self.ctx)
            while c.shape[2:] != f.shape[2:]:
                c = half(c, 'max')
            f = f * (1 - c)
        return self.detector(f).asnumpy()[0,0,:,:]

    def mask(self, feat, x, y, r, cover):
        cover = ndinput4(cover, self.ctx)
        feat  = feat * (1 - cover)

        cover = cover.pad(mode='constant', constant_value=1, pad_width=(0,0,0,0,r-1,r,r-1,r))
        feat  = feat.pad(mode='constant',  constant_value=0, pad_width=(0,0,0,0,r-1,r,r-1,r))

        position = ndinput4(position_mask(r), self.ctx)

        slice = lambda c: c[:,:,x-1:x+2*r-1,y-1:y+2*r-1]
        f = nd.concat(slice(feat), position, dim=1)
        p = self.masker(f).sigmoid() * (1 - slice(cover))

        return p.asnumpy()[0,0,:,:]

    def save(self, path):
        self.feature_saver(path)
        self.detector_saver(path)
        self.masker_saver(path)
