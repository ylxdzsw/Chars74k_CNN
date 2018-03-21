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

class Masker(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Masker, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Conv2D(1, kernel_size=(1,1), activation='sigmoid')

    def hybrid_forward(self, F, x, f):
        x = F.concat(x, f, dim=2)
        return self.fc(x)

def ndinput(x, ctx):
    return nd.expand_dims(nd.array(x, ctx=ctx), 0)

def half(x):
    return nd.Pooling(x, pool_type='avg', kernel=(2,2), stride=(2,2), pooling_convention='full')

def get_net(name, layers, ctx, path):
    net = nn.HybridSequential()
    net.add(*layers)

    params = net.collect_params()

    if path != None:
        params.load(path+name+'.model', ctx=ctx)
    else:
        net.initialize(ctx=ctx, init=mx.init.Xavier())

    net.hybridize()

    return net, mx.gluon.Trainer(params, 'adam'), lambda path: params.save(path+name+'.model')

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
            nn.Conv2D(2, kernel_size=(1,1))
        ], self.ctx, path)

        self.suffuser, self.suffuser_trainer, self.suffuser_saver = get_net('suffuser', [
            nn.Conv2DTranspose(16, kernel_size=(2,2), strides=(2,2))
        ], self.ctx, path)

        self.masker, self.masker_trainer, self.masker_saver = get_net('masker', [
            Masker()
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

    def learn_mask(self, image, cover, mask, level):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

        image = ndinput(image, self.ctx)
        mask  = nd.expand_dims(ndinput(mask, self.ctx), 0)
        cover = nd.expand_dims(ndinput(cover, self.ctx), 0)

        for _ in range(level):
            image, cover = half(image), half(cover)

        L = 0
        with mx.autograd.record():
            feat = self.feature(image)
            feat = feat * (1 - cover)


        L.backward()
        self.feature_trainer.step(1)
        self.detector_trainer.step(1)

    def detect(self, x, cover=None):
        x = ndinput(x, self.ctx)
        f = self.feature(x)
        if cover:
            f *= 1 - cover
        p = self.detector(f)
        return f, p

    def save(self, path):
        self.feature_saver(path)
        self.detector_saver(path)
        self.suffuser_saver(path)
        self.masker_saver(path)
