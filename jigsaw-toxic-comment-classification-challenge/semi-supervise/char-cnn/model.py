import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn
import mxnet.contrib.text as text

class Residual(nn.HybridBlock):
    def __init__(self, nkernel, nbottleneck, **kwargs):
        super(Residual, self).__init__(**kwargs)
        with self.name_scope():
            self.downconv = nn.Conv1D(nbottleneck, kernel_size=3, padding=1, activation='relu')
            self.upconv = nn.Conv1D(nkernel, kernel_size=1)
            self.bn = nn.BatchNorm()

    def hybrid_forward(self, F, x):
        out = self.bn(self.upconv(self.downconv(x)))
        return F.relu(out + x)

class Transpose(nn.HybridBlock):
    def __init__(self, spec, **kwargs):
        super(Transpose, self).__init__(**kwargs)
        self.spec = spec

    def hybrid_forward(self, F, x):
        return F.transpose(x, self.spec)

class Output(nn.HybridBlock):
    def __init__(self, nlabel, **kwargs):
        super(Output, self).__init__(**kwargs)
        self.avgpool = nn.GlobalAvgPool1D()
        self.maxpool = nn.GlobalMaxPool1D()
        self.flat = nn.Flatten()
        self.compressor = nn.Dense(50, activation='relu')
        self.dense = nn.Dense(6)

    def hybrid_forward(self, F, x):
        avgpool = self.avgpool(x)
        maxpool = self.maxpool(x)
        flat = self.flat(F.concat(avgpool, maxpool, dim=1))
        feat = self.compressor(flat)
        return feat, self.dense(feat)

class Model(object):
    def __init__(self, vsize, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()

        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.Embedding(vsize, 1024),
                Transpose((0, 2, 1)),

                nn.Conv1D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool1D(pool_size=2, strides=2),

                Residual(256, 64),
                Residual(256, 64),
                Residual(256, 64),
                Residual(256, 64),
                Residual(256, 64),
                nn.MaxPool1D(pool_size=2, strides=2),

                Residual(256, 64),
                Residual(256, 64),
                Output(6)
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx, init=mx.init.Xavier())

        self.trainer = mx.gluon.Trainer(self.net.collect_params(), 'adam', {'wd': 0.00005})
        self.net.hybridize()

    def train(self, x, y):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
        x, y = nd.array(x, self.ctx), nd.array(y, self.ctx)
        with mx.autograd.record():
            _, p = self.net(x)
            L = loss(p, y).mean()
        L.backward()
        self.trainer.step(1)
        return L.asscalar()

    def predict(self, x):
        f, p = self.net(nd.array(x, self.ctx))
        return f.asnumpy(), p.asnumpy()

    def save(self, file):
        return self.net.save_params(file)
