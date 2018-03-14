import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Net(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.hidden = nn.Dense(50, 'relu')
            self.out = nn.Dense(6)

    def hybrid_forward(self, F, x):
        feat = self.hidden(x)
        return feat, self.out(feat)

class Model(object):
    def __init__(self, embedding, ctx='cpu'):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        self.net = Net()
        self.net.initialize(ctx=self.ctx, init=mx.init.Xavier())
        self.net.hybridize()
        self.trainer = mx.gluon.Trainer(self.net.collect_params(), 'adam', {'wd': 0.00005})

    def train(self, x, y):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
        x, y = nd.array(x, self.ctx), nd.array(y, self.ctx)
        with mx.autograd.record():
            f, p = self.net(x)
            L = loss(p, y).mean()
        L.backward()
        self.trainer.step(1)
        return L.asscalar()

    def predict(self, x):
        f, p = self.net(nd.array(x, self.ctx))
        return f.asnumpy(), p.asnumpy()
