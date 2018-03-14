import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

class Model(object):
    def __init__(self, ctx='cpu'):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        self.net = nn.Dense(6)
        self.net.initialize(ctx=self.ctx, init=mx.init.Xavier())
        self.trainer = mx.gluon.Trainer(self.net.collect_params(), 'adam', {'wd': 0.00002})
        self.net.hybridize()

    def train(self, x, y):
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
        x, y = nd.array(x, self.ctx), nd.array(y, self.ctx)
        with mx.autograd.record():
            p = self.net(x)
            L = loss(p, y).mean()
        L.backward()
        self.trainer.step(1)
        return L.asscalar()

    def predict(self, x):
        return self.net(nd.array(x, self.ctx)).asnumpy()
