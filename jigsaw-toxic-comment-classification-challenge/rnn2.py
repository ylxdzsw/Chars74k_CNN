import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn
import mxnet.gluon.rnn as rnn

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
        self.dense = nn.Dense(6)

    def hybrid_forward(self, F, x):
        avgpool = self.avgpool(x)
        maxpool = self.maxpool(x)
        flat = self.flat(F.concat(avgpool, maxpool, dim=1))
        return self.dense(flat)

class ResidualBiGRU(nn.Block):
    def __init__(self, nhidden, **kwargs):
        super(ResidualBiGRU, self).__init__(**kwargs)

        with self.name_scope():
            self.cell = rnn.BidirectionalCell(
                rnn.GRUCell(nhidden),
                rnn.GRUCell(nhidden),
            )

    def forward(self, x):
        h = self.cell.unroll(x.shape[1], x, merge_outputs=True)[0]
        return nd.concat(x, nd.relu(h), dim=2)

class Model(object):
    def __init__(self, vsize, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()

        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                nn.Embedding(vsize, 512),
                Transpose((0, 2, 1)),
                nn.Conv1D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool1D(pool_size=2, strides=2),
                Transpose((0, 2, 1)),
                ResidualBiGRU(100),
                Transpose((0, 2, 1)),
                nn.Conv1D(256, kernel_size=1, activation='relu'),
                Output(6)
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx, init=mx.init.Xavier())

    def train(self, x, y, lr):
        trainer = mx.gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': lr})
        loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
        x, y = nd.array(x, self.ctx), nd.array(y, self.ctx)
        with mx.autograd.record():
            p = self.net(x)
            L = loss(p, y).mean()
        L.backward()
        trainer.step(1)
        return L.asscalar()

    def predict(self, x):
        return self.net(nd.array(x, self.ctx)).asnumpy()

    def save(self, file):
        return self.net.save_params(file)
