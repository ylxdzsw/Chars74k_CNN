import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn
import mxnet.gluon.rnn as rnn

class BiGRU(nn.Block):
    def __init__(self, nhidden, downsample=False, **kwargs):
        super(BiGRU, self).__init__(**kwargs)

        with self.name_scope():
            self.cell = rnn.BidirectionalCell(
                rnn.GRUCell(nhidden),
                rnn.GRUCell(nhidden),
            )
            self.projector = nn.Conv1D(nhidden, kernel_size=1, strides=1 + downsample, activation='relu')

    def forward(self, x):
        x = self.cell.unroll(x.shape[1], x, merge_outputs=True)[0]
        return self.projector(nd.relu(x).transpose((0, 2, 1))).transpose((0, 2, 1))

class Model(object):
    def __init__(self, vsize, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()

        self.net = nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                nn.Embedding(vsize, 256),
                BiGRU(1024),
                nn.Dropout(0.5),
                BiGRU(512),
                nn.GlobalMaxPool1D(),
                nn.Flatten(),
                nn.Dense(6)
            )

        if file != None:
            self.net.load_params(file, ctx=self.ctx)
        else:
            self.net.initialize(ctx=self.ctx)

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
