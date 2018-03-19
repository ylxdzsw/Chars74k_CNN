import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn

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
        return F.concat(x, self.net(x), dim=2)

class Detector(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Detector, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            self.net.add(
                nn.Conv2D(16, kernel_size=(3,3), padding=(1,1), dilation=(1,1), activation='relu'),
                DenseDilation(16),
                DenseDilation(16),
                nn.Conv2D(2, kernel_size=(1,1))
            )

    def hybrid_forward(self, F, x):
        return F.log_softmax(self.net(x), axis=2)

class Masker(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Masker, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Conv2D(1, kernel_size=(1,1), activation='sigmoid')

    def hybrid_forward(self, F, x, f):
        x = F.concat(x, f, dim=2)
        return self.fc(x)

class Model(object):
    def __init__(self, vsize, ctx='cpu', file=None):
        self.ctx = mx.cpu() if ctx != 'gpu' else mx.gpu()
        self.deconv = nn.Conv2DTranspose(16, kernel_size=(2,2), strides=(2,2))

