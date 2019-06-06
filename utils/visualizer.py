# coding:utf8
import visdom
import time
import numpy as np
import torchvision


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)

        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        image = img_.detach().cpu().numpy()
        # image = image / np.max(image)
        self.vis.images(image,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )
    def imshow(self, name, img_, **kwargs):

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        # image = img_.detach().cpu().permute(0,2,3,1).numpy() * std + mean
        # image = image / np.max(image)
        image = torchvision.utils.make_grid(img_.detach().cpu()).numpy().transpose(1,2,0)
        image = std * image + mean
        image = image.clip(0, 1)
        self.vis.images(image.transpose(2,0,1),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def depth(self, name, img_, **kwargs):

        image = img_.detach().cpu().numpy()
        image = image / np.max(image)
        self.vis.images(image,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        :param lr:
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
