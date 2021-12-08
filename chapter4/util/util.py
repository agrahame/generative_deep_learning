from fastai.vision.all import *

class GANIntToFloatTensor(DisplayedTransform):
    order = 10 #Need to run after PIL transforms on the GPU
    def __init__(self, div=127.5, scale=127.5): store_attr()
    def encodes(self, img:TensorImage): return img.float().sub_(self.scale).div_(self.div)
    def decodes(self, img:TensorImage): return (((img.clamp(-1., 1.) * self.div) + self.scale).long()) if self.div else img

def GANImageBlock(cls=PILImage):
    return TransformBlock(type_tfms=cls.create, batch_tfms=GANIntToFloatTensor)