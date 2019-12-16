import torchvision.models as models


def get_resnet18(config):
    return models.resnet18()
    
def get_vgg16(config):   
    return models.vgg16()