#Arquitectura de Modelo basado en MobileNet
#Aqui se encuentra solamente la arquitectura para la implementeación del modelo, ya sea entrenamiento, validación o test.
#En un archivo aparte para lograr modularidad.
import torchvision.models as modelos
import torch.nn as nn

def ModeloBMN():
    mobilenet3 = modelos.mobilenet_v3_large()
    modelo = mobilenet3.features
    clasf = nn.Sequential(
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(960,512),nn.LeakyReLU(),nn.Linear(512,128),nn.LeakyReLU(),nn.Linear(128,5)
    )
    modelo.classifier = clasf
    return modelo




