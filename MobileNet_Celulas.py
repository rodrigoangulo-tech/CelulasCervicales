import torch
import torchvision.models as modelos
import torch.nn as nn
from pathlib import Path
from torchmetrics import MetricCollection
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import torchmetrics as tm
import matplotlib.pyplot as plt
import ModeloBMN

dir = Path.cwd()
dir1 = dir / 'Celulas' / 'DataSet' / 'Entrenamiento'
dir2 = dir / 'Celulas' / 'DataSet' / 'Test'

print(dir1)
print(dir2)
#print("Funciona la  GPU? ", torch.cuda.is_available())
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

mobilenet3 = modelos.mobilenet_v3_large(weights = True) #Se requiere preentrenamiento

modelo = mobilenet3
modelo = mobilenet3.features

#modelo.classifier[3] = nn.Linear(in_features=1280, out_features=5)#Numero de salidas 5: NILM, LSIL, HSIL, ASC y SCC

clasf = nn.Sequential(
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(960,512),nn.LeakyReLU(),nn.Linear(512,128),nn.LeakyReLU(),nn.Linear(128,5), nn.LeakyReLU()
)

modelo.classifier = clasf
modelo = modelo.to(dispositivo)

#Un tensor de prueba para verificar correcta estructura del modelo
p = torch.rand((1,3,224,224))
p = p.to(dispositivo)
y = modelo(p)
print(y)

summary(modelo, input_size=(3,224,224))

transformacion1 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.299,0.224,0.225]),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomRotation(15),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.1))
])

transformacion2 = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.299,0.224,0.225])
])

dataset_entrm = datasets.ImageFolder(root=dir1, transform=transformacion1)
dataset_val = datasets.ImageFolder(root=dir2, transform=transformacion2)
cargar_entrm = DataLoader(dataset_entrm,batch_size=32,shuffle=True)
cargar_val = DataLoader(dataset_val,shuffle=False)

criterio = nn.CrossEntropyLoss().to(dispositivo)
optimizador = optim.Adam(modelo.parameters(),lr=0.001)
num_epocas = 50


'''
summary(mobilenet3d, input_size=(3,224,224))
#summary(mobilenet3d, input_size=(3,224,224))

'''
metricas = tm.MetricCollection([
    tm.Accuracy(task = "multiclass", num_classes=5),tm.Precision(task="multiclass",num_classes=5),
    tm.Recall(task="multiclass",num_classes=5),
    tm.Specificity(task="multiclass",num_classes=5)])
metricas = metricas.to(dispositivo)
costo_entr = []
costo_val = []


for epoca in range(num_epocas):
    # entrenamiento
    modelo.train()
    costo_acumulado = 0.0
    for entradas, etiquetas in cargar_entrm:
        entradas = entradas.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        optimizador.zero_grad()

        salidas = modelo(entradas)
        costo = criterio(salidas,etiquetas)

        costo.backward()
        optimizador.step()

        costo_acumulado += costo.item()

    costo_entr.append(costo.item())
    print(f"Epoca [{epoca}],Costo: {costo_acumulado/len(cargar_entrm):.2f}")

    #Evaluacion
    modelo.eval()
    correcto=0.0
    total= 0.0

    with torch.no_grad():
        for entradas, etiquetas in cargar_val:
            entradas = entradas.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)
            salidas = modelo(entradas)
            costov = criterio(salidas, etiquetas)
            az,prediccion = torch.max(salidas.data,1)

            total += etiquetas.size(0)
            correcto += (prediccion==etiquetas).sum().item()

        costo_val.append(costov.item())

    precision = 100*correcto/total

print(f' Precision de Validacion: {precision:.2f}%') #precision simple
dirA = dir / 'Modelo_Celulas.pth' #Guarda el modelo
torch.save(modelo.state_dict(),dirA)

print(costo_entr)
print(costo_val)
plt.plot(costo_entr,label = "Costo entrenamiento")
plt.plot(costo_val,label="Costo validacion")
plt.legend()
plt.title("Costo por epoca")
plt.show()

