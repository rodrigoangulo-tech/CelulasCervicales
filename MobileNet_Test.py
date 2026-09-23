import numpy as np
import torch
from torchvision import transforms, datasets
from pathlib import Path
import ModeloBMN
import torchmetrics as tm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sb
import matplotlib.pyplot as plt


#Se define el dispositivo en el cual se ejecutara el modelo
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Se carga el modelo, defineindo su arquitectura y cargando los pesos previamente guardados
Ar = "Modelo_Celulas1.pth"
modelo = ModeloBMN.ModeloBMN()
modelo.load_state_dict(torch.load(Ar))
modelo = modelo.to(dispositivo) #Se pasa el modelo a la GPU
modelo.eval() #Se debe poner el modelo en modo evaluacion

#Se extraen las etiquetas de las clases
dirE = Path.cwd()
dirE = dirE / 'Celulas' / 'Dataset' / 'Entrenamiento'
V = []
for x in dirE.iterdir():
    if x.is_dir():
     V.append(x.parts[-1])

print(V)
W = [0,1,2,3,4]
#Se cargan las imagenes
transformacion = transforms.Compose([
    transforms.Resize(224),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.299,0.224,0.225])
])
dirT = Path.cwd()
dirT = dirT / 'Celulas' /'DataSet'/ 'Test'
dataset_test = datasets.ImageFolder(root= dirT, transform=transformacion)

cargar_test = DataLoader(dataset_test,shuffle=False)

#Se definen las metricas
metricas = tm.MetricCollection([
    tm.Accuracy(task = "multiclass",num_classes=5,average="macro"),tm.Precision(task = "multiclass",num_classes=5,average="macro"),
    tm.Recall(task = "multiclass",num_classes=5,average="macro"),tm.Specificity(task = "multiclass",num_classes=5,average="macro"),tm.F1Score(task = "multiclass",num_classes=5,average="macro"),
    tm.ConfusionMatrix(task = "multiclass",num_classes=5)
])

metricas.to(dispositivo)
Y=[]
YP =[]
Mc = np.zeros((5,5)) #Esto hace que funcione
with torch.no_grad():
    for entradas, etiquetas in cargar_test:
        entradas = entradas.to(dispositivo)
        #print(etiquetas)
        etiquetas = etiquetas.to(dispositivo)
        salidas = modelo(entradas)
        az, prediccion = torch.max(salidas.data,1)
        y = etiquetas.cpu()
        yp = prediccion.cpu()
        Y.append(y)
        YP.append(yp)
        Mc += confusion_matrix(y_pred=yp, y_true=y, labels=W) #Acumula las matrices parciales en una matriz de confusion real, no funciona con etiquetas del vector V
        #val = metricas(etiquetas, prediccion)


exactitud = accuracy_score(Y,YP)
precision = precision_score(Y,YP,average='macro')
sensibilidad = recall_score(Y,YP,average='macro')
F1 = 2/(1/sensibilidad+1/precision)
print(Mc)
print(f'Precision: {precision:.4f}\nSensibilidad: {sensibilidad:.4f}\nExactitud: {exactitud:.4f}\nPunruacion F1:{F1:.4f}')

sb.heatmap(Mc,annot=True,xticklabels=V,yticklabels=V,cmap ='OrRd') #Se crea la matriz para mostrar los datos
plt.xlabel('Estado')
plt.ylabel('Prediccion')
plt.title('Matriz de Confusion')
plt.show()





