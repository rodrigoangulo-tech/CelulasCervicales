import os
from pathlib import Path
import random
import shutil
import math

Carpetas = ['Entrenamiento', 'Test'] # Carpetas que deben hacerse parala red neuronal
dir = Path.cwd()
print(dir)
diri= dir / 'Celulas' / 'Enhanced Cervical Cytology Image dataset'  #Directorio inicial para la distribucion
print(diri)
dirf = Path.cwd()
dirf = dirf / 'Celulas' / 'DataSet' #Directorio final donde se guardaran los archivos
print(dirf)

#En la libreria pathlib, las rutas son objetos con metodos y propiedades, no simples cadenas de caracteres... eso es obsoleto!
#Creacion de Carpetas, solo es necesario que corra una vez.

def Creacion_Carpetas(Carpetas):
    for carpeta in Carpetas:
        nueva_car = dirf / carpeta
        nueva_car.mkdir(exist_ok=True)
        print('Se crean las carpetas: ')# Crea la carpeta de la lista de carpetas, exist_ok = True evita que ponga error si la carpeta ya esta creada
        for x in diri.iterdir():
            if x.is_dir():
                subcar = nueva_car / x.parts[-1] #parts es una tupla que divide las partes de una ruta, aqui se agarra la ultima
                subcar.mkdir(exist_ok=True) #Crea las subcarpetas con el nombre de las clases
                print(subcar) #muestra las carpetas creadas

#Conteo de Archivos
def Conteo_Archivos():
    Ndo = []  # Numero de Objetos
    k = 0  # Contador
    for x in diri.iterdir():
        if x.is_dir(): #Si el objeto es una carpeta dentro del directorio inicial
            k = 0
            for c in x.iterdir(): #Cuenta los archivos dentro de cada directorio
                k += 1
            Ndo.append(k)
    limite = min(Ndo) #El numero de menor de todas las clases es el limite para las demas
    print(limite)
    return limite

def Distribucion_Archivos(limite):
    k = 0
    listarch = []
    listafin = []
    for x in diri.iterdir(): #Itera sobre las clases
        listarch = [] #Reinicia las lista
        listafin = []
        listaEntr = []
        listaTest = []
        if x.is_dir():
            for archivo in x.iterdir(): #Agarra todos los arhcivos en una lista
                if archivo.is_file():
                    listarch.append(archivo)

            for k in range(limite):
                Nuar = random.choice(listarch) #Nuevo archivo
                indice = listarch.index(Nuar)
                listarch.pop(indice)
                listafin.append(Nuar) #Elige archivos aleatoriamente en una nueva lista hasta que llegue al limite

            random.shuffle(listafin) #Mezcla aleatoriamente la lista
            #print(listafin)
            print(len(listafin))
            div = math.floor(0.8*limite)
            listaEntr = listafin [:div] #Crea la lista de entrenamiento
            listaTest = listafin [div:] #Crea la lista de test
            print(f'Ent:{len(listaEntr)} Test: {len(listaTest)}')
            #Copia los archivos en las listas en el lugar correspondiente

            y = dirf / Carpetas[0] / x.parts[-1]  # Ruta final Entrenamiento
            for ar1 in listaEntr: #archivo
                shutil.copy(ar1,y)


            y = dirf / Carpetas[1] / x.parts[-1] #Ruta final Test
            for ar2 in listaTest:
                shutil.copy(ar2,y)

Creacion_Carpetas(Carpetas)
limite = Conteo_Archivos()
Distribucion_Archivos(limite)


