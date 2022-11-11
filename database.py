from deta import Deta  # pip install deta
from skimage.io import imread,imshow,imsave
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# Initialize with a project key
DETA_KEY = "e0aeitf6_7Ez5rjjPeN12TfdyAbN34KdzT7Jse29P"
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("MetadataPACS")
dv = deta.Drive('DLO_prueba')

# FUNCIONES DE LA BASE DE DATOS
def insert_met(id, name, age, te, r, op, neu, rad):
    """Returns the user on a successful user creation, otherwise raises and error"""
    return db.put({"key": id, "nombre": name, "edad": age, 'tipo_estudio' : te, 'razon': r,
    'operario': op, 'm_neumologo': neu, 'm_radiologo': rad})
def get_met(period):
    """If not found, the function will return None"""
    return db.get(period)
# FUNCIONES DEL DRIVE
def upload(imname, file):
    # Subimos una Imagen
    return dv.put(name = imname, path =file)
def get_image(name):
    # Obtenemos una Imagen cuando la pedimos por el nombre
    return dv.get(name)
def list_images():
    # Lista de las Imágenes que guarda
    return dv.list()

# FUNCIONES PARA INTEGRACIÓN
def lst_nombres(): # Lista de Nombres
    res = db.fetch()
    df = pd.DataFrame(res.items)
    #display(df)

    lst_names = list(df['nombre'])
    lst_id = list(df['key'])
    return lst_names, lst_id

def get_radiog(id): # Obtenemos la Imagen
    iml = get_image(id)
    content = iml.read().decode("utf-8") 
    iml.close()

    glst = []
    content = content.replace('\r', '')
    imlst = content.split('\n')
    imlst = imlst[:len(imlst)-1]

    for lst in imlst:
        l = lst.split(' ')
        l = list(map(float, l))
        glst.append(l)

    im = np.array(glst)
    return im
