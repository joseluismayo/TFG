import os
import shutil
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar el modelo preentrenado MobileNetV2
modelo = MobileNetV2(weights='imagenet')

# Ruta donde están las imágenes no organizadas
ruta_imagenes = r'C:\Users\Usuario\Downloads\Wild Animals detection.v1i.tensorflow\train'  # Ruta donde están las imágenes no organizadas

ruta_gatos = 'C:/Users/Usuario/Documents/TFG/Train/Gato'  # Ruta para las imágenes de gatos
ruta_perros = 'C:/Users/Usuario/Documents/TFG/Train/Perro'  # Ruta para las imágenes de perros
ruta_ardilla = 'C:/Users/Usuario/Documents/TFG/Train/Ardilla'
ruta_cerdos = 'C:/Users/Usuario/Documents/TFG/Train/Cerdo'
ruta_jabalis = 'C:/Users/Usuario/Documents/TFG/Train/Jabali'
ruta_ciervos = 'C:/Users/Usuario/Documents/TFG/Train/Ciervo'
ruta_lince = 'C:/Users/Usuario/Documents/TFG/Train/Lince'
ruta_corzo = 'C:/Users/Usuario/Documents/TFG/Train/Corzo'
ruta_lagarto = 'C:/Users/Usuario/Documents/TFG/Train/Lagarto'
ruta_tejon = 'C:/Users/Usuario/Documents/TFG/Train/Tejon'
ruta_zorro = 'C:/Users/Usuario/Documents/TFG/Train/Zorro'
ruta_rata = 'C:/Users/Usuario/Documents/TFG/Train/Rata'
ruta_bison = 'C:/Users/Usuario/Documents/TFG/Train/Bison'
ruta_nutria = 'C:/Users/Usuario/Documents/TFG/Train/Nutria'
ruta_foca = 'C:/Users/Usuario/Documents/TFG/Train/Foca'
ruta_lobo = 'C:/Users/Usuario/Documents/TFG/Train/Lobo'
ruta_vaca = 'C:/Users/Usuario/Documents/TFG/Train/Vaca'
ruta_toro = 'C:/Users/Usuario/Documents/TFG/Train/Toro'
ruta_raton = 'C:/Users/Usuario/Documents/TFG/Train/Raton'
ruta_cabra = 'C:/Users/Usuario/Documents/TFG/Train/Cabra'
ruta_oveja = 'C:/Users/Usuario/Documents/TFG/Train/Oveja'
ruta_bisonte = 'C:/Users/Usuario/Documents/TFG/Train/Bisonte'
ruta_lagartija = 'C:/Users/Usuario/Documents/TFG/Train/Lagartija'
ruta_sapo = 'C:/Users/Usuario/Documents/TFG/Train/Sapo'
ruta_rana = 'C:/Users/Usuario/Documents/TFG/Train/Rana'
ruta_caballo = 'C:/Users/Usuario/Documents/TFG/Train/Caballo'
ruta_gamo = 'C:/Users/Usuario/Documents/TFG/Train/Gamo'
ruta_conejo = 'C:/Users/Usuario/Documents/TFG/Train/Conejo'
ruta_liebre = 'C:/Users/Usuario/Documents/TFG/Train/Liebre'
ruta_ganso = 'C:/Users/Usuario/Documents/TFG/Train/Ganso'

animal_map = {
    'cat': ruta_gatos,
    'dog': ruta_perros,
    'squirrel': ruta_ardilla,
    'pig': ruta_cerdos,
    'wild boar': ruta_jabalis,
    'deer': ruta_ciervos,
    'lynx': ruta_lince,
    'roe deer': ruta_corzo,
    'lizard': ruta_lagarto,
    'badger': ruta_tejon,
    'fox': ruta_zorro,
    'rat': ruta_rata,
    'bison': ruta_bison,
    'otter': ruta_nutria,
    'seal': ruta_foca,
    'wolf': ruta_lobo,
    'cow': ruta_vaca,
    'bull': ruta_toro,
    'mouse': ruta_raton,
    'goat': ruta_cabra,
    'sheep': ruta_oveja,
    'bison': ruta_bisonte,
    'lizard': ruta_lagartija,
    'toad': ruta_sapo,
    'frog': ruta_rana,
    'horse': ruta_caballo,
    'stag': ruta_gamo,
    'rabbit': ruta_conejo,
    'hare': ruta_liebre,
    'goose': ruta_ganso
}

# Función para mover la imagen a la carpeta correspondiente según la clase predicha
def mover_imagen_a_carpeta(clase_predicha, img_path, imagen_nombre):
    clase_predicha = clase_predicha.lower()  # Convertir a minúsculas

    # Verifica si la clase está en el mapa de animales
    for animal, carpeta in animal_map.items():
        if animal in clase_predicha:
            destino_img_path = os.path.join(carpeta, imagen_nombre)
            shutil.move(img_path, destino_img_path)
            print(f'{imagen_nombre} movida a la carpeta "{animal}"')
            return

    # Si no se encuentra la clase en el mapa
    print(f'{imagen_nombre} no es un animal reconocido, no se movió.')

# Listar todas las imágenes en la carpeta
imagenes = [f for f in os.listdir(ruta_imagenes) if f.endswith(('jpeg', 'png', 'jpg'))]

# Función para preprocesar la imagen
def preprocesar_imagen(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar la imagen
    img_array = image.img_to_array(img)  # Convertir la imagen a un array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Agregar la dimensión de batch
    img_array = preprocess_input(img_array)  # Normalizar la imagen
    return img_array

# Cargar el modelo MobileNetV2 preentrenado
modelo = MobileNetV2(weights='imagenet')

# Iterar sobre todas las imágenes y clasificarlas
for imagen_nombre in imagenes:
    img_path = os.path.join(ruta_imagenes, imagen_nombre)
    
    # Preprocesar la imagen
    img_array = preprocesar_imagen(img_path)
    
    # Hacer la predicción
    predicciones = modelo.predict(img_array)
    
    # Decodificar las predicciones
    predicciones_decodificadas = decode_predictions(predicciones, top=1)[0]
    
    # Obtener el nombre de la clase con mayor probabilidad
    clase_predicha = predicciones_decodificadas[0][1]  # El nombre de la clase
    
    print(f'Predicción para {imagen_nombre}: {clase_predicha}')
    
    # Llamar a la función para mover la imagen a la carpeta correspondiente basada en la clase predicha
    mover_imagen_a_carpeta(clase_predicha, img_path, imagen_nombre)