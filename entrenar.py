import os
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16
import shutil

# Limpiar la sesión de Keras
K.clear_session()

# Directorios de imágenes
datos_entrenamiento = 'Animales'  # Cambiar a la ruta correcta
datos_validacion = 'AnimalesValidacion'  # Cambiar a la ruta correcta

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)

# Función para redimensionar sin distorsión, manteniendo la relación de aspecto
def redimensionar_manteniendo_relacion_aspecto(img, tamaño_max=100):
    altura, ancho = img.shape[:2]
    factor_escala = tamaño_max / float(max(altura, ancho))
    
    nuevo_ancho = int(ancho * factor_escala)
    nueva_altura = int(altura * factor_escala)
    
    # Redimensionar la imagen con una interpolación de alta calidad
    img_redimensionada = cv2.resize(img, (nuevo_ancho, nueva_altura), interpolation=cv2.INTER_AREA)

    # Crear una imagen de fondo cuadrada (100x100) con color negro
    fondo = np.zeros((tamaño_max, tamaño_max, 3), dtype=np.uint8)
    
    # Calcular las coordenadas para centrar la imagen en el fondo
    x_offset = (tamaño_max - nuevo_ancho) // 2
    y_offset = (tamaño_max - nueva_altura) // 2
    
    # Copiar la imagen redimensionada en el centro del fondo negro
    fondo[y_offset:y_offset + nueva_altura, x_offset:x_offset + nuevo_ancho] = img_redimensionada
    
    return fondo

# Mover todas las imágenes de validación a entrenamiento al inicio
for clase in os.listdir(datos_validacion):
    clase_validacion = os.path.join(datos_validacion, clase)
    clase_entrenamiento = os.path.join(datos_entrenamiento, clase)

    if not os.path.exists(clase_entrenamiento):
        os.makedirs(clase_entrenamiento)

    if os.path.isdir(clase_validacion):
        imagenes = os.listdir(clase_validacion)
        for imagen in imagenes:
            origen = os.path.join(clase_validacion, imagen)
            destino = os.path.join(clase_entrenamiento, imagen)

            if not os.path.exists(destino):
                shutil.move(origen, destino)
                print(f'Movida {imagen} de {clase} de la carpeta de validación a la carpeta de entrenamiento')
            else:
                print(f'La imagen {imagen} ya está en la carpeta de entrenamiento')

# Mover un 20% de las imágenes de entrenamiento a validación
for clase in os.listdir(datos_entrenamiento):
    clase_entrenamiento = os.path.join(datos_entrenamiento, clase)
    clase_validacion = os.path.join(datos_validacion, clase)

    if not os.path.exists(clase_validacion):
        os.makedirs(clase_validacion)

    if os.path.isdir(clase_entrenamiento):
        imagenes = os.listdir(clase_entrenamiento)
        total_imagenes = len(imagenes)
        imagenes_a_mover = random.sample(imagenes, int(0.2 * total_imagenes))  # Selecciona un 20%

        for imagen in imagenes_a_mover:
            origen = os.path.join(clase_entrenamiento, imagen)
            destino = os.path.join(clase_validacion, imagen)

            if not os.path.exists(destino):
                shutil.move(origen, destino)
                print(f'Movida {imagen} de {clase} de la carpeta de entrenamiento a la carpeta de validación')
            else:
                print(f'La imagen {imagen} ya está en la carpeta de validación')

print("Proceso de mover un 20% de las imágenes a la carpeta de validación completado.")

# Función para mostrar imágenes aleatorias de un directorio
def mostrar_imagenes(directorio, num_imagenes=5, tamaño_max=80):
    clases = os.listdir(directorio)  # Lista de clases
    imagenes = []

    for clase in clases:
        ruta_clase = os.path.join(directorio, clase)
        if os.path.isdir(ruta_clase):  # Verificar que es una carpeta
            imagenes_clase = os.listdir(ruta_clase)
            if imagenes_clase:
                imagen = random.choice(imagenes_clase)  # Selecciona una imagen aleatoria
                imagenes.append(os.path.join(ruta_clase, imagen))

    # Mostrar imágenes
    fig, axes = plt.subplots(1, min(num_imagenes, len(imagenes)), figsize=(4, 1))
    if len(imagenes) == 1:
        axes = [axes]  # Para evitar errores si solo hay una imagen

    for ax, img_path in zip(axes, imagenes):
        # Leer la imagen con OpenCV
        img = cv2.imread(img_path)
        
        # Redimensionar la imagen manteniendo la relación de aspecto
        img_resized = redimensionar_manteniendo_relacion_aspecto(img, tamaño_max)
        
        # Convertir a escala de grises
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Mostrar la imagen en escala de grises
        ax.imshow(img_gray, cmap='gray')
        ax.set_title(os.path.basename(img_path))
        ax.axis('off')

    plt.show()

# Mostrar imágenes de entrenamiento
print("Imágenes de entrenamiento:")
mostrar_imagenes(datos_entrenamiento, num_imagenes=5)

# Mostrar imágenes de validación
print("Imágenes de validación:")
mostrar_imagenes(datos_validacion, num_imagenes=5)

print("Proceso de mover todas las imágenes de validación a entrenamiento completado.")

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Parámetros del modelo
epocas = 150  # Número de épocas
altura = 100  # Altura de las imágenes de entrada
longitud = 100  # Longitud de las imágenes de entrada
batch_size = 32  # Número de imágenes procesadas por lote
pasos = 194  # Número de pasos por época
pasos_validacion = 34  # Número de pasos para validación
clases = 19  # Número de clases (una para cada clase de animal)
lr = 0.0001  # Tasa de aprendizaje
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Transformación de las imágenes para entrenamiento (Data Augmentation)
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.4,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.7,  # Incrementar el rango de zoom
    horizontal_flip=True,
    brightness_range=[0.2, 1.0],
    channel_shift_range=20.0,
)

# Transformación de las imágenes para validación
validacion_datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza las imágenes para validación
)

# Carga de las imágenes de entrenamiento
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical',
)

# Carga de las imágenes de validación
imagen_validacion = validacion_datagen.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical',
)

# Cargar el modelo preentrenado (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(altura, longitud, 3))

# Congelar las capas base del modelo
base_model.trainable = False

# Crear el modelo con transfer learning (añadiendo nuevas capas)
cnn = Sequential([
    base_model,  # Añadir el modelo base preentrenado
    Flatten(),  # Aplanar la salida de las capas convolucionales
    Dense(512, activation='relu'),
    Dropout(0.5),  # Dropout para evitar el sobreajuste
    Dense(256, activation='relu'),
    Dropout(0.5),  # Otro Dropout
    Dense(clases, activation='softmax')  # Capa de salida
])

# Compilar el modelo
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

# Entrenar el modelo
cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion, callbacks=[early_stop, lr_scheduler, tensorboard_callback])

# Guardar el modelo y los pesos entrenados
directorio = './modelo/'  # Directorio donde se guardarán el modelo y los pesos
if not os.path.exists(directorio):
    os.mkdir(directorio)  # Si no existe, lo creamos

cnn.save('./modelo/modelo.keras')  # Guardamos el modelo
cnn.save_weights('./modelo/pesos.weights.h5')  # Guardamos los pesos del modelo