import sys
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Procesar las imágenes para entrenar el algoritmo
from tensorflow.keras.optimizers import Adam  # Optimizador para entrenar el algoritmo
from tensorflow.keras.models import Sequential, Model  # Model es necesario para usar transfer learning
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D  # Las capas con las que haremos el maxpolling
from tensorflow.keras import backend as K  # Hace que se limpie la sesión para empezar el entrenamiento desde una sesión limpia
from tensorflow.keras.callbacks import EarlyStopping  # Importamos el EarlyStopping para evitar sobreentrenar el modelo
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import VGG16  # Carga de VGG16 como modelo preentrenado

K.clear_session()

# Rutas de las carpetas de entrenamiento y validación
datos_entrenamiento = r'C:\Users\Usuario\Documents\TFG\Train'
datos_validacion = r'C:\Users\Usuario\Documents\TFG\Validacion'

# Porcentaje de imágenes a mover (20%)
porcentaje_a_mover = 0.2

# Mover imágenes de validación a entrenamiento al inicio
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

print("Proceso de mover todas las imágenes de validación a entrenamiento completado.")

# EarlyStopping para evitar sobreajuste
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Recorremos cada clase (subcarpeta) dentro de la carpeta de entrenamiento
for clase in os.listdir(datos_entrenamiento):
    clase_entrenamiento = os.path.join(datos_entrenamiento, clase)
    clase_validacion = os.path.join(datos_validacion, clase)

    if not os.path.exists(clase_validacion):
        os.makedirs(clase_validacion)

    if os.path.isdir(clase_entrenamiento):
        imagenes = os.listdir(clase_entrenamiento)
        cantidad_a_mover = int(len(imagenes) * porcentaje_a_mover)
        imagenes_a_mover = random.sample(imagenes, cantidad_a_mover)
        for imagen in imagenes_a_mover:
            origen = os.path.join(clase_entrenamiento, imagen)
            destino = os.path.join(clase_validacion, imagen)

            if not os.path.exists(destino):
                shutil.move(origen, destino)
                print(f'Movida {imagen} de {clase} a la carpeta de validación')
            else:
                print(f'La imagen {imagen} ya está en la carpeta de validación')

print("Imagenes movidas")

# Parámetros del modelo
epocas = 20  # Número de épocas
altura = 100  # Altura de las imágenes de entrada
longitud = 100  # Longitud de las imágenes de entrada
batch_size = 64  # Número de imágenes procesadas por lote
pasos = 94  # Número de pasos por época
pasos_validacion = 10  # Número de pasos para validación
clases = 19  # Número de clases (una para cada clase de animal)
lr = 0.0001  # Tasa de aprendizaje
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Transformación de las imágenes para entrenamiento (Data Augmentation)
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.5,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    zoom_range=0.9,  # Incrementar el rango de zoom
    horizontal_flip=True,
    brightness_range=[0.2, 1.0],
    channel_shift_range=20.0,
)


# Transformación de las imágenes para validación
validacion_datagen = ImageDataGenerator(
    rescale = 1./255,  # Normaliza las imágenes para validación
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
cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion, callbacks=[early_stop, lr_scheduler])

# Guardar el modelo y los pesos entrenados 
directorio = './modelo/'  # Directorio donde se guardarán el modelo y los pesos
if not os.path.exists(directorio):
    os.mkdir(directorio)  # Si no existe, lo creamos

cnn.save('./modelo/modelo.keras')  # Guardamos el modelo
cnn.save_weights('./modelo/pesos.weights.h5')  # Guardamos los pesos del modelo
