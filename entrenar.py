import sys
import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Procesar las imágenes para entrenar el algoritmo
from tensorflow.keras.optimizers import Adam  # Optimizador para entrenar el algoritmo
from tensorflow.keras.models import Sequential  # Librería para hacer redes neuronales secuenciales (capas en orden)
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D  # Las capas con las que haremos el maxpolling
from tensorflow.keras import backend as K  # Hace que se limpie la sesión para empezar el entrenamiento desde una sesión limpia
from tensorflow.keras.callbacks import EarlyStopping # Importamos el EarlyStopping para evitar sobreentrenar el modelo
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

K.clear_session()

# Rutas de las carpetas de entrenamiento y validación
datos_entrenamiento = r'C:\Users\Usuario\Documents\TFG\Train'
datos_validacion = r'C:\Users\Usuario\Documents\TFG\Validacion'

# Porcentaje de imágenes a mover (20%)
porcentaje_a_mover = 0.2

# Mover imágenes de validación a entrenamiento al inicio
# Recorremos cada clase (subcarpeta) dentro de la carpeta de validación
for clase in os.listdir(datos_validacion):
    # Ruta completa de la clase dentro de la carpeta de validación
    clase_validacion = os.path.join(datos_validacion, clase)
    
    # Ruta de la clase correspondiente en la carpeta de entrenamiento
    clase_entrenamiento = os.path.join(datos_entrenamiento, clase)

    # Si la clase no existe en la carpeta de entrenamiento, la creamos
    if not os.path.exists(clase_entrenamiento):
        os.makedirs(clase_entrenamiento)

    # Verificamos que sea un directorio (y no un archivo) en la carpeta de validación
    if os.path.isdir(clase_validacion):
        # Obtenemos la lista de imágenes en la carpeta de la clase
        imagenes = os.listdir(clase_validacion)

        # Movemos todas las imágenes de la carpeta de validación a la carpeta de entrenamiento
        for imagen in imagenes:
            # Rutas completas de las imágenes
            origen = os.path.join(clase_validacion, imagen)
            destino = os.path.join(clase_entrenamiento, imagen)

            # Verificamos si la imagen ya está en la carpeta de entrenamiento
            if not os.path.exists(destino):
                # Movemos la imagen
                shutil.move(origen, destino)
                print(f'Movida {imagen} de {clase} de la carpeta de validación a la carpeta de entrenamiento')
            else:
                print(f'La imagen {imagen} ya está en la carpeta de entrenamiento')

print("Proceso de mover todas las imágenes de validación a entrenamiento completado.")

#Declaramos el early_Stop, ajustando los pesos del sobreajuste
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Recorremos cada clase (subcarpeta) dentro de la carpeta de entrenamiento
for clase in os.listdir(datos_entrenamiento):
    # Ruta completa de la clase dentro de la carpeta de entrenamiento
    clase_entrenamiento = os.path.join(datos_entrenamiento, clase)
    
    # Ruta de la clase correspondiente en la carpeta de validación
    clase_validacion = os.path.join(datos_validacion, clase)

    # Si la clase no existe en la carpeta de validación, la creamos
    if not os.path.exists(clase_validacion):
        os.makedirs(clase_validacion)

    # Verificamos que sea un directorio (y no un archivo) en la carpeta de entrenamiento
    if os.path.isdir(clase_entrenamiento):
        # Obtenemos la lista de imágenes en la carpeta de la clase
        imagenes = os.listdir(clase_entrenamiento)

        # Calculamos cuántas imágenes mover (el 20%)
        cantidad_a_mover = int(len(imagenes) * porcentaje_a_mover)

        # Seleccionamos aleatoriamente las imágenes a mover
        imagenes_a_mover = random.sample(imagenes, cantidad_a_mover)

        # Movemos las imágenes seleccionadas a la carpeta de validación
        for imagen in imagenes_a_mover:
            # Rutas completas de las imágenes
            origen = os.path.join(clase_entrenamiento, imagen)
            destino = os.path.join(clase_validacion, imagen)

            # Verificamos si la imagen ya está en la carpeta de validación
            if not os.path.exists(destino):
                # Movemos la imagen
                shutil.move(origen, destino)
                print(f'Movida {imagen} de {clase} a la carpeta de validación')
            else:
                print(f'La imagen {imagen} ya está en la carpeta de validación')

print("Imagenes movidas")

# Definición de los parámetros del modelo
epocas = 20  # Número de épocas (pasadas completas por el conjunto de datos)
altura = 100  # Altura de las imágenes de entrada
longitud = 100  # Longitud de las imágenes de entrada
batch_size = 16  # Número de imágenes procesadas por lote
pasos = 50  # Número de pasos por época
pasos_validacion = 10  # Número de pasos para validación
filtro_convolucion1 = 32  # Número de filtros en la primera capa convolucional
filtro_convolucion2 = 64  # Número de filtros en la segunda capa convolucional
tamano_filtro1 = (3, 3)  # Tamaño de los filtros de la primera capa
tamano_filtro2 = (2, 2)  # Tamaño de los filtros de la segunda capa
tamano_pool = (2, 2)  # Tamaño del pool (reducción de tamaño de la imagen)
clases = 18  # Número de clases (una para cada letra del alfabeto)
lr = 0.0001  # Tasa de aprendizaje del optimizador
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Transformación de las imágenes para entrenamiento (DATA AUGMENTATION)
entrenamiento_datagen = ImageDataGenerator(
    rescale = 1./255,  # Normaliza las imágenes dividiendo los valores de los píxeles por 255 para que estén entre 0 y 1
    shear_range = 0.5,  # Inclina la imagen para que el algoritmo aprenda a reconocer cuando la imagen esté de lado
    rotation_range = 40, # Rota la imagen para entrenarlo mejor
    width_shift_range = 0.4, # Desplazamiento horizontal de la imagen de hasta un 20%
    height_shift_range = 0.4, # Desplazamiento vertical de la imagen de hasta un 20%
    zoom_range = 0.7,  # Hace zoom en varias imágenes para que el algoritmo aprenda a reconocer objetos a diferentes distancias
    horizontal_flip = True,  # Invierte las imágenes horizontalmente para que la red aprenda a reconocer objetos desde diferentes ángulos
    brightness_range=[0.2,1.0], # cambio de brillo en las imagenes
    channel_shift_range=20.0, # cambio de color de las imagenes
)

# Transformación de las imágenes para validación (solo normalización)
validacion_datagen = ImageDataGenerator(
    rescale = 1./255,  # Normaliza las imágenes para validación
)

# Carga las imágenes de entrenamiento
imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),  # Redimensiona las imágenes a la altura y longitud especificada
    batch_size=batch_size,  # Tamaño del lote de imágenes
    class_mode='categorical',  # La salida es categórica (una etiqueta por clase)
)

print(imagen_entrenamiento.class_indices)  # Muestra las clases de las imágenes de entrenamiento

# Carga las imágenes de validación
imagen_validacion = validacion_datagen.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),  # Redimensiona las imágenes de validación
    batch_size=batch_size,  # Tamaño del lote de imágenes
    class_mode='categorical',  # La salida es categórica
)
# Creamos la red neuronal convolucional
cnn = Sequential()  # La red es secuencial (capas apiladas)

# Añadimos la primera capa convolucional con más filtros
cnn.add(Convolution2D(64, tamano_filtro1, padding='same', input_shape=(altura, longitud, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))  # Capa de maxpooling

# Añadimos la segunda capa convolucional con más filtros
cnn.add(Convolution2D(128, tamano_filtro2, padding='same', activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))  # Capa de maxpooling

# Añadimos la tercera capa convolucional con más filtros
cnn.add(Convolution2D(256, tamano_filtro2, padding='same', activation="relu"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))  # Capa de maxpooling

# Aplanamos la salida de las capas convolucionales
cnn.add(Flatten())  # Aplana la imagen para que pase a la capa densa

# Añadimos una capa densa con 512 neuronas
cnn.add(Dense(512, activation="relu"))

# Añadimos una capa densa con 256 neuronas
cnn.add(Dense(256, activation="relu"))

# Añadimos una capa densa con 128 neuronas
cnn.add(Dense(128, activation="relu"))

# Penaliza los pasos grandes a la hora del aprendizaje haciendo que se eviten los pasos grandes y así el sobreajuste
cnn.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))
cnn.add(Dense(128, activation="relu", kernel_regularizer=l2(0.001)))

# Añadimos Dropout para evitar sobreajuste (se desactivan el 50% de las neuronas)
cnn.add(Dropout(0.7))

# Agrega una capa extra
cnn.add(Dense(128, activation="relu"))
cnn.add(Dropout(0.5))  # Otro Dropout para mayor regularización

# Capa de salida con tantas neuronas como clases (usamos softmax para clasificación multicategórica)
cnn.add(Dense(clases, activation="softmax"))

# Parámetros para optimizar el algoritmo (compilamos el modelo)
cnn.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = lr), metrics=['accuracy'])

# Entrenamos el modelo
cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion, callbacks=[early_stop, lr_scheduler])

# Guardamos el modelo y los pesos entrenados
directorio = './modelo/'  # Directorio donde se guardarán el modelo y los pesos
if not os.path.exists(directorio):
    os.mkdir(directorio)  # Si no existe, lo creamos

cnn.save('./modelo/modelo.keras')  # Guardamos el modelo
cnn.save_weights('./modelo/pesos.weights.h5')  # Guardamos los pesos del modelo
