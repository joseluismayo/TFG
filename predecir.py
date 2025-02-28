import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud = 100
altura = 100
modelo = './modelo/modelo.keras'
pesos = './modelo/pesos.weights.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

animal = [
    "ardilla",
    "bisonte",
    "caballo",
    "cabra",
    "cerdo",
    "ciervo",
    "conejo",
    "corzo",
    "gato",
    "jabali",
    "liebre",
    "lince",
    "lobo",
    "oveja",
    "perro",
    "tejon",
    "toro",
    "vaca",
    "zorro"
]



def predict(nombre_imagen):
    x = load_img(nombre_imagen, target_size=(longitud,altura))
    #conjunto de valores que referencia a la imagen
    x = img_to_array(x)
    #en la primera dimension, añadimos dimension extra para poder procesar la informacion sin problema
    x = np.expand_dims(x, axis=0)
    #la variable que guarda la prediccion de lo que cree que es la imagen
    prediccion = cnn.predict(x)
    #cogemos la primera dimension que es la que tiene la solucion y lo que nos interesa
    resultado = prediccion[0]
    #nos devuelve el indice de mayor valor
    respuesta = np.argmax(resultado)

    animal_predicho = animal[respuesta]

    print(f"El animal es: {animal_predicho}")

predict('perro.jpg')
        


