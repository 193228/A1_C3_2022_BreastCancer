from tensorflow import keras

from configuracion import leerArchivo
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

def filtradoDatos(ruta):
    df = leerArchivo(ruta) #leemos el dataset del cancer de mama
    df.isnull().sum() #comprobamos si existen valores nulos
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True) #comprobamos que la columna esta vacia y lo eliminamos
    return df

def obtencionVariables(df): #obtenemos el df filtrado
    # Variable Independiente
    x = df.drop('diagnosis', axis=1)
    # Variable Dependiente
    y = df.diagnosis
    #convertimos los datos categoricos de vy al formato binario para su posterior proceso
    lb = LabelEncoder()
    y = lb.fit_transform(y)
    return {"x":x,"y":y}

def EscaladaDatos(datos):
    xtrain, xtest, ytrain, ytest = train_test_split(datos['x'], datos['y'], test_size=0.3, random_state=40)
    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.transform(xtest)
    return {"xtrain":xtrain,"xtest":xtest,"ytrain":ytrain,"ytest":ytest}

def entrenamientoNeuronal(entrenamiento,eta,generaciones):
    classifier = Sequential()
    classifier.add(Dense(units=9, kernel_initializer='he_uniform', activation='relu', input_dim=30))
    classifier.add(Dense(units=9, kernel_initializer='he_uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    classifier.summary()
    classifier.compile(keras.optimizers.Adam(learning_rate=eta), loss='binary_crossentropy', metrics=['accuracy'])
    model = classifier.fit(entrenamiento['xtrain'], entrenamiento['ytrain'], batch_size=100, epochs=generaciones)
    y_pred = classifier.predict(entrenamiento['xtest'])
    y_pred = (y_pred > 0.5)
    return model

def graficaPrecision(model):
    plt.plot(model.history['accuracy'])
    plt.title('modelo de precision')
    plt.ylabel('Precision')
    plt.xlabel('epocas')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def graficaPerdida(model):
    plt.plot(model.history['loss'])
    plt.title('modelo de perdidas')
    plt.ylabel('perdidas')
    plt.xlabel('epocas')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()