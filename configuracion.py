import pandas as pd

def leerArchivo(ruta):
    dataset = pd.read_csv(ruta)
    return dataset

def ajuste(ventana): #Obtenemos los datos de la vista
    valores = {}
    try:
        if ventana.radioEta.isChecked():
            eta = 0.01
        else:
            eta = float(ventana.inputEta.text())

        if ventana.radioUmbral.isChecked():
            generaciones = 100
        else:
            generaciones = int(ventana.inputGeneraciones.text())

        valores = {
            "eta": eta,
            "generaciones": generaciones
        }

    except:
        print("datos incompletos")
        pass

    return valores