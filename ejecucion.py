from configuracion import ajuste
from redNeuronal import *

def precisionPerdida(ventana):
    inputs = ajuste(ventana)
    df = filtradoDatos("breast_cancer.csv")
    variables = obtencionVariables(df)
    datos = EscaladaDatos(variables)
    model = entrenamientoNeuronal(datos,inputs['eta'],inputs['generaciones'])
    graficaPrecision(model)

def historialPerdida(ventana):
    inputs = ajuste(ventana)
    df = filtradoDatos("breast_cancer.csv")
    variables = obtencionVariables(df)
    datos = EscaladaDatos(variables)
    model = entrenamientoNeuronal(datos,inputs['eta'],inputs['generaciones'])
    graficaPerdida(model)