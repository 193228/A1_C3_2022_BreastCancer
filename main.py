import sys
import PyQt5
from vista.vistaPrincipal import Ui_MainWindow as ventanaPrincipal
from ejecucion import precisionPerdida, historialPerdida

class MyApp(PyQt5.QtWidgets.QMainWindow, ventanaPrincipal):
    def __init__(self):
        PyQt5.QtWidgets.QMainWindow.__init__(self)
        ventanaPrincipal.__init__(self)
        self.setupUi(self)
        acciones(self)

def acciones(ventana):
    ventana.botonPrecision.clicked.connect(lambda: precisionPerdida(ventana))
    ventana.botonPerdida.clicked.connect(lambda: historialPerdida(ventana))

if __name__ == '__main__':
    app = PyQt5.QtWidgets.QApplication(sys.argv)  # crea un objeto de aplicación (Argumentos de sys)
    window = MyApp()
    window.show()
    window.setFixedSize(window.size())
    app.exec_()
