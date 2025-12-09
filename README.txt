================================================================================
                              BUILDBRAIN
                    AI/ML Model Trainer - Desktop Application
================================================================================

DESCRIPCION DEL PROYECTO
================================================================================

BuildBrain es una aplicación de escritorio intuitiva y amigable para crear,
entrenar y gestionar modelos de aprendizaje automático sin necesidad de
experiencia previa en código. Proporciona una interfaz tipo asistente 
(wizard) que guía al usuario paso a paso a través del proceso completo
de creación de un modelo ML.

La aplicación soporta múltiples tipos de modelos:
- Aprendizaje Supervisado (Regresión y Clasificación)
- Aprendizaje No Supervisado (Clustering)
- Detección de Anomalías
- Redes Neuronales Profundas (Deep Learning)

Permite a los usuarios cargar datasets, definir clases, configurar 
hiperparámetros y monitorear el entrenamiento en tiempo real.


HERRAMIENTAS Y TECNOLOGÍAS UTILIZADAS
================================================================================

Lenguaje Principal:
  - Python 3.13

Frameworks y Librerías:
  - Flet 0.21.0+ (Framework GUI multiplataforma)
  - scikit-learn 1.3.0+ (Modelos de ML clásicos)
  - PyTorch 2.0.0+ (Deep Learning)
  - TensorFlow/Keras (Alternativa para redes neuronales)
  - XGBoost 2.0.0+ (Gradient Boosting)
  - pandas 2.0.0+ (Manipulación de datos)
  - NumPy 1.24.0+ (Operaciones numéricas)
  - Pillow 10.0.0+ (Procesamiento de imágenes)
  - matplotlib 3.7.0+ (Visualización)
  - Plotly 5.15.0+ (Gráficas interactivas)
  - joblib 1.3.0+ (Serialización de modelos)
  - ONNX 1.14.0+ (Exportación de modelos)
  - imbalanced-learn 0.11.0+ (Balanceo de datasets)

Utilidades:
  - pyperclip 1.8.2+ (Manejo del portapapeles)
  - tkinter (Diálogos de selección de archivos)


PROBLEMA RESUELTO
================================================================================

El aprendizaje automático es una tecnología poderosa pero compleja que requiere
conocimientos profundos en programación, matemáticas y ciencia de datos.
Esto crea una barrera significativa para:

- Profesionales no técnicos que desean explorar ML
- Estudiantes que buscan entender conceptos de ML de forma práctica
- Equipos pequeños sin especialistas en data science
- Empresas que quieren prototipar modelos rápidamente

BuildBrain elimina estas barreras al proporcionar:
- Una interfaz gráfica intuitiva y moderna
- Un asistente paso a paso que no requiere código
- Validación automática de configuraciones
- Visualización en tiempo real del entrenamiento
- Exportación de modelos en múltiples formatos
- Gestión automática de proyectos


INSTRUCCIONES PARA EJECUTAR EL PROYECTO
================================================================================

REQUISITOS PREVIOS:
  - Python 3.10 o superior instalado
  - pip (gestor de paquetes de Python)
  - Git (opcional, para clonar el repositorio)

INSTALACION:

1. Clonar o descargar el repositorio:
   git clone <url-del-repositorio>
   cd BuildBrain

2. Crear un entorno virtual (recomendado):
   python -m venv .venv

3. Activar el entorno virtual:
   
   En Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   
   En Windows (CMD):
   .venv\Scripts\activate
   
   En macOS/Linux:
   source .venv/bin/activate

4. Instalar dependencias:
   pip install -r requirements.txt

   Nota: La primera instalación puede tomar algunos minutos,
   especialmente las librerías PyTorch y TensorFlow.

EJECUCION:

1. Con el entorno virtual activado, ejecutar:
   python main.py

2. La aplicación abrirá en una ventana de escritorio.

3. Para crear un nuevo modelo:
   - Haz clic en "Crear Nuevo Modelo"
   - Sigue los 6 pasos del asistente
   - Carga tus datos en formato CSV o selecciona carpetas con imágenes

4. Para abrir un proyecto existente:
   - Selecciona un proyecto de la lista de recientes
   - O haz clic en "Abrir Modelo" para navegar


ESTRUCTURA DEL PROYECTO
================================================================================

BuildBrain/
├── main.py                      # Punto de entrada de la aplicación
├── requirements.txt             # Dependencias del proyecto
├── README.txt                   # Este archivo
├── logo.png                     # Logo de la aplicación
├── assets/                      # Recursos (imágenes, iconos)
│   ├── logo.png
│   └── logo.ico
├── src/                         # Código fuente
│   ├── __init__.py
│   ├── gui/                     # Interfaz gráfica (Flet)
│   │   ├── welcome_screen.py
│   │   ├── main_window.py
│   │   ├── new_project_wizard.py
│   │   ├── dataset_uploader.py
│   │   ├── model_selector.py
│   │   ├── task_selector.py
│   │   ├── class_manager.py
│   │   ├── training_controls.py
│   │   ├── logs_visualization.py
│   │   ├── settings_panel.py
│   │   └── project_list.py
│   ├── core/                    # Lógica central
│   │   ├── dataset_manager.py
│   │   ├── project_config.py
│   │   └── pytorch_trainer.py
│   ├── training/                # Entrenamiento de modelos
│   │   ├── models.py
│   │   ├── trainer.py
│   │   ├── model_trainer.py
│   │   └── model_factory.py
│   ├── data/                    # Procesamiento de datos
│   │   └── preprocessor.py
│   └── utils/                   # Utilidades
│       └── helpers.py
├── projects/                    # Proyectos guardados (se crea automáticamente)
├── config/                      # Archivos de configuración
│   └── Latrix.json
├── locales/                     # Archivos de idioma
│   ├── en.json
│   └── es.json
└── .vscode/                     # Configuración de VS Code (opcional)


CARACTERISTICAS PRINCIPALES
================================================================================

Asistente de Creacion (6 pasos):
  1. Nombre del Proyecto
  2. Configuracion de Tareas (Tipo de modelo, parametros)
  3. Gestion de Clases (Definir categorias)
  4. Clases Balanceadas (Verificar distribucion)
  5. Carga de Dataset (Subir imagenes o datos)
  6. Entrenamiento y Logs (Monitorear progreso)

Gestion de Proyectos:
  - Crear nuevos proyectos
  - Abrir proyectos existentes
  - Eliminar proyectos
  - Vista de proyectos recientes

Capacidades de Entrenamiento:
  - Configuracion de hiperparametros
  - Entrenamiento rapido y avanzado
  - Visualizacion de metricas en tiempo real
  - Exportacion de modelos (joblib, ONNX)
  - Copia de logs para documentacion


RESOLUCIÓN DE PROBLEMAS
================================================================================

Problema: "ModuleNotFoundError: No module named..."
Solucion: Asegúrate de haber instalado todas las dependencias con:
  pip install -r requirements.txt

Problema: La aplicación se abre pero los iconos no cargan
Solucion: Asegúrate de tener la carpeta "assets" en el directorio raiz
  con los archivos logo.png y logo.ico

Problema: Error al cargar imágenes
Solucion: Verifica que las imágenes tengan formatos soportados (PNG, JPG, JPEG, GIF, BMP)

Problema: El entrenamiento es lento
Solucion: Esto es normal en la primera ejecucion. Considera:
  - Usar menos epochs la primera vez
  - Reducir el tamaño del dataset
  - Usar GPU si está disponible


CONTACTO Y SOPORTE
================================================================================

Creador: Glienware
Repositorio: https://github.com/Glienware/BuildBrain
Fecha de Creacion: 2025

Para reportar errores o sugerir mejoras, por favor crea un issue en el
repositorio de GitHub.


NOTAS FINALES
================================================================================

Esta aplicación está en desarrollo activo. Algunas características pueden
cambiar o mejorarse en futuras versiones. Se recomienda mantener un respaldo
de tus proyectos importantes.

La primera ejecucion puede tomar tiempo adicional mientras se descargan
e instalan las dependencias necesarias.

¡Gracias por usar BuildBrain!

================================================================================