**BuildBrain — Características principales (Reseña de funciones añadidas)**

Este documento resume las funciones más importantes de la aplicación BuildBrain y destaca las mejoras añadidas en versiones recientes. Está escrito en español y utiliza el formato "Añadido: ..." para mostrar los cambios relevantes.

**Características principales**

- **Añadido: Ejecución empaquetada como .exe**: Empaquetado mediante PyInstaller para distribuir una única aplicación ejecutable en Windows (versión 1.0.0).
- **Añadido: Almacenamiento persistente de proyectos y modelos**: Ahora los modelos y proyectos se guardan en la ruta persistente del sistema (AppData\\Local\\BuildBrain\\projects) cuando se ejecuta como .exe, evitando pérdida de datos temporales.
- **Añadido: Detección mejorada de modelos para pruebas**: La pestaña de pruebas busca modelos tanto en la raíz (modelo.plt) como en la subcarpeta `models/` y admite múltiples formatos (.plt, .pkl, .pth).
- **Añadido: Botón de "Refrescar" en la pestaña de prueba**: Permite recargar manualmente la lista de modelos sin reiniciar la aplicación.
- **Añadido: Soporte para XGBoost en el ejecutable**: Incluye la biblioteca y el DLL necesario para que los modelos XGBoost funcionen en la versión compilada.
- **Añadido: Asistente de creación de proyectos**: Interfaz paso a paso para crear nuevos proyectos y configurar modelos iniciales.
- **Añadido: Cargador de datasets y preprocesado**: Herramientas en la GUI para subir conjuntos de datos y aplicar preprocesamiento automático.
- **Añadido: Controles de entrenamiento y monitoreo**: Panel para arrancar/pausar entrenamientos, ver progreso y visualizar métricas y logs en tiempo real.
- **Añadido: Visualización de logs y métricas**: Gráficas y paneles que muestran históricos de entrenamiento y resultados de evaluación.
- **Añadido: Selector de modelo en la interfaz**: Permite elegir fácilmente modelos entrenados para evaluarlos o probarlos con nuevos datos.
- **Añadido: Localización básica**: Archivos `locales/en.json` y `locales/es.json` para soportar la interfaz en inglés y español.
- **Añadido: Manejo de errores y validaciones**: Mejoras en la detección y manejo de errores durante importación de paquetes (XGBoost, PyTorch) y operaciones de E/S.

**Resumen de mejoras técnicas**

- **Añadido: Comandos de compilación reproducibles**: Uso de PyInstaller con flags para incluir `locales`, `config` y dependencias nativas (por ejemplo xgboost.dll).
- **Añadido: Detección de contexto de ejecución**: Funciones que detectan `sys.frozen` para cambiar rutas entre ejecución en desarrollo (script) y ejecución empaquetada (.exe).
- **Añadido: Soporte para múltiples formatos de modelo**: Lectura y carga de modelos guardados en varios formatos comunes (.plt, .pkl, .pth).

**Funcionalidades del usuario (resumen rápido)**

- Crear y gestionar proyectos desde la GUI.
- Subir, preprocesar y validar datasets.
- Entrenar modelos con PyTorch / scikit-learn / XGBoost.
- Guardar y cargar modelos entrenados de forma persistente.
- Probar modelos desde la pestaña de evaluación con opción de refrescar lista.
- Visualizar métricas y registros de entrenamiento.
- Cambiar idioma de la interfaz entre inglés y español.

**Notas finales y próximos pasos sugeridos**

- Si deseas, puedo generar un instalador (MSI/NSIS) para distribuir `main.exe` con instalador o crear un paquete portable.
- Puedo ampliar el README con instrucciones de uso paso a paso, ejemplos de entrenamiento, y comandos para recompilar el ejecutable.

Si quieres que adapte el tono o que añada secciones (instalación, uso rápido, contribución, licencias), dime cómo prefieres que lo organice.
