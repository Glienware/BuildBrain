# 📊 Guía: Cómo Cargar Datasets en BuildBrain

## Opciones de Carga de Datos

BuildBrain soporta múltiples formas de cargar datasets para entrenar tus modelos:

---

## 1️⃣ Cargar Dataset CSV/Excel (Para Datos Tabulares)

### Recomendado para:
- ✅ Datos tabulares (filas y columnas)
- ✅ Datasets supervisados (con etiquetas)
- ✅ Datasets no supervisados (sin etiquetas)
- ✅ Detección de anomalías

### Paso a paso:

1. **En el Wizard**, en el paso **"Dataset Upload"**:
   - Haz click en el botón **"📊 Load CSV Dataset"**

2. **Selecciona tu archivo**:
   - Formato: `.csv`, `.xlsx`, `.xls`
   - Ejemplo de estructura CSV:

   ```csv
   feature1,feature2,feature3,label
   1.5,2.3,3.1,A
   2.1,3.4,1.9,B
   1.8,2.5,2.9,A
   3.2,1.5,2.1,C
   ```

3. **Características automáticas**:
   - ✅ Detección automática de columnas
   - ✅ Preprocesamiento automático (normalización)
   - ✅ División automática train/test (80/20)

---

## 2️⃣ Cargar Imágenes por Clases (Para Visión por Computadora)

### Recomendado para:
- ✅ Clasificación de imágenes
- ✅ Deep Learning (CNN, ResNet)
- ✅ Datasets con múltiples clases

### Paso a paso:

1. **En el Wizard**, en el paso **"Classes Setup"**:
   - Define tus clases (ej: `gato`, `perro`, `pájaro`)

2. **En el paso "Dataset Upload"**:
   - Para cada clase, haz click en **"Select Images"** o **"Select Folder"**

3. **Opción A: Select Images**:
   - Selecciona múltiples imágenes de una vez
   - Formatos: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`

4. **Opción B: Select Folder**:
   - Selecciona una carpeta con todas las imágenes
   - Busca recursivamente en subcarpetas
   - Ej:
     ```
     mi_dataset/
     ├── gatos/
     │   ├── gato1.jpg
     │   ├── gato2.jpg
     │   └── gato3.jpg
     ├── perros/
     │   ├── perro1.jpg
     │   └── perro2.jpg
     └── pajaros/
         ├── pajaro1.jpg
         └── pajaro2.jpg
     ```

5. **Vista previa**:
   - Verás miniaturas de las primeras imágenes
   - Contador automático por clase

---

## 📋 Formatos Soportados

### Archivos CSV/Excel
| Formato | Extensión | Soporte |
|---------|-----------|---------|
| CSV | `.csv` | ✅ Total |
| Excel | `.xlsx` | ✅ Total |
| Excel 97 | `.xls` | ✅ Total |
| JSON | `.json` | ⚠️ Próximamente |
| Parquet | `.parquet` | ⚠️ Próximamente |

### Imágenes
| Formato | Extensión | Soporte |
|---------|-----------|---------|
| PNG | `.png` | ✅ Total |
| JPEG | `.jpg` / `.jpeg` | ✅ Total |
| GIF | `.gif` | ✅ Total |
| BMP | `.bmp` | ✅ Total |

---

## 🔧 Requisitos del Dataset

### Para Supervisados (Clasificación/Regresión)
```
Mínimo: 20-30 muestras por clase
Óptimo: 100+ muestras por clase
Columnas: Features + 1 columna Target (label)
```

### Para No Supervisados (Clustering)
```
Mínimo: 10+ muestras
Óptimo: 100+ muestras
Columnas: Solo features (sin labels)
```

### Para Deep Learning (Imágenes)
```
Mínimo: 10-20 imágenes por clase
Óptimo: 100+ imágenes por clase
Resolución: Mínimo 32x32 píxeles
Tamaño archivo: No debe exceder 500MB total
```

---

## ✅ Checklist Antes de Cargar

- [ ] Archivo en formato correcto (CSV, XLSX, imágenes)
- [ ] No hay caracteres especiales en nombres de archivos
- [ ] No hay espacios al inicio/final de valores
- [ ] Dataset tiene las mismas columnas en todas las filas
- [ ] Imágenes tienen nombres descriptivos (sin caracteres especiales)
- [ ] Ningún archivo corrupto o vacío

---

## 🐛 Troubleshooting

### ❌ "Error: No se puede cargar el archivo"
**Solución:**
- Verifica que el archivo no esté abierto en otro programa
- Comprueba que el formato sea correcto (.csv, .xlsx)
- Intenta renombrar el archivo sin espacios especiales

### ❌ "Error: No se cargan las imágenes de la carpeta"
**Solución:**
- Verifica que la carpeta contiene archivos de imagen
- Comprueba que las extensiones son correctas (.jpg, .png, etc.)
- Asegúrate que no hay subcarpetas con nombres especiales
- Prueba con la opción "Select Images" en lugar de "Select Folder"

### ❌ "Error: Memoria insuficiente"
**Solución:**
- Reduce el tamaño de las imágenes (resize a 256x256)
- Reduce el batch size en el entrenamiento
- Usa un dataset más pequeño para probar primero
- Cierra otros programas

### ❌ "Error: No aparecen los datos en el vista previa"
**Solución:**
- Actualiza el wizard con el botón "Next"
- Cierra y reabre el programa
- Verifica que el archivo no esté vacío

---

## 💡 Tips y Mejores Prácticas

### 1. Organiza tu Dataset
```
✅ BIEN:
dataset/
├── train/
│   ├── class_A/
│   ├── class_B/
│   └── class_C/
└── test/
    ├── class_A/
    ├── class_B/
    └── class_C/

❌ MAL:
dataset/
├── img1.jpg
├── img2.jpg
├── img3.jpg
└── labels.txt
```

### 2. Normaliza Tus Datos
```python
# BuildBrain lo hace automáticamente, pero bueno saber:
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("datos.csv")
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)
```

### 3. Balance tu Dataset
```
Ideal: Mismo número de muestras por clase
Ejemplo:
- Clase A: 100 imágenes
- Clase B: 100 imágenes
- Clase C: 100 imágenes

Si está desbalanceado, BuildBrain puede:
✅ Usar "Balance Dataset" en el wizard
✅ Aplicar pesos a las clases automáticamente
```

### 4. Tamaño Recomendado
```
Mínimo viable: 30-50 muestras por clase
Bueno: 100-500 muestras por clase
Excelente: 1000+ muestras por clase

Más datos = Mejor modelo (en general)
```

---

## 📚 Ejemplos de Datasets

### Ejemplo 1: CSV para Clasificación
**Archivo: iris.csv**
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
```

**En BuildBrain:**
1. Click "Load CSV Dataset"
2. Selecciona iris.csv
3. BuildBrain automáticamente:
   - Detecta 4 features
   - Detecta 3 clases en la columna "species"
   - Divide en train/test

### Ejemplo 2: Imágenes Organizadas
**Estructura:**
```
flores/
├── rosa/
│   ├── rosa1.jpg
│   ├── rosa2.jpg
│   └── rosa3.jpg
├── tulipan/
│   ├── tulipan1.jpg
│   └── tulipan2.jpg
└── girasol/
    ├── girasol1.jpg
    └── girasol2.jpg
```

**En BuildBrain:**
1. En "Classes Setup": Define las clases (rosa, tulipan, girasol)
2. En "Dataset Upload":
   - Haz click "Select Folder" para cada clase
   - O sube las imágenes individuales

---

## 🚀 Próximos Pasos

Después de cargar el dataset:
1. ✅ Datos cargados automáticamente
2. ✅ Modelo inicializado
3. 🎯 Click "Crear Modelo" para entrenar
4. 📊 Ver métricas en tiempo real
5. 💾 Modelo guardado automáticamente

---

## 📞 Soporte

Si tienes problemas cargando tu dataset:
1. Verifica este documento
2. Revisa los logs en `logs/` folder
3. Intenta con un dataset de ejemplo primero
4. Abre un issue con detalles del error

---

¡Listo! Ahora puedes cargar tus datos en BuildBrain. 🎉
