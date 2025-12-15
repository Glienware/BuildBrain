# 🎯 BuildBrain - Flujos Pre-Configurados 

## 📋 Indice de Flujos

1. ✅ **Consulta Usuarios con Deuda** - Flujo completo de consulta a DB
2. ✅ **Búsqueda de Documentos** - Búsqueda vectorial semántica
3. ✅ **Análisis de Sentimiento** - Análisis de emociones en textos

---

## 1️⃣ CONSULTA USUARIOS CON DEUDA

**Archivo:** `Consulta_Usuarios_Deuda.buildbm`

**Propósito:** Consultar usuarios con deuda mayor a un monto específico.

### Diagrama de Flujo

```
┌─────────────────────┐
│   Manual Trigger    │  ← Usuario pregunta
│  (node_0)           │     "Usuarios que deben más de 300"
└──────────┬──────────┘
           │ output
           ▼
┌─────────────────────────────────┐
│  Interpretar Intención (LLM)    │  ← IA entiende qué SQL ejecutar
│  (node_1)                        │     Output: {accion, tabla, 
│                                  │              filtro, operador, valor}
└──────────┬──────────────────────┘
           │ response
           ▼
┌─────────────────────┐
│  Query MySQL        │  ← Ejecuta: SELECT * FROM usuarios 
│  (node_2)           │              WHERE deuda > 300
│                     │     Output: Array de filas
└──────────┬──────────┘
           │ rows
           ▼
┌──────────────────────────────────────┐
│  Formatear Respuesta (LLM)           │  ← IA explica en español
│  (node_3)                             │     "Se encontraron 2 usuarios..."
└──────────┬───────────────────────────┘
           │ response
           ▼
┌─────────────────────┐
│  Dashboard Output   │  ← Muestra tabla con resultados
│  (node_4)           │
└─────────────────────┘
```

### Nodos

| # | ID | Tipo | Nombre | Descripción |
|---|----|----|--------|-------------|
| 1 | node_0 | `manual_trigger` | Manual Trigger | Entrada de consulta del usuario |
| 2 | node_1 | `llm` | Interpretar Intención | LLM que genera instrucciones SQL |
| 3 | node_2 | `database` | Query MySQL | Ejecuta SELECT en DB |
| 4 | node_3 | `llm` | Formatear Respuesta | LLM que resume resultados |
| 5 | node_4 | `dashboard_output` | Dashboard Output | Muestra tabla de resultados |

### Conexiones

```
node_0 (output)      → node_1 (input)
node_1 (response)    → node_2 (input)
node_2 (rows)        → node_3 (input)
node_3 (response)    → node_4 (input)
```

### Variables Principales

```json
{
  "consulta_usuario": "usuarios que deben más de 300",
  "filtro": "deuda",
  "valor": 300
}
```

### Configuración de Cada Nodo

#### Node 0: Manual Trigger
```
Execution Name: "Consulta de deuda usuarios"
Input Variables: ["consulta_usuario"]
Mode: Test
Raw JSON: {
  "consulta": "usuarios que deben más de 300"
}
```

#### Node 1: LLM (Interpretar)
```
Model: Claude
Prompt: "Eres un agente que interpreta solicitudes...
         Devuelve JSON con: accion, tabla, filtro, 
         operador, valor"
Temperature: 0.5
```

#### Node 2: Database
```
DB Type: MySQL
Host: localhost
Database: automation_test
Operation: SELECT
Table: usuarios
Columns: id,nombre,email,telefono,deuda,estado
Where: deuda > {{ node_1.valor }}
Order By: deuda DESC
Limit: 50
```

#### Node 3: LLM (Formatear)
```
Model: Claude
Prompt: "Eres un asistente que explica resultados
         de base de datos en lenguaje claro..."
```

#### Node 4: Dashboard Output
```
Title: "Usuarios con Deuda > 300"
Display Mode: Table
Data Source: {{ node_2.rows }}
Widget ID: deuda_usuarios_table
```

---

## 2️⃣ BÚSQUEDA DE DOCUMENTOS

**Archivo:** `Busqueda_Documentos.buildbm`

**Propósito:** Búsqueda semántica de documentos usando embeddings vectoriales.

### Diagrama de Flujo

```
┌─────────────────────┐
│   Manual Trigger    │  ← Usuario pregunta
│  (node_0)           │     "política de privacidad"
└──────────┬──────────┘
           │ output
           ▼
┌─────────────────────┐
│  Vector Search      │  ← Busca en Pinecone
│  (node_1)           │     Output: Top 5 resultados
└──────────┬──────────┘
           │ results
           ▼
┌──────────────────────────────┐
│  Procesar Resultados (LLM)   │  ← Resume los documentos
│  (node_2)                     │
└──────────┬───────────────────┘
           │ response
           ▼
┌─────────────────────┐
│  Dashboard Output   │  ← Muestra JSON con resultados
│  (node_3)           │
└─────────────────────┘
```

### Nodos

| # | ID | Tipo | Nombre | Descripción |
|---|----|----|--------|-------------|
| 1 | node_0 | `manual_trigger` | Búsqueda Input | Entrada de búsqueda |
| 2 | node_1 | `vector_store_extended` | Vector Search | Búsqueda en Pinecone |
| 3 | node_2 | `llm` | Procesar Resultados | LLM resume documentos |
| 4 | node_3 | `dashboard_output` | Dashboard Output | Muestra JSON |

### Conexiones

```
node_0 (output) → node_1 (query)
node_1 (results) → node_2 (input)
node_2 (response) → node_3 (input)
```

---

## 3️⃣ ANÁLISIS DE SENTIMIENTO

**Archivo:** `Analisis_Sentimiento.buildbm`

**Propósito:** Analizar sentimiento de textos y clasificar como positivo/negativo.

### Diagrama de Flujo

```
┌─────────────────────┐
│   Manual Trigger    │  ← Usuario ingresa texto
│  (node_0)           │     "Me encanta el servicio"
└──────────┬──────────┘
           │ output
           ▼
┌───────────────────────────────────┐
│  Analizador Sentimiento (LLM)     │  ← Analiza y clasifica
│  (node_1)                          │     Output: {sentimiento, 
│                                    │              confianza, tono}
└──────────┬────────────────────────┘
           │ response
           ▼
┌──────────────────────────────────┐
│  Clasificación (If/Else)         │  ← Ramifica positivo/negativo
│  (node_2)                         │
└──────────┬─────────────────────┬─────────────┘
           │ true               │ false
           ▼                    ▼
    ┌─────────────────┐  ┌──────────────────┐
    │  Log Output     │  │  Log Output      │
    │  (node_3)       │  │  (alternativo)   │
    │                 │  │                  │
    │ "Positivo"      │  │ "Negativo/Neutr" │
    └─────────────────┘  └──────────────────┘
```

### Nodos

| # | ID | Tipo | Nombre | Descripción |
|---|----|----|--------|-------------|
| 1 | node_0 | `manual_trigger` | Texto Input | Entrada de texto |
| 2 | node_1 | `llm` | Analizador Sentimiento | LLM clasifica sentimiento |
| 3 | node_2 | `if_else` | Clasificación | Ramificación según sentimiento |
| 4 | node_3 | `log_output` | Guardar Resultado | Log del resultado |

### Conexiones

```
node_0 (output) → node_1 (input)
node_1 (response) → node_2 (input)
node_2 (true) → node_3 (input)
```

---

## 📊 Comparativa de Flujos

| Aspecto | Consulta Deuda | Búsqueda Docs | Sentimiento |
|---------|---|---|---|
| **Nodos** | 5 | 4 | 4 |
| **Complejidad** | ⭐⭐⭐ Alta | ⭐⭐ Media | ⭐⭐ Media |
| **Input** | Manual | Manual | Manual |
| **Output** | Table | JSON | Log |
| **AI Calls** | 2 LLM | 1 LLM | 1 LLM |
| **DB Query** | ✅ MySQL | ❌ No | ❌ No |
| **Vector DB** | ❌ No | ✅ Pinecone | ❌ No |
| **Tiempo Est.** | 5-10s | 3-5s | 2-4s |

---

## 🚀 Cómo Usar Desde la UI

### Cargar un Flujo

1. Click en botón **"Load Agent"** (icono 📤)
2. Selecciona el archivo `.buildbm`:
   - `Consulta_Usuarios_Deuda.buildbm`
   - `Busqueda_Documentos.buildbm`
   - `Analisis_Sentimiento.buildbm`
3. Los 5 nodos se cargan automáticamente ✓
4. El canvas se actualiza con las conexiones ✓

### Modificar un Flujo

1. Click en un nodo para seleccionarlo
2. El **Inspector (panel derecho)** muestra la configuración
3. Modifica los parámetros que necesites:
   - Cambiar prompts de LLM
   - Cambiar tabla o columnas de DB
   - Cambiar modo display de Dashboard
4. Click **"Execute"** para probar

### Guardar tu Flujo Modificado

1. Click en **"Save Agent"** (icono 💾)
2. Dale un nombre único
3. Elige formato:
   - `.buildbm` (recomendado)
   - `.json` (estándar)
4. Se guarda en `projects/`
5. Luego puedes cargarlo con **"Load Agent"**

---

## 🔌 Tipos de Puerto (Port Bindings)

### Node 0: Manual Trigger
- **Output Ports:**
  - `output` (ANY) → Data enviada al siguiente nodo

### Node 1: LLM
- **Input Ports:**
  - `input` (ANY) → Texto/datos para analizar
  - `context` (ANY) → Contexto adicional
- **Output Ports:**
  - `response` (STRING) → Respuesta del LLM
  - `parsed` (JSON) → JSON parseado
  - `metadata` (JSON) → Metadata

### Node 2: Database
- **Input Ports:**
  - `input` (ANY) → Parámetros de query
- **Output Ports:**
  - `rows` (ARRAY) → Array de resultados
  - `count` (NUMBER) → Cantidad de filas
  - `error` (STRING) → Error si hay

### Node 3: If/Else
- **Input Ports:**
  - `input` (ANY) → Dato a evaluar
- **Output Ports:**
  - `true` (ANY) → Si condición es verdadera
  - `false` (ANY) → Si condición es falsa

### Node 4: Dashboard Output
- **Input Ports:**
  - `input` (ANY) → Data a mostrar
- **Output Ports:**
  - `rendered` (BOOLEAN) → Confirmación de renderizado

---

## 📝 Formato .BUILDBM

Los archivos `.buildbm` son JSON con esta estructura:

```json
{
  "version": "1.0",
  "name": "Nombre del Flujo",
  "description": "Descripción breve",
  "nodes": {
    "node_0": { ... },
    "node_1": { ... }
  },
  "connections": [
    {
      "source_node": "node_0",
      "source_port": "output",
      "target_node": "node_1",
      "target_port": "input"
    }
  ],
  "variables": { ... },
  "metadata": { 
    "created_at": "ISO datetime",
    "updated_at": "ISO datetime"
  }
}
```

---

## 🎓 Casos de Uso Reales

### Caso 1: Sistema de Atención al Cliente
```
Usuario pregunta → LLM interpreta → Query usuarios → 
  LLM formula respuesta → Chat output
```

### Caso 2: Análisis de Feedback
```
Feedback ingresado → LLM analiza sentimiento → 
  If/Else clasifica → Guarda a DB
```

### Caso 3: Búsqueda Inteligente
```
Usuario busca → Vector DB busca similar → 
  LLM resume documentos → Muestra resultados
```

---

## 📦 Archivos Incluidos

```
projects/
├── Consulta_Usuarios_Deuda.buildbm    (4.3 KB)
├── Busqueda_Documentos.buildbm        (2.7 KB)
├── Analisis_Sentimiento.buildbm       (2.7 KB)
├── README.md                           (Documentación)
└── flows/                              (Flujos guardados)
```

---

**Versión:** 1.0  
**Fecha:** Diciembre 2025  
**BuildBrain © 2025**
