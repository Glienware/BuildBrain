# BuildBrain - Flujos Pre-Configurados 🚀

Este proyecto incluye varios flujos `.buildbm` listos para usar. Descarga, carga y ejecuta en segundos.

## 📦 Flujos Disponibles

### 1. **Consulta Usuarios con Deuda** 
**Archivo:** `Consulta_Usuarios_Deuda.buildbm`

**Caso de Uso:** Consultar la base de datos MySQL para usuarios con deuda.

**Flujo:**
```
Manual Trigger (Usuario pregunta)
    ↓
AI Agent (Interpreta la intención)
    ↓
Database Query (Ejecuta SELECT en MySQL)
    ↓
AI Agent (Formatea la respuesta)
    ↓
Dashboard Output (Muestra tabla)
```

**Nodos:**
- **node_0:** Manual Trigger - Recibe consulta del usuario
- **node_1:** LLM (Interpretar Intención) - Entiende qué quiere el usuario
- **node_2:** Database Query - SELECT * FROM usuarios WHERE deuda > valor
- **node_3:** LLM (Formatear Respuesta) - Resume en lenguaje humano
- **node_4:** Dashboard Output - Muestra resultado en tabla

**Cómo usar:**
1. Clic en "Load Agent"
2. Selecciona `Consulta_Usuarios_Deuda.buildbm`
3. Modifica los parámetros si quieres (tabla, condiciones, etc.)
4. Clic en "Execute"
5. Resultado aparece en la tabla

---

### 2. **Búsqueda de Documentos**
**Archivo:** `Busqueda_Documentos.buildbm`

**Caso de Uso:** Buscar documentos usando vector embeddings.

**Flujo:**
```
Manual Trigger (Entrada de búsqueda)
    ↓
Vector Database (Búsqueda semántica)
    ↓
AI Agent (Procesa resultados)
    ↓
Dashboard Output (Muestra JSON)
```

**Nodos:**
- **node_0:** Manual Trigger - Recibe query de búsqueda
- **node_1:** Vector Store (Pinecone) - Búsqueda semántica
- **node_2:** LLM - Procesa y resume resultados
- **node_3:** Dashboard - Muestra en JSON

**Configuración necesaria:**
- Pinecone API key (editar en node_1)
- Embeddings model (configurar en node_1)

---

### 3. **Análisis de Sentimiento**
**Archivo:** `Analisis_Sentimiento.buildbm`

**Caso de Uso:** Analizar sentimiento de textos (comentarios, reviews, etc.)

**Flujo:**
```
Manual Trigger (Texto usuario)
    ↓
AI Agent (Analiza sentimiento)
    ↓
If/Else (Clasifica positivo/negativo)
    ↓
Log Output (Guarda resultado)
```

**Nodos:**
- **node_0:** Manual Trigger - Entrada de texto
- **node_1:** LLM - Análisis de sentimiento (retorna JSON con sentimiento + confianza)
- **node_2:** If/Else - Ramifica según sentimiento
- **node_3:** Log Output - Guarda el resultado

---

## 🔧 Cómo Cargar un Flujo Pre-Configurado

1. **Abre la aplicación BuildBrain**
2. **Clic en el botón "Load Agent"** (icono de upload)
3. **Selecciona el archivo .buildbm** que quieres cargar
4. **Los nodos aparecerán automáticamente** con sus configuraciones
5. **Modifica parámetros si es necesario** (en el inspector derecho)
6. **Ejecuta con el botón "Execute"**

---

## 💾 Cómo Guardar tu Propio Flujo

1. **Crea tu flujo** agregando nodos y conectándolos
2. **Clic en "Save Agent"**
3. **Elige un nombre** (ej: "Mi Flujo de Análisis")
4. **Selecciona formato:**
   - `.buildbm` - Formato personalizado (recomendado)
   - `.json` - JSON estándar
5. **Se guarda automáticamente en `projects/`**
6. **Luego puedes cargar con "Load Agent"**

---

## 📝 Estructura del Archivo .BUILDBM

```json
{
  "version": "1.0",
  "name": "Nombre del Flujo",
  "description": "Descripción",
  "nodes": {
    "node_0": {
      "type": "manual_trigger",
      "display_name": "Manual Trigger",
      "category": "INPUT",
      "position": { "x": 50, "y": 200 },
      "settings": { ... }
    },
    "node_1": { ... },
    ...
  },
  "connections": [
    {
      "source_node": "node_0",
      "source_port": "output",
      "target_node": "node_1",
      "target_port": "input"
    },
    ...
  ],
  "variables": { ... },
  "metadata": { ... }
}
```

---

## 🔌 Tipos de Nodos Disponibles

### **INPUT**
- `manual_trigger` - Entrada manual del usuario

### **AI**
- `llm` - Modelo de lenguaje
- `open_router` - OpenRouter API
- `agent` - Agente con tools
- `vector_store_extended` - Búsqueda vectorial

### **DATABASE**
- `database` - Query SQL (MySQL, PostgreSQL, SQLite)

### **LOGIC**
- `if_else` - Ramificación condicional

### **OUTPUT**
- `dashboard_output` - Dashboard visual
- `log_output` - Log/Console
- `api_response` - Webhook response

### **TRANSFORMATION**
- `data_transformer` - Transformar datos
- `json_parser` - Parsear JSON
- `http_request` - HTTP calls

---

## 🎯 Plantillas Rápidas

### Plantilla: Database → Dashboard
```json
{
  "nodes": {
    "node_0": { "type": "manual_trigger", ... },
    "node_1": { "type": "database", ... },
    "node_2": { "type": "dashboard_output", ... }
  },
  "connections": [
    { "source_node": "node_0", "target_node": "node_1" },
    { "source_node": "node_1", "target_node": "node_2" }
  ]
}
```

### Plantilla: LLM → Conditional → Outputs
```json
{
  "nodes": {
    "node_0": { "type": "manual_trigger", ... },
    "node_1": { "type": "llm", ... },
    "node_2": { "type": "if_else", ... },
    "node_3": { "type": "dashboard_output", ... }
  }
}
```

---

## ⚙️ Configuración Requerida

Para que los flujos funcionen correctamente:

### **MySQL/Database**
```
Host: localhost (o tu servidor)
Port: 3306
Database: automation_test
Username: root
Password: (sin contraseña o la tuya)
```

### **OpenAI/Claude**
- Agrega tu API key en los nodos LLM
- O usa OpenRouter con tu API key

### **Pinecone (para búsqueda vectorial)**
- Crea índice en Pinecone
- Agrega API key en node_1

---

## 🚀 Ejemplo Completo: Ejecución Manual

**Paso 1:** Cargar `Consulta_Usuarios_Deuda.buildbm`
```
Archivo cargado con 5 nodos conectados ✓
```

**Paso 2:** Modificar parámetros (opcional)
```
node_0 (Manual Trigger)
  - Cambiar: "usuarios que deben más de 300"

node_2 (Database Query)
  - Verificar credenciales MySQL
  - Verificar tabla "usuarios"
```

**Paso 3:** Ejecutar el workflow
```
Clic en "Execute" ✓
```

**Paso 4:** Ver resultados
```
Dashboard muestra tabla con usuarios filtrados ✓
```

---

## 🐛 Troubleshooting

**Error: "Agente no cargado"**
- Verifica que el archivo .buildbm esté en `projects/`
- Comprueba que el JSON sea válido

**Error: "Conexión a BD rechazada"**
- Verifica credenciales MySQL
- Asegúrate que el servidor está corriendo
- Comprueba que la base de datos existe

**Error: "API key inválida"**
- Agrega tu OpenAI/Claude key en los nodos LLM
- Verifica que no haya espacios en blanco

---

## 📚 Documentación Completa

Para más detalles sobre:
- Configuración de nodos
- Variables y binding
- Ejecución avanzada
- APIs personalizadas

Consulta el README principal del proyecto.

---

**Versión:** 1.0  
**Última actualización:** Diciembre 2025  
**BuildBrain © 2025**
