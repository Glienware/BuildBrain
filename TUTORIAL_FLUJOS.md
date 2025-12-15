# 🎬 TUTORIAL: Cómo Usar los Flujos Pre-Configurados

## 📺 PARTE 1: Cargando el Flujo Principal

### Paso 1: Abre la Aplicación
```bash
cd BuildBrain
python main.py
```

Verás la pantalla principal con el canvas vacío.

---

### Paso 2: Click en "Load Agent"

En la barra superior, busca el botón **"Load Agent"** (icono 📤)

```
┌─────────────────────────────────────────┐
│  [≡] │ AUTOFLOW DSK  [0 nodes]          │
│  [Add Node] [Load Agent] [Execute] [Save]│
└─────────────────────────────────────────┘
                    ↑
                 Click aquí
```

---

### Paso 3: Selecciona el Archivo

Se abrirá un diálogo para seleccionar archivos.

**Busca la carpeta `projects/` y selecciona:**

```
✓ Consulta_Usuarios_Deuda.buildbm
```

---

### Paso 4: El Flujo se Carga Automáticamente

Los 5 nodos aparecerán en el canvas:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Manual    │───▶│  Interpretar│───▶│   Query     │───▶│ Formatear   │───▶│  Dashboard  │
│  Trigger    │    │ Intención   │    │  MySQL      │    │ Respuesta   │    │   Output    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
   node_0              node_1             node_2              node_3            node_4
```

✅ **Status:** El flujo está completamente cargado

---

## 📋 PARTE 2: Entendiendo Cada Nodo

### Node 0: Manual Trigger 📥

**Propósito:** Simular la entrada del usuario

Click en el nodo para ver su configuración en el inspector derecho:

```
╔═══════════════════════════════════════╗
║  MANUAL TRIGGER                       ║
╠═══════════════════════════════════════╣
║ Execution Name                        ║
║ [ Consulta de deuda usuarios ]        ║
║                                       ║
║ Input Variables                       ║
║ [+] consulta_usuario                  ║
║                                       ║
║ Raw JSON Input                        ║
║ {                                     ║
║   "consulta": "usuarios que deben     ║
║   más de 300"                         ║
║ }                                     ║
║                                       ║
║ Mode                                  ║
║ (●) Test  ( ) Production              ║
╚═══════════════════════════════════════╝
```

**Qué hace:**
- ✓ Simula al usuario escribiendo: "usuarios que deben más de 300"
- ✓ Pasa esa consulta al siguiente nodo (LLM)
- ✓ Funciona como disparador del flujo

**Modificar:** Cambia el contenido de `Raw JSON Input` si quieres probar otra consulta

---

### Node 1: Interpretar Intención 🤖

**Propósito:** AI entiende qué SQL ejecutar

Click en el nodo para ver:

```
╔════════════════════════════════════════╗
║  LLM / AGENT - INTERPRETAR INTENCIÓN   ║
╠════════════════════════════════════════╣
║ Model                                  ║
║ [Claude ▼]                             ║
║                                        ║
║ System Prompt                          ║
║ "Eres un agente que interpreta         ║
║  solicitudes del usuario..."           ║
║                                        ║
║ Temperature: 0.5                       ║
║ Max Tokens: 500                        ║
╚════════════════════════════════════════╝
```

**Qué hace:**
- Recibe: "usuarios que deben más de 300"
- Entiende: Hay que buscar en tabla "usuarios" donde "deuda > 300"
- Output: JSON con instrucciones estructuradas
- Pasa al siguiente nodo (Database Query)

**Nota:** Este nodo NO ejecuta el SQL, solo lo "entiende"

---

### Node 2: Query MySQL 🗄️

**Propósito:** Ejecutar el SELECT real en la base de datos

Click en el nodo:

```
╔════════════════════════════════════════╗
║  DATABASE QUERY (MySQL)                ║
╠════════════════════════════════════════╣
║ Database Type: MySQL                   ║
║ Host: localhost                        ║
║ Database: automation_test              ║
║ Username: root                         ║
║ Password: (vacío)                      ║
║                                        ║
║ Operation: SELECT                      ║
║ Table: usuarios                        ║
║ Columns: id,nombre,email,...           ║
║                                        ║
║ Where Condition                        ║
║ [ deuda ] [ > ] [ 300 ]                ║
║                                        ║
║ Order By: deuda DESC                   ║
║ Limit: 50                              ║
║                                        ║
║ Query Preview:                         ║
║ SELECT id,nombre,email,... FROM        ║
║ usuarios WHERE deuda > 300 ORDER BY    ║
║ deuda DESC LIMIT 50                    ║
╚════════════════════════════════════════╝
```

**Qué hace:**
- Recibe las instrucciones del LLM anterior
- Ejecuta: `SELECT * FROM usuarios WHERE deuda > 300`
- Output: Array de filas (JSON con datos reales de la BD)
- Pasa al siguiente nodo (LLM que formatea)

**Verifica:** El botón "Test Connection" para asegurar que MySQL está activo

---

### Node 3: Formatear Respuesta 🤖

**Propósito:** AI explica los datos en lenguaje humano

Similar a node_1 pero con un prompt diferente:

```
╔════════════════════════════════════════╗
║  LLM - FORMATEAR RESPUESTA             ║
╠════════════════════════════════════════╣
║ Prompt                                 ║
║ "Eres un asistente que explica         ║
║  resultados de BD en lenguaje claro"   ║
║                                        ║
║ Datos:                                 ║
║ {{ DatabaseQuery.rows }}               ║
║ (Array con usuarios que deben)         ║
║                                        ║
║ Output esperado:                       ║
║ {                                      ║
║   "mensaje": "Se encontraron 2         ║
║    usuarios con deuda > 300"            ║
║   "usuarios": [                        ║
║     "Juan debe $350",                  ║
║     "Pedro debe $480"                  ║
║   ]                                    ║
║ }                                      ║
╚════════════════════════════════════════╝
```

**Qué hace:**
- Recibe los datos brutos del Database
- Los interpreta y explica en español
- Crea un mensaje amigable para el usuario
- Pasa al último nodo (Dashboard)

---

### Node 4: Dashboard Output 📊

**Propósito:** Mostrar los resultados visualmente

```
╔════════════════════════════════════════╗
║  DASHBOARD OUTPUT                      ║
╠════════════════════════════════════════╣
║ Title                                  ║
║ [ Usuarios con Deuda > 300 ]           ║
║                                        ║
║ Display Mode                           ║
║ (●) Table  ( ) JSON                    ║
║                                        ║
║ Data Source                            ║
║ {{ DatabaseQuery.rows }}               ║
║                                        ║
║ Widget ID                              ║
║ [ deuda_usuarios_table ]               ║
║                                        ║
║ Preview                                ║
║ ┌─────────────────────────────────┐   ║
║ │ id│ Nombre    │ Deuda │ Estado  │   ║
║ ├──┼──────────┼───────┼──────────┤   ║
║ │1 │ Juan P.   │ $350  │ Activo   │   ║
║ │5 │ Pedro S.  │ $480  │ Suspendido│  ║
║ └─────────────────────────────────┘   ║
╚════════════════════════════════════════╝
```

**Qué hace:**
- Recibe el mensaje formateado del LLM anterior
- Muestra los datos en formato tabla
- El usuario ve el resultado final ✓

---

## 🎯 PARTE 3: Ejecutar el Flujo Completo

### Paso 1: Verifica la Configuración

Antes de ejecutar, asegúrate que:

- [ ] **MySQL está corriendo** (verificar con "Test Connection" en node_2)
- [ ] **Base de datos existe:** `automation_test`
- [ ] **Tabla existe:** `usuarios`
- [ ] **Tienes usuarios con deuda > 300 en la BD**

### Paso 2: Click en "Execute"

En la barra superior:

```
[Execute] ← Click aquí
```

### Paso 3: Observa la Ejecución

El flujo se ejecutará en este orden:

```
1. node_0 (Manual Trigger) - Env\u00eda "usuarios que deben > 300"
   ✓ 0.1s

2. node_1 (LLM - Interpretar) - Entiende qué consultar
   ✓ 2-3s (llama a OpenAI/Claude)

3. node_2 (Database Query) - Ejecuta SELECT en MySQL
   ✓ 0.5s (consulta la BD)

4. node_3 (LLM - Formatear) - Resume en español
   ✓ 2-3s (llama a OpenAI/Claude)

5. node_4 (Dashboard) - Muestra tabla
   ✓ 0.1s (renderiza UI)

Total: ~5-7 segundos
```

### Paso 4: Ver Resultados

En el panel derecho (Inspector), la tabla aparecerá con los usuarios:

```
┌────────────────────────────────────────────┐
│ Usuarios con Deuda > 300                   │
├──┬──────────┬───────────┬────────┬─────────┤
│ID│ Nombre   │ Email     │ Deuda  │ Estado  │
├──┼──────────┼───────────┼────────┼─────────┤
│1 │ Juan P.  │ juan@...  │ $350   │ Activo  │
│5 │ Pedro S. │ pedro@... │ $480   │ Suspendido│
└──┴──────────┴───────────┴────────┴─────────┘
```

✅ **¡Flujo completo ejecutado exitosamente!**

---

## 🔧 PARTE 4: Modificar y Guardar

### Cambiar la Condición

Digamos que quieres usuarios con deuda > **500** en lugar de 300:

1. Click en **node_2** (Database Query)
2. En el inspector, busca la sección "Where"
3. Cambia el valor de **300** a **500**
4. Click "Execute" nuevamente
5. Los resultados se actualizan

### Cambiar la Tabla

Si quieres consultar otra tabla:

1. Click en **node_2** (Database Query)
2. Cambia `Table: usuarios` a `Table: [otra tabla]`
3. Actualiza las columnas según lo que tenga esa tabla
4. Ejecuta nuevamente

### Cambiar el Modo de Display

Para mostrar JSON en lugar de tabla:

1. Click en **node_4** (Dashboard Output)
2. Cambia `Display Mode` de "Table" a "JSON"
3. Ejecuta nuevamente
4. Verás los datos en formato JSON

### Guardar tu Flujo Modificado

Una vez que hagas cambios:

1. Click en **"Save Agent"** (icono 💾)
2. Dale un nombre: `Mi_Flujo_Deuda_500`
3. Elige formato: `.buildbm` (recomendado)
4. Se guarda en `projects/`
5. Luego puedes cargarlo con "Load Agent" en cualquier momento

---

## 🎓 PARTE 5: Otros Flujos

### Cargar Análisis de Sentimiento

1. Click "Load Agent"
2. Selecciona `Analisis_Sentimiento.buildbm`
3. Se cargan 4 nodos:
   - node_0: Manual Trigger (texto)
   - node_1: LLM (analiza sentimiento)
   - node_2: If/Else (ramificación)
   - node_3: Log Output

### Cargar Búsqueda de Documentos

1. Click "Load Agent"
2. Selecciona `Busqueda_Documentos.buildbm`
3. Se cargan 4 nodos:
   - node_0: Manual Trigger (búsqueda)
   - node_1: Vector Store (Pinecone)
   - node_2: LLM (procesa)
   - node_3: Dashboard

---

## 🐛 Troubleshooting

### Error: "Connection refused"
**Causa:** MySQL no está corriendo  
**Solución:** Inicia MySQL server
```bash
# Windows
net start MySQL80

# macOS
brew services start mysql-server

# Linux
sudo systemctl start mysql
```

### Error: "Database not found"
**Causa:** La BD `automation_test` no existe  
**Solución:** Créala en MySQL
```sql
CREATE DATABASE automation_test;
USE automation_test;
CREATE TABLE usuarios (
  id INT PRIMARY KEY,
  nombre VARCHAR(100),
  email VARCHAR(100),
  telefono VARCHAR(20),
  deuda DECIMAL(10,2),
  estado VARCHAR(50)
);
```

### Error: "API key invalid"
**Causa:** No configuraste la clave de OpenAI  
**Solución:** 
1. Abre el nodo LLM
2. Ve a la sección "API Key" 
3. Ingresa tu clave de OpenAI/Claude
4. Guarda

### El Flujo No Se Ejecuta
**Causa:** Posibles validaciones fallidas  
**Solución:** 
1. Revisa el log en la consola
2. Verifica que todos los nodos estén conectados
3. Comprueba que no haya conflictos de tipos de datos

---

## 📚 Próximos Pasos

Una vez domines los flujos pre-configurados:

1. **Crea tu propio flujo** desde cero
   - Agrega nodos con "Add Node"
   - Conéctalos arrastrando
   - Configura cada uno

2. **Explora otros nodos:**
   - HTTP Request (para APIs)
   - Data Transformer (para procesar datos)
   - Memory (para guardar estado)
   - Schedule (para ejecutar periódicamente)

3. **Integra con sistemas externos:**
   - APIs REST
   - Webhooks
   - Bases de datos diferentes
   - Servicios en la nube

4. **Comparte tus flujos:**
   - Exporta como `.buildbm`
   - Comparte con tu equipo
   - Reutiliza en otros proyectos

---

**¡Listo! Ya sabes cómo usar los flujos pre-configurados. ¡Experimenta y crea! 🚀**

Versión: 1.0  
Fecha: Diciembre 2025  
BuildBrain © 2025
