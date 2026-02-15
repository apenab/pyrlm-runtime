# Plan de Integracion: Pydantic Monty en PyRLM-Runtime

## Resumen Ejecutivo

[Pydantic Monty](https://github.com/pydantic/monty) es un interprete de Python minimo escrito en Rust, disenado especificamente para ejecutar codigo generado por LLMs de forma segura. Reemplazar el `PythonREPL` actual (basado en `exec()`/`eval()` de CPython) por Monty representaria una mejora **critica** en seguridad, rendimiento y confiabilidad para PyRLM-Runtime.

---

## Analisis Comparativo: Estado Actual vs Monty

| Aspecto | PythonREPL Actual (`env.py`) | Pydantic Monty |
|---------|------------------------------|----------------|
| **Aislamiento** | Sandbox por whitelist de builtins/modulos | Interprete Rust separado, sin acceso a CPython |
| **Filesystem** | Bloqueado via import guard (bypasseable) | Bloqueado a nivel de runtime Rust |
| **Red** | No controlado explicitamente | Aislamiento total, sin capacidad de red |
| **Variables de entorno** | Accesibles si se escapa el sandbox | Completamente bloqueadas |
| **Limites de recursos** | Solo truncado de stdout (4000 chars) | Memoria, tiempo de ejecucion, allocations, stack depth |
| **Startup** | Instantaneo (mismo proceso) | Sub-microsegundo (<1us) |
| **Ejecucion infinita** | No hay timeout (puede colgar) | Limites de tiempo configurables |
| **Memory bombs** | Sin proteccion (`[0]*10**9` crash) | Limites de memoria enforced |
| **Serializacion** | No soportada | Codigo y estado de ejecucion serializables |
| **Type checking** | No | Integrado via `ty` |
| **Funciones externas** | Inyectadas en globals directamente | API explicita con pause/resume |

### Vulnerabilidades Actuales que Monty Resuelve

1. **Escape del sandbox**: El REPL actual usa `exec()` de CPython, que es fundamentalmente inseguro. Un LLM podria generar codigo que:
   - Acceda a `__builtins__.__import__('os')` via introspection
   - Use `eval()`/`exec()` anidados para escapar restricciones
   - Acceda a `__class__.__bases__` para navegar el MRO y obtener acceso a modulos bloqueados

2. **Sin timeout de ejecucion**: Un `while True: pass` congela el proceso completo sin recuperacion.

3. **Sin limites de memoria**: `x = "A" * (10**10)` puede crashear el host.

4. **Funciones globales expuestas**: `globals()` y `locals()` estan en los builtins permitidos, lo que permite inspeccion del entorno.

---

## Plan de Implementacion

### Fase 1: Preparacion y Compatibilidad

#### 1.1 Agregar dependencia `pydantic-monty`
- **Archivo**: `pyproject.toml`
- **Accion**: Agregar `pydantic-monty` a las dependencias del proyecto
- **Consideracion**: Verificar compatibilidad con Python 3.12+ (ya es el minimo del proyecto)

#### 1.2 Crear adapter para Monty
- **Archivo nuevo**: `src/pyrlm_runtime/env_monty.py`
- **Accion**: Implementar una nueva clase `MontyREPL` que exponga la misma interfaz que `PythonREPL`:
  ```python
  class MontyREPL:
      def __init__(self, *, stdout_limit=4000, resource_limits=None):
          ...

      def exec(self, code: str) -> ExecResult:
          # Usar Monty.run() o Monty.start()/resume()
          ...

      def get(self, name: str) -> Any:
          ...

      def set(self, name: str, value: Any) -> Any:
          ...
  ```
- **Patron**: Misma interfaz `ExecResult(stdout, error)` para compatibilidad total

#### 1.3 Definir protocolo formal para el REPL
- **Archivo**: `src/pyrlm_runtime/env.py`
- **Accion**: Extraer un `Protocol` (interfaz) comun:
  ```python
  class REPLAdapter(Protocol):
      def exec(self, code: str) -> ExecResult: ...
      def get(self, name: str) -> Any: ...
      def set(self, name: str, value: Any) -> None: ...
  ```
- **Beneficio**: Permite intercambiar PythonREPL y MontyREPL sin cambiar el resto del codigo

---

### Fase 2: Integracion de Funciones Externas

#### 2.1 Mapear funciones del REPL a external functions de Monty
Las funciones que `rlm.py` inyecta en el REPL (lineas ~544-600) deben registrarse como **external functions** de Monty:

| Funcion actual | Tipo en Monty |
|----------------|---------------|
| `peek(n)` | External function |
| `tail(n)` | External function |
| `lenP()` | External function |
| `llm_query(text)` | External function (con pause/resume) |
| `llm_query_batch(chunks)` | External function (con pause/resume) |
| `ask(question, text)` | External function (con pause/resume) |
| `ask_chunks(q, chunks)` | External function (con pause/resume) |
| `ask_chunks_first(q, chunks)` | External function (con pause/resume) |
| `extract_after(marker)` | External function |
| `pick_first_answer(answers)` | External function |
| `ctx.slice(start, end)` | External function |
| `ctx.find(pattern)` | External function |
| `ctx.chunk(size)` | External function |

- **Archivo**: `src/pyrlm_runtime/env_monty.py`
- **Patron**: Usar el mecanismo `start()`/`resume()` de Monty para funciones que necesitan I/O externo (subcalls LLM)
- **Clave**: Las funciones de subcall (`llm_query`, `ask`, etc.) pausan la ejecucion, permiten que el host haga la llamada LLM, y reanudan con el resultado

#### 2.2 Implementar context object como external functions
- El objeto `ctx` actualmente se inyecta como variable global
- En Monty, sus metodos deben exponerse como funciones externas tipadas:
  ```
  ctx_slice(start: int, end: int) -> str
  ctx_find(pattern: str, regex: bool, case_sensitive: bool) -> list
  ctx_chunk(size: int, overlap: int) -> list
  ctx_chunk_documents(docs_per_chunk: int) -> list
  ctx_num_documents() -> int
  ctx_get_document(index: int) -> str
  ctx_document_lengths() -> list
  ```

---

### Fase 3: Configuracion de Limites de Recursos

#### 3.1 Integrar resource limits de Monty con Policy
- **Archivo**: `src/pyrlm_runtime/policy.py`
- **Accion**: Extender `Policy` para incluir limites de ejecucion del REPL:
  ```python
  @dataclass
  class Policy:
      # Existentes
      max_steps: int = 40
      max_subcalls: int = 200
      max_total_tokens: int = 200_000

      # Nuevos (Monty)
      repl_max_execution_time_ms: int = 5000    # 5 seg por ejecucion
      repl_max_memory_mb: int = 128             # 128 MB por ejecucion
      repl_max_stack_depth: int = 100           # Profundidad de stack
      repl_max_allocations: int = 1_000_000     # Numero de allocations
  ```

#### 3.2 Crear configuracion de MontyREPL desde Policy
- **Archivo**: `src/pyrlm_runtime/env_monty.py`
- **Accion**: Traducir los limites de `Policy` a la configuracion de `LimitTracker` de Monty
- **Beneficio**: Un solo lugar para configurar todos los limites (Policy)

---

### Fase 4: Serializacion y Cache

#### 4.1 Cache de codigo compilado
- **Archivo**: `src/pyrlm_runtime/cache.py`
- **Accion**: Extender el cache para almacenar codigo Monty pre-compilado:
  ```python
  class MontyCodeCache:
      def get_compiled(self, code: str) -> bytes | None:
          # Retorna Monty.dump() cacheado
          ...

      def set_compiled(self, code: str, compiled: bytes) -> None:
          ...
  ```
- **Beneficio**: Evitar re-parseo de codigo identico en iteraciones del RLM loop

#### 4.2 Serializacion de estado de ejecucion
- **Archivo**: `src/pyrlm_runtime/env_monty.py`
- **Accion**: Exponer `MontySnapshot.dump()`/`load()` para:
  - Persistir el estado del REPL entre pasos
  - Permitir recuperacion ante crashes
  - Habilitar ejecucion distribuida (futuro)

---

### Fase 5: Integracion en el Core RLM

#### 5.1 Modificar RLM para soportar ambos REPLs
- **Archivo**: `src/pyrlm_runtime/rlm.py`
- **Accion**: Agregar parametro `repl_backend` a la clase `RLM`:
  ```python
  @dataclass
  class RLM:
      adapter: ModelAdapter
      repl_backend: Literal["python", "monty"] = "monty"  # Default: monty
      ...
  ```
- **Logica**: Instanciar `PythonREPL` o `MontyREPL` segun configuracion
- **Beneficio**: Migracion gradual, fallback a PythonREPL si Monty no esta disponible

#### 5.2 Actualizar el loop principal
- **Archivo**: `src/pyrlm_runtime/rlm.py`
- **Accion**: Adaptar la ejecucion del REPL en el loop para manejar el patron pause/resume de Monty:
  ```
  Flujo actual:  code -> exec() -> resultado
  Flujo Monty:   code -> start() -> [pause en external_fn] -> ejecutar fn -> resume() -> ... -> complete
  ```
- **Clave**: Las funciones de subcall ahora usan el mecanismo nativo de pause/resume

#### 5.3 Manejar errores y limites de Monty
- **Archivo**: `src/pyrlm_runtime/policy.py`
- **Accion**: Agregar excepciones nuevas:
  ```python
  class REPLExecutionTimeout(PolicyError): ...
  class REPLMemoryExceeded(PolicyError): ...
  class REPLStackOverflow(PolicyError): ...
  ```
- **Integracion**: Capturar excepciones de Monty y traducirlas a PolicyErrors

---

### Fase 6: Type Checking (Opcional pero Recomendado)

#### 6.1 Agregar validacion de tipos pre-ejecucion
- **Archivo**: `src/pyrlm_runtime/env_monty.py`
- **Accion**: Usar `type_check=True` de Monty para validar codigo del LLM antes de ejecutarlo
- **Beneficio**: Detectar errores de tipo antes de ejecutar, ahorrando pasos del loop
- **Configuracion**: Proveer stubs de tipos para todas las funciones externas via `type_check_stubs`

#### 6.2 Crear stubs de tipos para funciones del REPL
- **Archivo nuevo**: `src/pyrlm_runtime/monty_stubs.py`
- **Contenido**: Definiciones de tipos para todas las funciones externas:
  ```python
  REPL_STUBS = """
  def peek(n: int = 2000) -> str: ...
  def tail(n: int = 2000) -> str: ...
  def lenP() -> int: ...
  def llm_query(text: str) -> str: ...
  def ask(question: str, text: str) -> str: ...
  ...
  """
  ```

---

### Fase 7: Testing

#### 7.1 Tests unitarios para MontyREPL
- **Archivo nuevo**: `tests/test_env_monty.py`
- **Cobertura**:
  - Ejecucion basica de codigo
  - Funciones externas (peek, tail, llm_query, etc.)
  - Limites de recursos (timeout, memoria, stack)
  - Serializacion/deserializacion
  - Type checking pre-ejecucion
  - Comparacion de resultados con PythonREPL (paridad)

#### 7.2 Tests de integracion
- **Archivo nuevo**: `tests/test_rlm_monty.py`
- **Cobertura**:
  - Loop completo del RLM con MontyREPL
  - Subcalls via pause/resume
  - Subcalls paralelos
  - Subcalls recursivos
  - Fallback de Monty a PythonREPL
  - Router con MontyREPL

#### 7.3 Tests de seguridad
- **Archivo nuevo**: `tests/test_security_monty.py`
- **Cobertura**:
  - Intento de acceso a filesystem
  - Intento de acceso a red
  - Intento de acceso a variables de entorno
  - Intento de import de modulos no permitidos
  - Code injection via strings
  - Memory bombs (`[0]*10**9`)
  - Infinite loops (verificar timeout)
  - Stack overflow (recursion infinita)
  - Introspection attacks (`__class__.__bases__`)

#### 7.4 Tests de rendimiento
- **Archivo nuevo**: `tests/test_perf_monty.py`
- **Cobertura**:
  - Benchmark: PythonREPL vs MontyREPL (startup, ejecucion)
  - Benchmark: Cache de codigo compilado
  - Benchmark: Overhead de external functions vs globals

---

### Fase 8: Retrocompatibilidad y Migracion

#### 8.1 Feature flag para seleccionar backend
- **Archivo**: `src/pyrlm_runtime/rlm.py`
- **Estrategia**:
  1. `repl_backend="monty"` (default para nuevas instalaciones)
  2. `repl_backend="python"` (fallback, deprecated warning)
  3. Auto-deteccion: si `pydantic-monty` no esta instalado, usar PythonREPL con warning

#### 8.2 Mantener PythonREPL como fallback
- **Archivo**: `src/pyrlm_runtime/env.py`
- **Accion**: No eliminar PythonREPL, marcarlo como legacy
- **Razon**: Usuarios que no pueden instalar binarios Rust (algunas plataformas)

#### 8.3 Actualizar prompts del sistema
- **Archivo**: `src/pyrlm_runtime/prompts.py`
- **Accion**: Revisar si los prompts del sistema necesitan ajustes para las funciones renombradas (e.g., `ctx.slice()` -> `ctx_slice()` si aplica)
- **Nota**: Idealmente mantener la misma API visible al LLM para no requerir cambios en prompts

---

### Fase 9: Actualizacion de Documentacion

#### 9.1 Actualizar README principal
- **Archivo**: `README.md`
- **Contenido a agregar**:
  - Seccion "Seguridad" explicando el sandbox Monty
  - Configuracion de limites de recursos
  - Guia de migracion de PythonREPL a MontyREPL
  - Nota sobre dependencia opcional de `pydantic-monty`

#### 9.2 Actualizar documentacion tecnica
- **Archivo**: `docs/README.md`
- **Contenido a agregar**:
  - Arquitectura del sandbox Monty
  - Diagrama de flujo pause/resume para external functions
  - Comparacion de seguridad (antes/despues)
  - Referencia de API de MontyREPL

#### 9.3 Crear guia de configuracion de seguridad
- **Archivo nuevo**: `docs/security.md`
- **Contenido**:
  - Modelo de amenazas y como Monty los mitiga
  - Configuracion recomendada de limites de recursos
  - Best practices para funciones externas
  - Guia de hardening para produccion

#### 9.4 Actualizar docstrings y type hints
- **Archivos**: Todos los archivos modificados
- **Accion**: Asegurar que todas las clases y funciones nuevas tengan docstrings descriptivos y type hints completos

#### 9.5 Actualizar CHANGELOG
- **Archivo**: `CHANGELOG.md` (crear si no existe)
- **Contenido**: Documentar el cambio como mejora mayor de seguridad

---

## Orden de Ejecucion Recomendado

```
Fase 1 (Preparacion)          ━━━━━━━━  ~1 dia
  1.1 Dependencia
  1.2 MontyREPL clase
  1.3 Protocolo REPLAdapter
         │
         ▼
Fase 2 (External Functions)   ━━━━━━━━  ~2 dias
  2.1 Mapear funciones
  2.2 Context como external fns
         │
         ▼
Fase 3 (Resource Limits)      ━━━━━━━━  ~1 dia
  3.1 Extender Policy
  3.2 Config MontyREPL
         │
         ▼
Fase 5 (Core Integration)     ━━━━━━━━  ~2 dias
  5.1 Parametro repl_backend
  5.2 Loop pause/resume
  5.3 Errores y limites
         │
         ▼
Fase 7 (Testing)              ━━━━━━━━  ~2 dias
  7.1 Unit tests
  7.2 Integration tests
  7.3 Security tests
         │
         ▼
Fase 4 (Serializacion)        ━━━━━━━━  ~1 dia
  4.1 Cache compilado
  4.2 Estado de ejecucion
         │
         ▼
Fase 6 (Type Checking)        ━━━━━━━━  ~1 dia
  6.1 Validacion pre-ejecucion
  6.2 Stubs de tipos
         │
         ▼
Fase 8 (Retrocompatibilidad)  ━━━━━━━━  ~1 dia
  8.1 Feature flag
  8.2 Legacy PythonREPL
  8.3 Ajuste de prompts
         │
         ▼
Fase 9 (Documentacion)        ━━━━━━━━  ~1 dia
  9.1 README principal
  9.2 Docs tecnicos
  9.3 Guia de seguridad
  9.4 Docstrings
  9.5 CHANGELOG
```

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigacion |
|--------|-------------|---------|------------|
| Monty no soporta toda la sintaxis Python que los LLMs generan (clases, match) | Alta | Medio | Fallback a PythonREPL, ajustar prompts para evitar sintaxis no soportada |
| Overhead de external functions vs globals directos | Media | Bajo | Benchmark, cache de codigo compilado |
| Monty es proyecto nuevo (posibles bugs) | Media | Alto | Tests exhaustivos, fallback a PythonREPL |
| Incompatibilidad con algunas plataformas (ARM, musl) | Baja | Medio | PythonREPL como fallback, dependencia opcional |
| Cambios en API de Monty (pre-1.0) | Alta | Medio | Pinear version, wrapper abstracto |

---

## Metricas de Exito

1. **Seguridad**: 0 escapes de sandbox en tests de seguridad (vs vulnerabilidades conocidas en PythonREPL)
2. **Rendimiento**: Startup del REPL < 1ms, ejecucion de codigo simple < 10ms
3. **Compatibilidad**: 100% de tests existentes pasan con MontyREPL
4. **Limites**: Timeout, memoria y stack enforced correctamente en todos los tests
5. **Cobertura**: > 90% code coverage en `env_monty.py`
