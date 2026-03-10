# Cambios en la Biblioteca pyrlm-runtime

> Documento preparado para actualizar la presentacion tecnica sobre mejoras en pyrlm-runtime.
> Fecha: 2026-03-01

---

## 1. Resumen de todos los cambios

Se realizaron cambios en dos archivos de la biblioteca (`src/pyrlm_runtime/`):

### Archivo: `src/pyrlm_runtime/rlm.py`
- **Nuevo parametro `subcall_max_tokens`**: Permite configurar el limite de tokens por defecto para subcalls a nivel del dataclass `RLM`, en lugar de tener un valor hardcodeado en cada funcion.
- **Nuevo parametro `auto_finalize_min_length`**: Previene la finalizacion prematura cuando la respuesta almacenada en `auto_finalize_var` es demasiado corta (esqueleto, placeholder).
- **Refactorizacion de `max_tokens` en todas las funciones de subcall**: `subcall()`, `subcall_batch()`, `ask()`, `ask_chunk()`, `ask_chunked()`, `ask_chunks()`, `ask_chunks_first()` ahora aceptan `max_tokens: int | None = None` con fallback al valor de `self.subcall_max_tokens`.
- **Importacion de `MaxStepsExceeded`**: Se agrego la importacion desde `policy.py` para capturar la excepcion de forma explicita.
- **Verificacion de longitud minima en `maybe_auto_finalize()`**: Si la respuesta es mas corta que `auto_finalize_min_length`, se bloquea la finalizacion y el loop continua procesando.
- **Fallback inteligente en `MaxStepsExceeded`**: En lugar de retornar `NO_ANSWER` cuando se agotan los pasos, el sistema ahora (1) revisa si hay un valor valido en `auto_finalize_var`, (2) le pide al modelo un resumen final basado en toda la conversacion, y (3) solo retorna `NO_ANSWER` como ultimo recurso.

### Archivo: `src/pyrlm_runtime/adapters/generic_chat.py`
- **Fix de `default_response_parser` para modelos de razonamiento**: Cambiado el acceso directo a `content` por un acceso seguro con `.get("content") or ""` para manejar el caso donde `content` es `null` (modelos que usan tokens internos de razonamiento).

---

## 2. Detalle de cada cambio (codigo antes/despues)

### 2.1 Importacion de `MaxStepsExceeded`

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Linea**: 13

```diff
- from .policy import Policy, estimate_tokens
+ from .policy import MaxStepsExceeded, Policy, estimate_tokens
```

**Por que**: Anteriormente `policy.check_step()` lanzaba `MaxStepsExceeded` pero no se capturaba explicitamente en el loop principal. El loop simplemente dejaba que la excepcion propagara sin control. Ahora se captura para implementar el fallback inteligente.

---

### 2.2 Nuevo parametro `auto_finalize_min_length`

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 45-47

```diff
  auto_finalize_var: str | None = None
+ # Minimum character length for auto_finalize_var to trigger (prevents premature finalization)
+ auto_finalize_min_length: int = 0
  logger: logging.Logger | None = None
```

**Por que**: Resuelve el "problema de finalizacion prematura". Cuando un modelo escribe un esqueleto o placeholder como respuesta parcial (ej. `"skeleton"`, `"processing..."`), el sistema lo detectaba como respuesta valida y finalizaba. Con `auto_finalize_min_length > 0`, se requiere que la respuesta tenga al menos N caracteres antes de aceptarla. Ver seccion 4 para mas detalle.

---

### 2.3 Nuevo parametro `subcall_max_tokens`

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 68-69

```diff
  max_concurrent_subcalls: int = 10
+ # Default max_tokens for subcall responses (increase for reasoning models)
+ subcall_max_tokens: int = 256
  # REPL backend: "python" (default) or "monty" (pydantic-monty sandbox)
```

**Por que**: Los modelos de razonamiento (GPT-5, DeepSeek-v3.2, Qwen3-next) consumen tokens internos en su cadena de pensamiento (`<think>...</think>`), dejando poco espacio para el contenido real. Con 256 tokens, un modelo que gasta 200 en razonamiento solo tiene 56 para la respuesta. Al exponer `subcall_max_tokens` como parametro configurable del dataclass, el usuario puede ajustarlo a 512 o 1024 sin modificar cada llamada individual. Ver seccion 3 para mas detalle.

---

### 2.4 Refactorizacion de `subcall()` con `max_tokens` opcional

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 124-131

```diff
  def subcall(
      text: str,
      *,
      model: str | None = None,
-     max_tokens: int = 256,
+     max_tokens: int | None = None,
      depth: int = 1,
  ) -> str:
+     if max_tokens is None:
+         max_tokens = self.subcall_max_tokens
      nonlocal subcall_made
```

**Antes**: El valor por defecto de 256 estaba hardcodeado en la firma de la funcion. No habia forma de cambiar el default para todas las subcalls sin modificar cada llamada.

**Despues**: El parametro acepta `None`, y cuando no se especifica, se usa `self.subcall_max_tokens` del dataclass. Esto permite un control centralizado.

**Por que**: Centraliza la configuracion. Un usuario puede hacer `RLM(adapter=..., subcall_max_tokens=1024)` y todas las subcalls usaran 1024 tokens, ideal para modelos de razonamiento.

---

### 2.5 Refactorizacion de `subcall_batch()` con `max_tokens` opcional

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 282-293

```diff
  def subcall_batch(
      chunks: object,
      *args: object,
      model: str | None = None,
-     max_tokens: int = 256,
+     max_tokens: int | None = None,
      chunk_size: int | None = None,
      overlap: int = 0,
      question: str | None = None,
      parallel: bool | None = None,
  ) -> list[str]:
+     if max_tokens is None:
+         max_tokens = self.subcall_max_tokens
      remaining_args = list(args)
```

**Por que**: Misma razon que `subcall()`. Consistencia en toda la API de subcalls.

---

### 2.6 Refactorizacion de `ask()`, `ask_chunk()`, `ask_chunked()`, `ask_chunks()`, `ask_chunks_first()`

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 354-435

```diff
- def ask(question: str, text: str, *, max_tokens: int = 256) -> str:
+ def ask(question: str, text: str, *, max_tokens: int | None = None) -> str:
      return subcall(f"Question: {question}\nSnippet:\n{text}", max_tokens=max_tokens)

- def ask_chunk(question: str, text: str, *, max_tokens: int = 256) -> str:
+ def ask_chunk(question: str, text: str, *, max_tokens: int | None = None) -> str:
      return ask(question, text, max_tokens=max_tokens)

  def ask_chunked(
      question: str,
      chunks: object,
      *,
-     max_tokens: int = 256,
+     max_tokens: int | None = None,
      ...

  def ask_chunks(
      question: str,
      chunks: object,
      *,
-     max_tokens: int = 256,
+     max_tokens: int | None = None,
      ...

  def ask_chunks_first(
      question: str,
      chunks: object,
      *,
-     max_tokens: int = 256,
+     max_tokens: int | None = None,
      ...
```

**Por que**: Todas estas funciones delegan a `subcall()` o `subcall_batch()`. Al pasar `None`, permiten que el fallback a `self.subcall_max_tokens` funcione correctamente en toda la cadena de llamadas.

---

### 2.7 Verificacion de longitud minima en `maybe_auto_finalize()`

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 504-509

```diff
  if cleaned.upper() == "NO_ANSWER" and self.fallback_code and not fallback_executed:
      last_error = "Auto-finalize blocked: NO_ANSWER."
      return None
+ if self.auto_finalize_min_length > 0 and len(cleaned) < self.auto_finalize_min_length:
+     last_error = (
+         f"Auto-finalize blocked: answer too short ({len(cleaned)} chars, "
+         f"minimum {self.auto_finalize_min_length}). Keep processing."
+     )
+     return None
  value = cleaned
```

**Por que**: Si el modelo asigna un valor parcial a la variable de auto-finalizacion (por ejemplo, una sola palabra o un esqueleto), el sistema lo acepta prematuramente. Con esta verificacion, si el valor tiene menos de N caracteres, se bloquea y se fuerza al modelo a seguir procesando. El mensaje de error se pasa al modelo como feedback para que sepa que debe continuar. Ver seccion 4.

---

### 2.8 Fallback inteligente en `MaxStepsExceeded`

**Archivo**: `src/pyrlm_runtime/rlm.py`
**Lineas**: 592-640

**ANTES** (1 linea):
```python
while True:
    policy.check_step()  # Lanzaba MaxStepsExceeded sin capturar
```
Cuando se agotaban los pasos, `MaxStepsExceeded` propagaba sin control y el resultado era una excepcion no manejada, o en configuraciones con `auto_finalize_var`, retornaba lo que hubiera (frecuentemente `NO_ANSWER`).

**DESPUES** (bloque completo):
```python
while True:
    try:
        policy.check_step()
    except MaxStepsExceeded:
        # 1. Revisar auto_finalize_var (aceptar incluso si es corto)
        if self.auto_finalize_var:
            value = repl.get(self.auto_finalize_var)
            if value is not None:
                text = str(value).strip()
                if text and text.upper() != "NO_ANSWER":
                    return text, trace

        # 2. Fallback inteligente: pedir resumen al modelo
        if self.conversation_history and history:
            try:
                summary_msgs = list(history) + [
                    {
                        "role": "user",
                        "content": (
                            "You have used all available steps. Based on all the "
                            "information you have gathered so far, provide your best "
                            "final answer NOW. Do NOT write code. Just write the answer "
                            "directly as plain text."
                        ),
                    }
                ]
                if self.max_history_tokens > 0:
                    summary_msgs = _trim_history(
                        summary_msgs, self.max_history_tokens
                    )
                summary_resp = self.adapter.complete(
                    summary_msgs,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                if summary_resp.text and summary_resp.text.strip():
                    trace.add(
                        TraceStep(
                            step_id=next_step_id(),
                            kind="root_call",
                            depth=0,
                            prompt_summary="[max_steps_summary]",
                            code=_truncate(summary_resp.text, 800),
                            usage=summary_resp.usage,
                        )
                    )
                    return summary_resp.text.strip(), trace
            except Exception:
                pass

        # 3. Ultimo recurso
        if last_stdout and last_stdout.strip():
            return last_stdout.strip(), trace
        return "NO_ANSWER", trace
```

**Por que**: Ver seccion 5 para la explicacion completa.

---

### 2.9 Fix de `default_response_parser` para modelos de razonamiento

**Archivo**: `src/pyrlm_runtime/adapters/generic_chat.py`
**Lineas**: 37-38

```diff
  def default_response_parser(data: dict[str, Any]) -> tuple[str, Usage | None]:
-     content = data["choices"][0]["message"]["content"]
+     message = data["choices"][0]["message"]
+     content = message.get("content") or ""
      usage_data = data.get("usage")
```

**ANTES**: Acceso directo a `["content"]`. Si el campo era `null` o no existia, lanzaba `KeyError` o `TypeError`.

**DESPUES**: Acceso seguro con `.get("content")` que retorna `None` si no existe, seguido de `or ""` que convierte `None` en string vacio.

**Por que**: Ver seccion 3 para la explicacion completa del problema de modelos de razonamiento.

---

## 3. El problema de los modelos de razonamiento ("Reasoning Model Problem")

### Contexto

Los modelos de lenguaje de nueva generacion (GPT-5, DeepSeek-v3.2, Qwen3-next, y similares) implementan un mecanismo de "razonamiento interno" (chain-of-thought) antes de generar la respuesta visible. Esto se manifiesta de dos formas en la API:

1. **Campo `content` nulo**: El modelo gasta todos los tokens en razonamiento interno y el campo `content` de la respuesta llega como `null` o no existe. Solo el campo `reasoning_content` tiene texto.

2. **Tokens consumidos por razonamiento**: De un presupuesto de 256 tokens, el modelo puede gastar 200 en razonamiento interno, dejando solo 56 tokens para la respuesta real. Esto resulta en respuestas truncadas o vacias.

### Formato tipico de respuesta de un modelo de razonamiento

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "reasoning_content": "<think>Let me analyze this text...\n...extensive internal reasoning...</think>"
    }
  }],
  "usage": {
    "prompt_tokens": 500,
    "completion_tokens": 256,
    "total_tokens": 756
  }
}
```

### Como se manifesto el problema

En benchmarks con modelos de razonamiento, el runtime fallaba con `TypeError` o `KeyError` en `default_response_parser` porque intentaba acceder a `data["choices"][0]["message"]["content"]` que era `null`. Esto causaba que toda la ejecucion del RLM fallara.

### Solucion implementada

**En `generic_chat.py`**: Acceso seguro al campo `content`:
```python
message = data["choices"][0]["message"]
content = message.get("content") or ""
```
- `.get("content")` retorna `None` si el campo no existe
- `or ""` convierte `None` en string vacio
- El runtime puede continuar operando con un string vacio en lugar de fallar

**En `rlm.py`**: Parametro `subcall_max_tokens` configurable:
```python
subcall_max_tokens: int = 256  # Aumentar a 512+ para modelos de razonamiento
```
- Permite al usuario asignar mas tokens a las subcalls
- Un solo punto de configuracion que afecta a todas las funciones (`subcall`, `subcall_batch`, `ask`, `ask_chunk`, `ask_chunked`, `ask_chunks`, `ask_chunks_first`)

### Ejemplo de uso

```python
# Para modelos de razonamiento, aumentar tokens de subcall
rlm = RLM(
    adapter=deepseek_adapter,
    subcall_max_tokens=1024,  # 4x el default para razonamiento
    max_tokens=2048,
)
```

---

## 4. El problema de la finalizacion prematura ("Premature Finalization Problem")

### Contexto

El mecanismo `auto_finalize_var` permite que el RLM finalice automaticamente cuando el modelo asigna un valor a una variable especifica en el REPL (tipicamente `answer`). El objetivo es evitar pasos innecesarios una vez que el modelo encuentra la respuesta.

### Como se manifesto el problema

En tareas complejas (especialmente con documentos largos o preguntas de multiples partes), el modelo a veces asignaba un valor parcial o esqueleto a la variable de finalizacion en los primeros pasos:

```python
# Paso 1: El modelo escribe esto
answer = "key"  # Solo una palabra, no la respuesta completa
```

El sistema detectaba que `answer` tenia un valor no vacio y no era `"NO_ANSWER"`, asi que finalizaba inmediatamente. El resultado era una respuesta incompleta como `"key"` cuando la respuesta correcta era `"The key term mentioned in the document is photosynthesis"`.

### Escenarios tipicos

1. **Respuesta esqueleto**: El modelo asigna un placeholder antes de procesarlo completamente.
2. **Respuesta parcial de primera iteracion**: El modelo encuentra una pista pero no ha analizado todo el contexto.
3. **Modelos de razonamiento**: Generan una respuesta corta externamente mientras razonan internamente, resultando en una asignacion prematura.

### Solucion implementada

Nuevo parametro `auto_finalize_min_length`:

```python
rlm = RLM(
    adapter=adapter,
    auto_finalize_var="answer",
    auto_finalize_min_length=20,  # Minimo 20 caracteres para aceptar
)
```

Logica en `maybe_auto_finalize()`:
```python
if self.auto_finalize_min_length > 0 and len(cleaned) < self.auto_finalize_min_length:
    last_error = (
        f"Auto-finalize blocked: answer too short ({len(cleaned)} chars, "
        f"minimum {self.auto_finalize_min_length}). Keep processing."
    )
    return None
```

### Comportamiento

- Si `auto_finalize_min_length = 0` (default): comportamiento anterior, cualquier valor no vacio finaliza.
- Si `auto_finalize_min_length = 20`: solo finaliza si la respuesta tiene al menos 20 caracteres.
- El mensaje de error se pasa al modelo como feedback, indicandole que debe continuar procesando.
- **Excepcion**: Cuando se agotan los pasos (`MaxStepsExceeded`), se acepta cualquier valor valido sin importar la longitud minima. Esto evita perder una respuesta parcial que podria ser util.

---

## 5. El fallback inteligente en agotamiento de pasos ("Max Steps Smart Fallback")

### Contexto

Cuando el RLM agota todos los pasos disponibles (`max_steps`), la politica lanza `MaxStepsExceeded`. Anteriormente, esto resultaba en una excepcion no manejada o un `NO_ANSWER` directo.

### Problema anterior

```
Pregunta: "Cual es el tema principal del documento?"
Resultado despues de 5 pasos: NO_ANSWER
```

Esto ocurria incluso cuando el modelo habia recopilado informacion parcial en pasos anteriores. Toda esa informacion se perdia.

### Solucion: cascada de tres niveles

El nuevo manejo de `MaxStepsExceeded` implementa una cascada de fallbacks:

#### Nivel 1: Revisar `auto_finalize_var`

```python
if self.auto_finalize_var:
    value = repl.get(self.auto_finalize_var)
    if value is not None:
        text = str(value).strip()
        if text and text.upper() != "NO_ANSWER":
            return text, trace
```

Si hay un valor en la variable de finalizacion (incluso si no cumple `auto_finalize_min_length`), se acepta. Al agotar los pasos, una respuesta parcial es mejor que ninguna.

#### Nivel 2: Pedir resumen al modelo

```python
if self.conversation_history and history:
    summary_msgs = list(history) + [{
        "role": "user",
        "content": (
            "You have used all available steps. Based on all the "
            "information you have gathered so far, provide your best "
            "final answer NOW. Do NOT write code. Just write the answer "
            "directly as plain text."
        ),
    }]
    summary_resp = self.adapter.complete(
        summary_msgs, max_tokens=self.max_tokens, temperature=0.0
    )
```

Si el historial de conversacion esta habilitado, el sistema le envia al modelo toda la conversacion acumulada con un prompt especial que le pide generar su mejor respuesta final basada en todo lo que ha recopilado. Esto es especialmente poderoso porque:

- El modelo tiene acceso a toda la informacion que recopilo en pasos anteriores (subcalls, resultados de REPL, etc.)
- Se le pide explicitamente que NO escriba codigo, solo texto plano
- El resultado se registra en el trace con `prompt_summary="[max_steps_summary]"` para poder identificarlo

#### Nivel 3: Ultimo recurso

```python
if last_stdout and last_stdout.strip():
    return last_stdout.strip(), trace
return "NO_ANSWER", trace
```

Si el resumen falla (excepcion, respuesta vacia), se usa el ultimo stdout del REPL. Solo si no hay nada disponible se retorna `NO_ANSWER`.

### Ejemplo de impacto

```
ANTES:
  Paso 1: subcall -> "The document discusses climate change"
  Paso 2: subcall -> "Specifically about ocean temperatures"
  Paso 3: subcall -> "And their effect on marine ecosystems"
  Paso 4: El modelo intenta sintetizar pero se queda sin pasos
  Resultado: NO_ANSWER

DESPUES:
  Paso 1-3: (igual que antes)
  Paso 4: Se agotan los pasos -> se pide resumen al modelo
  Modelo recibe toda la conversacion y genera:
  "The document discusses the impact of climate change on ocean
   temperatures and their effect on marine ecosystems."
  Resultado: respuesta sintetizada util
```

### Requisitos para que funcione el fallback inteligente

- `conversation_history=True` (default): El historial multi-turno debe estar habilitado para que el modelo tenga contexto.
- El adapter debe estar disponible para una llamada adicional.
- Se envuelve en `try/except Exception` para que fallos en esta llamada extra no rompan el runtime.

---

## 6. Tabla resumen de cambios

| Cambio | Archivo | Lineas | Tipo |
|--------|---------|--------|------|
| Importar `MaxStepsExceeded` | `rlm.py` | 13 | fix |
| Parametro `auto_finalize_min_length` | `rlm.py` | 45-47 | feat |
| Parametro `subcall_max_tokens` | `rlm.py` | 68-69 | feat |
| `subcall()` max_tokens opcional | `rlm.py` | 124-131 | refactor |
| `subcall_batch()` max_tokens opcional | `rlm.py` | 282-293 | refactor |
| `ask()` max_tokens opcional | `rlm.py` | 354 | refactor |
| `ask_chunk()` max_tokens opcional | `rlm.py` | 357 | refactor |
| `ask_chunked()` max_tokens opcional | `rlm.py` | 363 | refactor |
| `ask_chunks()` max_tokens opcional | `rlm.py` | 382 | refactor |
| `ask_chunks_first()` max_tokens opcional | `rlm.py` | 427 | refactor |
| Longitud minima en `maybe_auto_finalize()` | `rlm.py` | 504-509 | feat |
| Fallback inteligente en `MaxStepsExceeded` | `rlm.py` | 592-640 | feat |
| Fix null content en response parser | `generic_chat.py` | 37-38 | fix |

---

## 7. Impacto en la API publica

### Nuevos parametros del dataclass `RLM`

```python
@dataclass
class RLM:
    # ... parametros existentes ...
    auto_finalize_min_length: int = 0   # NUEVO
    subcall_max_tokens: int = 256       # NUEVO
```

### Cambios en firmas de funciones internas (no rompen compatibilidad)

Todas las funciones cambiaron `max_tokens: int = 256` a `max_tokens: int | None = None`. Esto es **retrocompatible** porque:
- Llamadas sin el parametro siguen funcionando (usan `self.subcall_max_tokens` que por defecto es 256)
- Llamadas con `max_tokens=N` siguen funcionando (se usa el valor proporcionado)

### Configuracion recomendada para modelos de razonamiento

```python
rlm = RLM(
    adapter=reasoning_model_adapter,
    subcall_max_tokens=1024,         # Mas espacio para razonamiento
    auto_finalize_var="answer",
    auto_finalize_min_length=20,     # Evitar esqueletos
    conversation_history=True,        # Necesario para fallback inteligente
    policy=Policy(max_steps=8),
)
```
