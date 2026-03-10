# Resumen de Cambios en pyrlm-runtime para Benchmarking 23F

## Documento de referencia para actualizar la presentacion

---

## 1. Vision General

Durante las sesiones de benchmarking con el corpus desclasificado del 23F, se realizaron mejoras
arquitectonicas significativas en la libreria `pyrlm-runtime`. Los cambios se centran en dos areas
principales:

1. **Robustez del bucle RLM**: El runtime ahora maneja de forma inteligente la terminacion por
   agotamiento de pasos (`MaxStepsExceeded`), evitando perdida de trabajo parcial. Ademas, se
   anadio un umbral minimo de longitud para la auto-finalizacion, previniendo que el modelo
   finalice prematuramente con respuestas incompletas.

2. **Compatibilidad con modelos de razonamiento**: Se corrigio el parser de respuestas del
   adaptador generico para manejar correctamente respuestas con `content: null`, tipicas de
   modelos de razonamiento (e.g., DeepSeek-R1, o3-mini).

3. **Configurabilidad de subcalls**: Se anadio un parametro `subcall_max_tokens` que permite
   controlar independientemente el presupuesto de tokens para las llamadas a sub-LLMs, critico
   cuando los modelos de subcall necesitan mas espacio para respuestas detalladas.

---

## 2. Cambios Detallados por Archivo

### 2.1 `src/pyrlm_runtime/rlm.py` -- Archivo principal del runtime

#### 2.1.1 Nuevo parametro: `subcall_max_tokens`

Se anadio un parametro al dataclass `RLM` para controlar el maximo de tokens en respuestas de
subcalls, independiente del `max_tokens` del modelo raiz.

**Antes:**
```python
@dataclass
class RLM:
    adapter: ModelAdapter
    policy: Policy | None = None
    cache: FileCache | None = None
    max_tokens: int = 512
    # ... otros parametros ...
    # No existia subcall_max_tokens
```

**Despues:**
```python
@dataclass
class RLM:
    adapter: ModelAdapter
    policy: Policy | None = None
    cache: FileCache | None = None
    max_tokens: int = 512
    # ... otros parametros ...
    # Default max_tokens for subcall responses (increase for reasoning models)
    subcall_max_tokens: int = 256
```

**Impacto:** Antes, los subcalls usaban un valor fijo de 256 tokens hardcodeado dentro de las
funciones. Ahora es configurable desde la instancia del RLM y se propaga automaticamente a todas
las funciones de subcall (`subcall()`, `subcall_batch()`, `ask()`, `ask_chunks()`, etc.).

#### 2.1.2 Propagacion de `subcall_max_tokens` a todas las funciones internas

Cada funcion interna que ejecuta subcalls ahora usa `self.subcall_max_tokens` como valor por
defecto cuando no se especifica `max_tokens` explicitamente.

**En la funcion `subcall()`:**
```python
def subcall(
    text: str,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    depth: int = 1,
) -> str:
    if max_tokens is None:
        max_tokens = self.subcall_max_tokens
    # ... resto de la logica
```

**En la funcion `subcall_batch()`:**
```python
def subcall_batch(
    chunks: object,
    *args: object,
    model: str | None = None,
    max_tokens: int | None = None,
    # ...
) -> list[str]:
    if max_tokens is None:
        max_tokens = self.subcall_max_tokens
    # ... resto de la logica
```

**En la funcion `ask()`:**
```python
def ask(question: str, text: str, *, max_tokens: int | None = None) -> str:
    return subcall(f"Question: {question}\nSnippet:\n{text}", max_tokens=max_tokens)
```

El patron `max_tokens: int | None = None` con fallback a `self.subcall_max_tokens` se aplica
de forma consistente en: `subcall()`, `subcall_batch()`, `ask()`, `ask_chunk()`, `ask_chunked()`,
`ask_chunks()` y `ask_chunks_first()`.

#### 2.1.3 Nuevo parametro: `auto_finalize_min_length`

Se anadio un umbral minimo de caracteres que debe cumplir la variable de auto-finalizacion para
ser aceptada. Esto previene que el modelo almacene una respuesta parcial corta en `final_answer`
y el sistema la acepte prematuramente.

**Declaracion en el dataclass:**
```python
@dataclass
class RLM:
    # ...
    auto_finalize_var: str | None = None
    # Minimum character length for auto_finalize_var to trigger (prevents premature finalization)
    auto_finalize_min_length: int = 0
    # ...
```

**Logica de validacion en `maybe_auto_finalize()`:**
```python
def maybe_auto_finalize() -> str | None:
    nonlocal last_error
    if not self.auto_finalize_var:
        return None
    value = repl.get(self.auto_finalize_var)
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            last_error = "Auto-finalize blocked: empty value."
            return None
        if cleaned.upper() == "NO_ANSWER" and self.fallback_code and not fallback_executed:
            last_error = "Auto-finalize blocked: NO_ANSWER."
            return None
        # --- NUEVO: Verificacion de longitud minima ---
        if self.auto_finalize_min_length > 0 and len(cleaned) < self.auto_finalize_min_length:
            last_error = (
                f"Auto-finalize blocked: answer too short ({len(cleaned)} chars, "
                f"minimum {self.auto_finalize_min_length}). Keep processing."
            )
            return None
        # --- FIN NUEVO ---
        value = cleaned
    # ... verificacion de can_finalize ...
```

**Comportamiento:**
- Si `auto_finalize_min_length = 0` (por defecto): no hay restriccion de longitud (comportamiento
  original).
- Si `auto_finalize_min_length = 2000`: la variable `final_answer` debe tener al menos 2000
  caracteres para que el sistema la acepte como respuesta final.
- Cuando se bloquea, el mensaje de error se pasa al modelo para que sepa que debe seguir
  procesando: `"Auto-finalize blocked: answer too short (450 chars, minimum 2000). Keep processing."`

#### 2.1.4 Smart Fallback en `MaxStepsExceeded`

Este es el cambio mas significativo para la robustez del benchmark. Antes, cuando se agotaban los
pasos, el sistema simplemente lanzaba la excepcion y perdia todo el trabajo parcial.

**Antes (comportamiento original):**
```python
while True:
    try:
        policy.check_step()
    except MaxStepsExceeded:
        # Se perdia todo el trabajo acumulado
        return "NO_ANSWER", trace
```

**Despues (smart fallback de 3 etapas):**
```python
while True:
    try:
        policy.check_step()
    except MaxStepsExceeded:
        # ETAPA 1: Intentar recuperar auto_finalize_var (relajando min_length)
        if self.auto_finalize_var:
            value = repl.get(self.auto_finalize_var)
            if value is not None:
                text = str(value).strip()
                if text and text.upper() != "NO_ANSWER":
                    return text, trace

        # ETAPA 2: Pedir al modelo un resumen final con el historial acumulado
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
                    summary_msgs = _trim_history(summary_msgs, self.max_history_tokens)
                summary_resp = self.adapter.complete(
                    summary_msgs, max_tokens=self.max_tokens, temperature=0.0
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

        # ETAPA 3: Fallback a ultimo stdout
        if last_stdout and last_stdout.strip():
            return last_stdout.strip(), trace
        return "NO_ANSWER", trace
```

**Las 3 etapas del smart fallback:**

| Etapa | Accion | Cuando aplica |
|-------|--------|---------------|
| 1 | Recuperar `auto_finalize_var` sin verificar `min_length` | Cuando hay una respuesta parcial almacenada en la variable de finalizacion |
| 2 | Pedir al modelo un resumen final basado en el historial de conversacion | Cuando `conversation_history=True` y hay historial acumulado |
| 3 | Devolver el ultimo stdout del REPL | Cuando hay output previo del REPL |

**Por que es critico:** En el benchmark del 23F, el RLM procesa ~167 documentos via subcalls.
Con 40 pasos maximos, es comun que el modelo acumule informacion significativa pero no llegue a
formular la respuesta final. Sin el smart fallback, todo ese procesamiento (potencialmente cientos
de subcalls y miles de tokens) se perdia, devolviendo "NO_ANSWER".

---

### 2.2 `src/pyrlm_runtime/adapters/generic_chat.py` -- Adaptador generico

#### 2.2.1 Fix para `content: null` en modelos de razonamiento

Algunos modelos de razonamiento (e.g., DeepSeek-R1, o3-mini via OpenRouter) devuelven
`"content": null` en la respuesta del mensaje, colocando el contenido util en otros campos
como `reasoning_content`. El parser original fallaba con `TypeError` porque intentaba operar
sobre `None`.

**Antes:**
```python
def default_response_parser(data: dict[str, Any]) -> tuple[str, Usage | None]:
    message = data["choices"][0]["message"]
    content = message.get("content", "")
    usage_data = data.get("usage")
    usage = Usage.from_dict(usage_data) if usage_data else None
    return content, usage
```

**Despues:**
```python
def default_response_parser(data: dict[str, Any]) -> tuple[str, Usage | None]:
    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    usage_data = data.get("usage")
    usage = Usage.from_dict(usage_data) if usage_data else None
    return content, usage
```

**Diferencia clave:**
- `message.get("content", "")` -- Si `content` existe con valor `null`/`None`, devuelve `None`
  (porque la clave SI existe, solo tiene valor nulo). Esto causaba errores downstream.
- `message.get("content") or ""` -- Si `content` es `None`, `False`, `""`, o no existe, siempre
  devuelve `""`. Es defensivo contra valores nulos.

**JSON de ejemplo que causaba el fallo:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "reasoning_content": "Let me analyze the documents..."
    }
  }]
}
```

---

## 3. Parametros Anadidos al Dataclass RLM

Resumen completo de los parametros nuevos en el dataclass `RLM`:

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `subcall_max_tokens` | `int` | `256` | Maximo de tokens para respuestas de subcalls. Se propaga a `subcall()`, `subcall_batch()`, `ask()`, `ask_chunks()`, etc. Aumentar para modelos de razonamiento que necesitan mas espacio. |
| `auto_finalize_min_length` | `int` | `0` | Longitud minima en caracteres que debe tener `auto_finalize_var` para ser aceptada. `0` = sin restriccion. Previene finalizacion prematura con respuestas cortas. |

Parametros que ya existian pero son relevantes para la configuracion del benchmark:

| Parametro | Tipo | Default | Descripcion |
|-----------|------|---------|-------------|
| `auto_finalize_var` | `str \| None` | `None` | Nombre de variable REPL que, al ser asignada, dispara auto-finalizacion |
| `subcall_adapter` | `ModelAdapter \| None` | `None` | Adaptador diferente (mas barato) para subcalls |
| `parallel_subcalls` | `bool` | `False` | Ejecutar subcalls en paralelo |
| `max_concurrent_subcalls` | `int` | `10` | Maximo de subcalls simultaneos |
| `conversation_history` | `bool` | `True` | Historial multi-turno para auto-correccion |
| `max_history_tokens` | `int` | `0` | Presupuesto de tokens para historial (0=ilimitado) |
| `require_repl_before_final` | `bool` | `False` | Forzar ejecucion REPL antes de finalizar |

---

## 4. Configuracion en el Benchmark (`step2_run_benchmark.py`)

El script de benchmark (`23-f-project/step2_run_benchmark.py`) demuestra como se configuran
todos estos parametros en conjunto para un caso de uso real.

### 4.1 Instanciacion del RLM en el benchmark

```python
rlm_max_tokens = int(os.getenv("RLM_MAX_TOKENS", "4096"))
rlm_subcall_max_tokens = int(os.getenv("RLM_SUBCALL_MAX_TOKENS", "2048"))

rlm = RLM(
    adapter=root_adapter,                    # openai/gpt-5
    subcall_adapter=subcall_adapter,         # openai/gpt-5-mini (mas barato)
    policy=policy,                           # max_steps=40, max_subcalls=300
    cache=cache,                             # Cache en disco para reutilizar subcalls
    system_prompt=system_prompt,             # Prompt especializado en documentos 23F
    subcall_system_prompt=subcall_prompt,    # Prompt de analisis documental
    max_tokens=rlm_max_tokens,              # 4096 tokens para respuesta raiz
    subcall_max_tokens=rlm_subcall_max_tokens,  # 2048 tokens para subcalls (NUEVO)
    conversation_history=True,               # Historial multi-turno habilitado
    max_history_tokens=40_000,               # Presupuesto de 40K tokens de historial
    parallel_subcalls=True,                  # Subcalls en paralelo
    max_concurrent_subcalls=20,              # Hasta 20 subcalls simultaneos
    require_repl_before_final=True,          # Forzar uso del REPL
    auto_finalize_var="final_answer",        # Variable de auto-finalizacion
    auto_finalize_min_length=2000,           # Minimo 2000 chars (NUEVO)
)
```

### 4.2 Configuracion de Policy

```python
policy = Policy(
    max_steps=40,              # Hasta 40 iteraciones del bucle principal
    max_subcalls=300,          # Hasta 300 llamadas a sub-LLMs
    max_total_tokens=2_000_000,  # Presupuesto total de 2M tokens
)
```

### 4.3 Configuracion de modelos (2 niveles)

```python
ROOT_MODEL = os.getenv("RLM_ROOT_MODEL", "openai/gpt-5")          # $1.25/M input
SUBCALL_MODEL = os.getenv("RLM_SUBCALL_MODEL", "openai/gpt-5-mini")  # $0.25/M input

root_adapter = OpenAICompatAdapter(model=ROOT_MODEL, base_url=BASE_URL)
subcall_adapter = OpenAICompatAdapter(model=SUBCALL_MODEL, base_url=BASE_URL)
```

### 4.4 Variables de entorno para configuracion

El benchmark permite sobreescribir la configuracion via variables de entorno:

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `RLM_ROOT_MODEL` | `openai/gpt-5` | Modelo raiz (orquestador) |
| `RLM_SUBCALL_MODEL` | `openai/gpt-5-mini` | Modelo para subcalls (extraccion) |
| `RLM_BASELINE_MODEL` | (=ROOT_MODEL) | Modelo para la linea base |
| `RLM_MAX_TOKENS` | `4096` | Max tokens respuesta raiz |
| `RLM_SUBCALL_MAX_TOKENS` | `2048` | Max tokens respuesta subcalls |
| `RLM_BASELINE_MAX_CHARS` | `300000` | Chars enviados a baseline (truncamiento) |
| `RLM_BASELINE_MAX_TOKENS` | `4096` | Max tokens respuesta baseline |
| `RLM_QUERIES` | `all` | Filtro de queries (`all`, `Q1`, `Q1,Q3`) |
| `RLM_NO_SUBCALL_MODEL` | `0` | Si `1`, usa el mismo modelo para root y subcalls |
| `LLM_BASE_URL` | OpenRouter | URL base de la API |
| `LLM_API_KEY` | -- | Clave de API |

---

## 5. Flujo Completo del Smart Fallback

Diagrama de decision cuando se agotan los pasos:

```
MaxStepsExceeded capturado
        |
        v
[1] auto_finalize_var existe?
        |
       SI --> repl.get(auto_finalize_var)
        |          |
        |     tiene valor valido? (no vacio, no "NO_ANSWER")
        |          |
        |         SI --> RETORNAR valor (sin verificar min_length)
        |          |
        |         NO --> continuar
        |
       NO --> continuar
        |
        v
[2] conversation_history habilitado y hay historial?
        |
       SI --> Enviar historial + mensaje "da tu mejor respuesta ahora"
        |          |
        |     respuesta valida?
        |          |
        |         SI --> RETORNAR respuesta del modelo
        |          |
        |         NO --> continuar (excepcion silenciada)
        |
       NO --> continuar
        |
        v
[3] Hay ultimo stdout del REPL?
        |
       SI --> RETORNAR ultimo stdout
        |
       NO --> RETORNAR "NO_ANSWER"
```

**Nota critica:** En la Etapa 1, el `auto_finalize_min_length` NO se verifica. Esto es
intencional: cuando se agotan los pasos, es preferible devolver una respuesta parcial (aunque
sea corta) a perder todo el trabajo. El `min_length` solo aplica durante el bucle normal para
forzar al modelo a seguir procesando cuando aun tiene pasos disponibles.

---

## 6. Impacto en los Resultados del Benchmark

Estas mejoras fueron necesarias por los siguientes problemas observados durante el benchmarking:

### Problema 1: Perdida de trabajo en MaxStepsExceeded
- **Sintoma:** El RLM procesaba 150+ documentos via subcalls pero al agotar los 40 pasos
  devolvia "NO_ANSWER".
- **Causa:** No habia mecanismo de recuperacion graceful.
- **Solucion:** Smart fallback de 3 etapas.

### Problema 2: Finalizacion prematura
- **Sintoma:** El modelo almacenaba una respuesta parcial corta en `final_answer` despues de
  procesar solo los primeros documentos, y el sistema la aceptaba como final.
- **Causa:** `auto_finalize_var` no tenia umbral de calidad.
- **Solucion:** `auto_finalize_min_length=2000` fuerza al modelo a seguir procesando.

### Problema 3: Subcalls con respuestas truncadas
- **Sintoma:** Las extracciones de subcalls (nombres, fechas, eventos) venian cortadas.
- **Causa:** El default de 256 tokens era insuficiente para analisis detallados.
- **Solucion:** `subcall_max_tokens=2048` configurable, ajustado a 2048 para el benchmark.

### Problema 4: Error con modelos de razonamiento
- **Sintoma:** `TypeError: expected str, got NoneType` al usar modelos con output de razonamiento.
- **Causa:** `content: null` en la respuesta JSON.
- **Solucion:** `message.get("content") or ""` en el parser.

---

## 7. Resumen de Archivos Modificados

| Archivo | Cambio | Lineas clave |
|---------|--------|-------------|
| `src/pyrlm_runtime/rlm.py` | Parametro `subcall_max_tokens` (linea 69) | Dataclass + propagacion a subcall/subcall_batch/ask/ask_chunks |
| `src/pyrlm_runtime/rlm.py` | Parametro `auto_finalize_min_length` (linea 47) | Dataclass + validacion en `maybe_auto_finalize()` (lineas 504-509) |
| `src/pyrlm_runtime/rlm.py` | Smart fallback en MaxStepsExceeded (lineas 597-640) | 3 etapas de recuperacion graceful |
| `src/pyrlm_runtime/adapters/generic_chat.py` | Fix `content or ""` (linea 38) | `default_response_parser` |
| `23-f-project/step2_run_benchmark.py` | Configuracion usando los nuevos parametros (lineas 293-309) | Instanciacion del RLM con subcall_max_tokens y auto_finalize_min_length |

---

## 8. Sesion 2: min_steps + SmartRouter + Corpus Interleaving

### 8.1 Vision General

Tras los benchmarks con gpt-oss:120b, se identificaron 3 problemas adicionales:

1. **RLM finaliza demasiado pronto**: Con `auto_finalize_min_length=2000` el modelo paraba en step 2 con solo 2K chars. No habia forma de forzar un minimo de iteraciones.
2. **SmartRouter inutilizable**: El SmartRouter existente tenia baseline hardcodeado (max_tokens=512, prompt generico "You are a helpful assistant"), no podia pasar la mayoria de parametros al RLM, y no tenia tests.
3. **Corpus sesgado**: Los documentos estaban ordenados Interior->Defensa->Exteriores. El baseline (que ve ~16% del corpus) solo veia docs de Interior, dandole una ventaja injusta en cronologia.

### 8.2 Nuevo parametro: `min_steps` en `src/pyrlm_runtime/rlm.py`

**Problema:** El modelo puede emitir `FINAL:` o asignar `final_answer` en step 1-2, antes de explorar adecuadamente el corpus.

**Solucion:** Nuevo campo en el dataclass RLM:
```python
@dataclass
class RLM:
    # ...
    # Minimum number of steps before finalization is allowed (0 = no minimum).
    min_steps: int = 0
```

**Logica implementada en 3 puntos:**

**a) `maybe_auto_finalize()`** -- Bloquea auto-finalizacion si no se alcanzo `min_steps`:
```python
if self.min_steps > 0 and policy.steps < self.min_steps:
    last_error = (
        f"Auto-finalize blocked: step {policy.steps}/{self.min_steps} "
        f"(min_steps={self.min_steps}). Keep processing."
    )
    return None
```

**b) `_can_finalize()`** -- Funcion de guarda ampliada con `min_steps` y `current_step`:
```python
def _can_finalize(
    *,
    require_repl: bool,
    repl_executed: bool,
    require_subcall: bool,
    subcall_made: bool,
    min_steps: int = 0,       # NUEVO
    current_step: int = 0,    # NUEVO
) -> bool:
    if min_steps > 0 and current_step < min_steps:
        return False
    # ... checks existentes ...
```

**c) `_guard_error()`** -- Mensaje de error descriptivo para el modelo:
```python
def _guard_error(..., min_steps: int = 0, current_step: int = 0) -> str:
    if min_steps > 0 and current_step < min_steps:
        return f"Guard: step {current_step}/{min_steps}, keep exploring before FINAL."
    # ... checks existentes ...
```

**Comportamiento critico:** El handler de `MaxStepsExceeded` NO es afectado por `min_steps`. Si el modelo agota los pasos, el smart fallback de 3 etapas sigue funcionando normalmente y devuelve lo que haya disponible.

**Se actualizaron 6 call sites de `_can_finalize()` y 2 de `_guard_error()`** para pasar los nuevos parametros.

---

### 8.3 Refactorizacion de SmartRouter en `src/pyrlm_runtime/router.py`

#### 8.3.1 Problemas arquitectonicos resueltos

| Problema | Antes | Despues |
|----------|-------|---------|
| Baseline hardcodeado | `max_tokens=512`, system prompt = "You are a helpful assistant" | Configurable via `RouterConfig.baseline_max_tokens` y `RouterConfig.baseline_system_prompt` |
| No podia pasar parametros RLM | Solo aceptaba adapter, policy, system_prompt, fallback_code, auto_finalize_var | Nuevo `rlm_extra_kwargs: dict` acepta cualquier parametro RLM |
| Sin tests | 0 tests | 6 tests en nuevo `tests/test_router.py` |

#### 8.3.2 Nuevos campos en `RouterConfig`

```python
@dataclass
class RouterConfig:
    baseline_threshold: int = 8000
    default_profile: ExecutionProfile = ExecutionProfile.DETERMINISTIC_FIRST
    baseline_max_tokens: int = 512                # NUEVO
    baseline_system_prompt: str | None = None      # NUEVO
    auto_calibrate: bool = False
    # ...
```

- **`baseline_max_tokens`**: Controla el presupuesto de tokens del baseline. Antes estaba hardcodeado a 512 (insuficiente para tareas reales). En el benchmark se configura a 8192.
- **`baseline_system_prompt`**: Permite un prompt de sistema diferente para baseline vs RLM. Cascada: `config.baseline_system_prompt` -> `SmartRouter.system_prompt` -> "You are a helpful assistant."

#### 8.3.3 Nuevo parametro: `rlm_extra_kwargs`

```python
class SmartRouter:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        # ... parametros existentes ...
        rlm_extra_kwargs: dict[str, Any] | None = None,  # NUEVO
        logger: logging.Logger | None = None,
    ) -> None:
```

**Flujo de merge en `_run_rlm()`:**
```python
def _run_rlm(self, query, context, profile):
    # 1. Profile kwargs (estrategia: parallelism, guards, recursion)
    rlm_kwargs = self._build_rlm_kwargs(profile)
    # 2. Extra kwargs override (user-level config)
    rlm_kwargs.update(self.rlm_extra_kwargs)
    # 3. Crear RLM con todo
    rlm = RLM(adapter=..., **rlm_kwargs)
```

**Orden de precedencia:** `base_kwargs` < `profile overrides` < `rlm_extra_kwargs`. El usuario puede forzar cualquier parametro RLM independientemente del perfil seleccionado.

#### 8.3.4 Baseline mejorado

```python
def _run_baseline(self, query, context):
    sys_prompt = (
        self.config.baseline_system_prompt
        or self.system_prompt
        or "You are a helpful assistant."
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    response = self.adapter.complete(
        messages,
        max_tokens=self.config.baseline_max_tokens,
        temperature=0.0,
    )
```

En el benchmark, SmartRouter comparte el mismo `system_prompt` (analista 23F experto) entre baseline y RLM, asegurando una comparacion justa.

---

### 8.4 Interleaving del corpus en `step2_run_benchmark.py`

**Problema:** El corpus estaba ordenado por ministerio: Interior (28 docs) -> Defensa (108 docs) -> Exteriores (31 docs). El baseline ve solo los primeros 300K chars (~16%), que correspondian exclusivamente a Interior. Interior contiene los documentos mas ricos en cronologia (telex GC, transcripciones telefonicas), dando al baseline una ventaja injusta en Q1.

**Solucion:** Funcion `_interleave_by_ministry()` que redistribuye round-robin:

```python
def _interleave_by_ministry(documents: list[dict]) -> list[dict]:
    """Round-robin interleave docs by ministry for fair baseline coverage."""
    by_ministry: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        by_ministry[doc["ministry"]].append(doc)
    interleaved: list[dict] = []
    queues = list(by_ministry.values())
    max_len = max(len(q) for q in queues)
    for i in range(max_len):
        for q in queues:
            if i < len(q):
                interleaved.append(q[i])
    return interleaved
```

**Efecto:** En los primeros 300K chars, el baseline ahora ve ~9 docs de cada ministerio en vez de solo Interior. Esto elimina el sesgo documental y hace la comparacion mas justa.

---

### 8.5 Benchmark refactorizado con SmartRouter

El benchmark ahora usa SmartRouter en vez de instanciar RLM directamente:

```python
from pyrlm_runtime import SmartRouter, RouterConfig, ExecutionProfile, Policy

QUERY_PROFILES = {
    "Q1_cronologia": ExecutionProfile.HYBRID,
    "Q2_red_implicados": ExecutionProfile.SEMANTIC_BATCHES,
    "Q3_papel_rey": ExecutionProfile.HYBRID,
}

router = SmartRouter(
    root_adapter,
    subcall_adapter=subcall_adapter,
    config=RouterConfig(
        baseline_threshold=8000,
        baseline_max_tokens=8192,
        default_profile=ExecutionProfile.HYBRID,
    ),
    policy=Policy(max_steps=40, max_subcalls=300, max_total_tokens=2_000_000),
    system_prompt=SYSTEM_PROMPT,
    auto_finalize_var="final_answer",
    rlm_extra_kwargs={
        "subcall_system_prompt": SUBCALL_PROMPT,
        "max_tokens": 8192,
        "subcall_max_tokens": 4096,
        "conversation_history": True,
        "max_history_tokens": 80_000,
        "max_concurrent_subcalls": 20,
        "auto_finalize_min_length": 8000,
        "min_steps": 5,
        "require_repl_before_final": True,
        "parallel_subcalls": True,
    },
)
result = router.run(query, context, profile=ExecutionProfile.HYBRID)
```

**SmartRouter decide automaticamente** si usar baseline (contexto < 8K chars) o RLM (contexto >= 8K chars). Para el corpus 23F (~1.8M chars), siempre elige RLM.

### 8.6 Parametros actualizados (sesion 2)

| Parametro | Valor anterior | Valor nuevo | Motivo |
|-----------|---------------|-------------|--------|
| `RLM_MAX_TOKENS` default | 4096 | 8192 | Respuestas raiz mas largas |
| `RLM_SUBCALL_MAX_TOKENS` default | 2048 | 4096 | Extracciones mas detalladas |
| `RLM_BASELINE_MAX_TOKENS` default | 4096 | 8192 | Respuestas baseline mas largas |
| `auto_finalize_min_length` | 2000 | 8000 | Forzar respuestas completas |
| `min_steps` | N/A (nuevo) | 5 | Minimo 5 iteraciones antes de parar |
| `max_history_tokens` | 40,000 | 80,000 | Menos perdida de contexto |
| System prompt min chars | "at least 2000 characters" | "at least 8000 characters" | Alineado con min_length |

---

### 8.7 Tests anadidos

#### 4 tests de regresion para `min_steps` en `tests/test_rlm_loop.py`:

| Test | Que verifica |
|------|-------------|
| `test_min_steps_blocks_early_auto_finalize` | auto-finalize NO se activa antes de min_steps |
| `test_min_steps_blocks_explicit_final` | FINAL: es rechazado antes de min_steps |
| `test_min_steps_zero_allows_immediate_finalize` | Con min_steps=0, FINAL funciona normalmente |
| `test_min_steps_does_not_block_at_max_steps_exhaustion` | MaxStepsExceeded bypasses min_steps |

#### 6 tests para SmartRouter en `tests/test_router.py` (archivo nuevo):

| Test | Que verifica |
|------|-------------|
| `test_router_baseline_below_threshold` | Context < threshold -> method="baseline" |
| `test_router_rlm_above_threshold` | Context > threshold -> method="rlm" |
| `test_router_baseline_uses_config_max_tokens` | baseline_max_tokens llega al adapter |
| `test_router_baseline_uses_system_prompt` | Baseline usa system_prompt del router |
| `test_router_baseline_system_prompt_cascade` | config.baseline_system_prompt > system_prompt |
| `test_router_rlm_extra_kwargs_merged` | rlm_extra_kwargs se propagan al RLM |

---

### 8.8 Archivos modificados (sesion 2)

| Archivo | Cambio |
|---------|--------|
| `src/pyrlm_runtime/rlm.py` | `min_steps` field + logica en `maybe_auto_finalize`, `_can_finalize`, `_guard_error` |
| `src/pyrlm_runtime/router.py` | `rlm_extra_kwargs` + `baseline_max_tokens`/`baseline_system_prompt` en RouterConfig + `_run_baseline` mejorado |
| `tests/test_rlm_loop.py` | 4 tests de regresion para `min_steps` |
| `tests/test_router.py` | 6 tests SmartRouter (archivo nuevo) |
| `23-f-project/step2_run_benchmark.py` | Interleaving + SmartRouter + parametros actualizados |
