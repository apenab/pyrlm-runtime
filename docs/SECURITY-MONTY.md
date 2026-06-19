# Nota de seguridad: cómo ejecuta RLM el código generado por el modelo (backend Monty)

## El planteamiento

En un RLM, el modelo de lenguaje **escribe código Python y ese código se
ejecuta**. Esa frase es la que dispara una revisión de seguridad. La pregunta
correcta no es "¿se ejecuta código?" (sí), sino **"¿dónde se ejecuta y qué puede
hacer?"**.

Con el backend Monty, el código no confiable del modelo corre en un intérprete
aislado en Rust con capacidades casi nulas, y su única salida al exterior es una
lista explícita de funciones que nosotros controlamos.

## Hay que distinguir dos cosas

1. **El código que escribe el LLM** — la entrada **no confiable**. Es arbitrario.
2. **Las "funciones al vuelo"** (`ask`, `llm_query`, `ctx.find(...)`, …) — **código
   nuestro, auditado**, definido en `rlm.py`. El modelo **no** las escribe; solo
   puede *invocarlas por nombre*.

La confusión típica es pensar que el modelo "crea funciones que se ejecutan con
permisos totales". No es así. El modelo solo puede **llamar** a una lista cerrada
de funciones que registramos previamente.

## Dónde se ejecuta el código del modelo: NO es `exec()` de Python

Con `repl_backend="monty"`, el código del LLM **nunca llega al intérprete de
Python del proceso anfitrión**. Se ejecuta dentro de
[pydantic-monty](https://github.com/pydantic/pydantic-monty), un intérprete de un
subconjunto de Python **escrito en Rust** y diseñado como sandbox. En cada paso
se crea una instancia nueva de Monty (`env_monty.py:266`):

```python
monty = Monty(code, inputs=..., external_functions=ext_fn_names)
monty.run(external_functions=dict(self._external_fns), limits=..., ...)
```

## Qué NO puede hacer el código del modelo (el argumento central)

El código del modelo, corriendo en Monty:

- **No puede importar módulos** — no hay `import os`, `socket`, `subprocess`. No
  existe el sistema de imports.
- **Sin acceso al sistema de ficheros** — no puede abrir, leer ni escribir
  archivos.
- **Sin acceso a red** — no puede abrir sockets ni hacer peticiones.
- **No puede ejecutar comandos del sistema** ni lanzar procesos.
- **No puede acceder a la memoria ni a las variables del proceso anfitrión.**
  Monty es un intérprete separado: el código solo ve los *inputs* que le pasamos
  explícitamente y las funciones externas que registramos.
- **No es Python real**, así que los trucos clásicos de fuga de sandbox de Python
  (`__builtins__`, `().__class__.__mro__`, `eval`, …) no aplican: ese modelo de
  objetos no existe en Monty.

## La única puerta hacia el exterior: la allowlist de funciones externas

El código del modelo solo puede salir del sandbox de **una** forma: llamando a una
función que **nosotros hayamos registrado explícitamente** como función externa.
Monty solo conoce los nombres que le pasamos en `external_functions`
(`env_monty.py:264, 277-278`). Una función que no esté en esa lista **no existe**
para el modelo.

Esas funciones externas sí son código Python real que corre en el proceso. Hoy
son, en esencia:

- `ask`, `llm_query`, `llm_batch`, … → hacen una llamada al proveedor del LLM
  (salida de red controlada y conocida, la misma que ya hace cualquier app que
  use el modelo).
- Métodos de `Context` (`ctx.find`, `ctx.chunk`, `ctx.slice`) → solo
  leen/segmentan texto **que ya está en memoria** (el documento que se está
  procesando). No abren nada nuevo.

Es decir: la superficie de ataque real no es "el código del modelo", sino
**exactamente esta lista de funciones registradas** — finita, pequeña y en
nuestro repositorio.

Mecanismo de registro: en `set()`, un callable se guarda en `self._external_fns`
(`env_monty.py:167-169`); los métodos de objetos complejos como `Context` se
aplanan a `ctx__find`, `ctx__chunk`, … (`_register_object`, `env_monty.py:200-220`),
y un reescritor de AST traduce `ctx.find(...)` → `ctx__find(...)` antes de
ejecutar. El modelo escribe Python idiomático, pero por debajo solo puede llamar
a funciones de la lista.

## Límites de recursos (anti-DoS)

Cada ejecución en Monty se corre con límites duros aplicados por el runtime de
Rust (`MontyLimits`, `env_monty.py:37-52`):

| Límite | Valor por defecto |
|---|---|
| Tiempo de ejecución | 5 s |
| Memoria | 128 MB |
| Asignaciones | 1.000.000 |
| Profundidad de recursión | 100 |

Esto acota el coste de un bucle infinito o de una bomba de memoria escrita por el
modelo: la ejecución se aborta al superar el límite.

## Frontera de confianza

```
┌─ Entrada NO confiable ─────────────────────────────┐
│  Código generado por el LLM                         │
│        │ se ejecuta dentro de ↓                     │
│  ┌─ Sandbox Monty (intérprete Rust) ─────────────┐  │
│  │  sin import · sin FS · sin red · sin shell     │  │
│  │  sin acceso a la memoria del proceso anfitrión │  │
│  │  límites de CPU/memoria/recursión              │  │
│  │       │ única salida ↓                         │  │
│  │  external_functions  (ALLOWLIST cerrada,       │  │
│  │  código nuestro auditado: ask/llm_query,       │  │
│  │  métodos de Context sobre texto en memoria)    │  │
│  └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Mensaje para seguridad:** el código no confiable (el del LLM) nunca se ejecuta
como Python en nuestro proceso; corre en un intérprete Rust aislado sin sistema
de ficheros, sin red, sin imports y sin shell. Lo único que toca el mundo
exterior es una lista explícita y auditable de funciones que nosotros decidimos
exponer, y cuyo alcance hoy es: llamar al propio LLM y leer texto ya cargado en
memoria. La vía de control de seguridad, por tanto, es **revisar y restringir esa
allowlist** — no "confiar en lo que escriba el modelo".

