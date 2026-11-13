# MCP server

`pyrlm_runtime.mcp_server` publishes the library's retrieval as tools of an
[MCP](https://modelcontextprotocol.io) server. Any MCP client (Claude Code,
Claude Desktop, Cursor, kagent, your own agent) can then search and read your
index without writing Python.

This release publishes the read-only retrieval tools `search`, `get_document`
and `get_pages`. None of them calls an LLM. The design and the planned tools
(`deep_search`, `rlm_run`) are in [RFC-0001](MCP-SERVER-RFC.md).

## Install

```bash
pip install "pyrlm-runtime[mcp]"                 # custom retriever
pip install "pyrlm-runtime[mcp,elasticsearch]"   # built-in Elasticsearch retriever
```

Importing `pyrlm_runtime` does not import the server or the `mcp` package.
Importing `pyrlm_runtime.mcp_server` without the extra raises an `ImportError`
that names the install command.

## Run

The command is `pyrlm-mcp`, or `python -m pyrlm_runtime.mcp_server`. It reads
its configuration from `PYRLM_MCP_*` environment variables. It does not load a
`.env` file unless you pass `--env-file PATH`.

### stdio (default)

The client starts the server as a subprocess and talks to it over stdin and
stdout. Register it in Claude Code:

```bash
claude mcp add pyrlm \
  -e PYRLM_MCP_ES_HOST=https://my-cluster:9200 \
  -e PYRLM_MCP_ES_INDEX=docs \
  -e PYRLM_MCP_ES_API_KEY=... \
  -- pyrlm-mcp
```

Clients that take a JSON configuration use the same command and environment:

```json
{
  "mcpServers": {
    "pyrlm": {
      "command": "pyrlm-mcp",
      "env": {
        "PYRLM_MCP_ES_HOST": "https://my-cluster:9200",
        "PYRLM_MCP_ES_INDEX": "docs",
        "PYRLM_MCP_ES_API_KEY": "..."
      }
    }
  }
}
```

### Streamable HTTP

Set `PYRLM_MCP_TRANSPORT=streamable-http`. The server listens on
`PYRLM_MCP_HOST:PYRLM_MCP_PORT` (default `127.0.0.1:8000`), serves MCP at
`/mcp` and answers `GET /health` with `{"status": "ok"}`.

```bash
PYRLM_MCP_TRANSPORT=streamable-http \
PYRLM_MCP_BEARER_TOKEN=change-me \
PYRLM_MCP_ES_HOST=https://my-cluster:9200 PYRLM_MCP_ES_INDEX=docs \
pyrlm-mcp
```

Clients send `Authorization: Bearer change-me`. To serve on a non-loopback
address, read [Security](#security) first: the server refuses to start without
a token and a Host allowlist.

### Try it without Elasticsearch

[`examples/mcp_server_demo.py`](../examples/mcp_server_demo.py) starts the
server as a subprocess with an in-memory retriever, connects a real MCP client,
lists the tools and calls each one:

```bash
uv run python examples/mcp_server_demo.py stdio
uv run python examples/mcp_server_demo.py http
```

## Configuration

Every variable has the prefix `PYRLM_MCP_`. Lists are comma-separated. A list
set to an empty value is an empty list, which differs from leaving it unset.

| Variable | Default | Meaning |
|---|---|---|
| `TRANSPORT` | `stdio` | `stdio` or `streamable-http` |
| `HOST` | `127.0.0.1` | HTTP bind address |
| `PORT` | `8000` | HTTP port |
| `ALLOWED_HOSTS` | none | Extra `Host` header values to accept, such as `mcp.example.com` or `mcp.example.com:*`. Required on a non-loopback host |
| `ALLOWED_ORIGINS` | none | Extra `Origin` header values to accept |
| `BEARER_TOKEN` | none | Token clients must send. Secret |
| `ALLOW_UNAUTHENTICATED` | `0` | `1` lets a non-loopback server start without a token, for a gateway that authenticates clients |
| `LOG_LEVEL` | `INFO` | Log level. Logs go to stderr |
| `NAME` | `pyrlm-runtime` | Server name reported to clients |
| `ES_HOST`, `ES_INDEX` | none | Elasticsearch cluster URL and index. Setting `ES_HOST` selects the built-in retriever |
| `ES_API_KEY` | none | Elasticsearch API key. Secret |
| `ES_CONTENT_FIELD`, `ES_VECTOR_FIELD` | `content`, `embedding` | Index field names |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_BASE_URL` | `openai`, none, OpenAI | Query embeddings for `semantic` and `hybrid`. The OpenAI provider reads `OPENAI_API_KEY` |
| `RETRIEVER_FACTORY` | none | `pkg.module:callable` that returns a `RetrieverProtocol` object. Called once, with no arguments. Exclusive with `ES_HOST` |
| `SEARCH_MODES` | `hybrid,lexical,semantic` | Modes `search` accepts. The first is the default. With `ES_HOST` and no `EMBEDDING_MODEL`, the default is `lexical` only |
| `MAX_TOP_K` | `50` | Upper bound for `top_k` |
| `MAX_CONTENT_CHARS` | `4000` | Per-hit `content` length with `include_content=true` |
| `MAX_DOCUMENT_CHARS` | `20000` | Slice length of `get_document`, and per-page length of `get_pages` |
| `MAX_RESPONSE_CHARS` | `60000` | Size cap of one `search` or `get_pages` response |
| `MAX_PAGES` | `20` | Pages per `get_pages` call |
| `METADATA_DENYLIST` | none | Metadata fields removed from every result and refused as filter keys. A dotted entry such as `owner.tax_id` removes a nested field |
| `FILTERABLE_FIELDS` | unset: any key | The only keys `filters` accepts. Set but empty: no key |
| `BASE_FILTERS` | none | JSON object merged into every search. The server's value wins on a shared key |

The server reads no other secret. The built-in retriever and the embedding
provider still read the credentials they already read, such as
`OPENAI_API_KEY` and Google application default credentials.

The command line builds the retriever at start-up. A missing `elasticsearch`
package, a factory that does not load, or an invalid variable stops the
process with exit code 2 and a message on stderr.

## Programmatic use

```python
from pyrlm_runtime.mcp_server import ServerConfig, build_asgi_app, build_server, run_server

config = ServerConfig(search_modes=("hybrid", "lexical"), metadata_denylist=("ssn",))
server = build_server(retriever=my_retriever, config=config)

run_server(server, config)                                   # stdio or HTTP, per config
app = build_asgi_app(server, config, bearer_token="change-me")  # ASGI app for your own server
```

`ServerConfig` holds no secret and serialises to JSON with
`config.model_dump_json()`. `ServerConfig.from_env()` and `Secrets.from_env()`
build the two objects from the environment.

`build_asgi_app` and `run_server` apply the security checks. Calling the SDK's
`server.run()` or `server.streamable_http_app()` directly skips them and is not
supported.

To mount the app inside another Starlette or FastAPI app, run the server's
session manager in your app's lifespan, as the MCP SDK documents for mounted
servers:

```python
import contextlib

from starlette.applications import Starlette
from starlette.routing import Mount

mcp_app = build_asgi_app(server, config, bearer_token=token)


@contextlib.asynccontextmanager
async def lifespan(app):
    async with server.session_manager.run():
        yield


app = Starlette(routes=[Mount("/rag", app=mcp_app)], lifespan=lifespan)
```

## Tools

The server publishes a tool only when it can serve it. `search` and
`get_document` need a retriever. `get_pages` also needs the retriever to have a
`get_pages` method, which `ElasticsearchRetriever` has. With no retriever the
server refuses to start. The server's `instructions` mention only the published
tools.

Every tool is annotated read-only and idempotent, and returns structured
content that matches its published output schema. Numeric arguments above a
limit are clamped, and the response reports the value used. A failure is a tool
error (`isError: true`), never an empty success.

### `search`

| Argument | Default | Meaning |
|---|---|---|
| `query` | required | Search text. Must not be blank |
| `mode` | first of `SEARCH_MODES` | `hybrid`, `lexical` or `semantic`. A mode the server does not offer is a tool error that lists the offered modes |
| `top_k` | `10` | Hits to return, clamped to `[1, MAX_TOP_K]` |
| `filters` | none | Metadata filters, in the filter syntax of `RetrieverProtocol` |
| `include_content` | `false` | Add each hit's text, truncated to `MAX_CONTENT_CHARS` |

Returns `results` (each hit has `doc_id`, `logical_doc_id`, `page_num`,
`score`, `preview`, `content`, `content_truncated` and `metadata`), `mode`,
`top_k_applied` and `truncated`. `truncated` is true when hits were dropped from
the tail to fit `MAX_RESPONSE_CHARS`.

`doc_id` identifies one indexed unit, such as a page. `logical_doc_id`
identifies the document the unit belongs to.

### `get_document`

| Argument | Default | Meaning |
|---|---|---|
| `doc_id` | required | `doc_id` of a hit |
| `offset` | `0` | First character of the slice |
| `max_chars` | `MAX_DOCUMENT_CHARS` | Slice length, clamped to `[1, MAX_DOCUMENT_CHARS]` |

Returns `content`, `offset`, `max_chars_applied`, `next_offset`, `total_chars`
and `metadata`. When `next_offset` is not null, call again with that offset to
read the rest. An unknown id is a tool error.

### `get_pages`

| Argument | Default | Meaning |
|---|---|---|
| `logical_doc_id` | required | `logical_doc_id` of a hit. A hit's `doc_id` finds nothing |
| `pages` | required | Page numbers to read, at most `MAX_PAGES` of them |
| `radius` | `0` | Neighbouring pages to add on each side, clamped to `[0, MAX_PAGES]` |

Returns `pages` (each with `doc_id`, `page_num`, `content`,
`content_truncated` and `metadata`), `radius_applied`, `max_pages_applied` and
`truncated`. When no page matches, the tool error reminds the client to pass the
`logical_doc_id`.

### Retriever contract

A custom retriever implements `RetrieverProtocol`. Search hits are dicts with
`doc_id`, `content`, `preview`, `score` and `metadata`, plus optional
`logical_doc_id` and `page_num`. `get` raises `KeyError` (or any `LookupError`)
for an unknown id, so the server can report "not found" instead of a backend
failure. `get_pages(logical_doc_id, *, pages, radius, max_pages)` is optional.

## Security

**Network exposure.** The default transport is stdio, which opens no port. Over
HTTP on a loopback host (`127.0.0.1`, `::1`, `localhost`) the server accepts
only loopback `Host` and `Origin` headers. On any other host, including
`0.0.0.0`, `::` and hostnames, the server refuses to start unless
`BEARER_TOKEN` is set (or `ALLOW_UNAUTHENTICATED=1`) and `ALLOWED_HOSTS` lists
the Host values clients use. Requests with another Host get HTTP 421. A server
bound to loopback behind a reverse proxy is exposed too: when `ALLOWED_HOSTS`
names a non-loopback host, the server also requires `BEARER_TOKEN` or
`ALLOW_UNAUTHENTICATED=1`.

**Bearer token.** The scheme is case-insensitive and the token is compared in
constant time. A request without it
gets HTTP 401 with `WWW-Authenticate: Bearer`. `/health` needs no token.
`/.well-known/*` answers 404, so OAuth-capable clients do not start an OAuth
flow. The token is a single shared secret. For OAuth, per-user identity, rate
limits or quotas, put a gateway in front of the server.

**What clients can read.** Search hits carry a preview by default; full text
needs `include_content=true` and is truncated. `search` and `get_pages`
responses have a size cap; `get_document` returns at most `MAX_DOCUMENT_CHARS`
of text plus the unit's metadata. `METADATA_DENYLIST` removes fields from
results and refuses them as filter keys, because a filter on a hidden field
reveals its value through which hits come back. A denied field's subfields
(such as Elasticsearch's `ssn.keyword`) and its parent object are refused as
well. `FILTERABLE_FIELDS` narrows the filter keys further.

**`BASE_FILTERS` is not an access boundary.** It scopes every search, but reads
by id (`get_document`, `get_pages`) skip it. To separate tenants, use an
Elasticsearch API key with document-level security, or one index per tenant.

**Error messages.** A backend failure reaches the client as the exception type
only, such as `the retrieval backend failed (ConnectionError)`. The full error,
which can name hosts or indexes, goes to the server log on stderr.

**Untrusted text.** Previews and content come from your documents. A document
can contain text written to steer the client's model. The server cannot remove
it. The server instructions tell clients to treat returned text as data, and
the integrator must account for this risk, for example when the client agent
also holds tools that change systems.

**Configuration is trusted.** `RETRIEVER_FACTORY` imports and runs the code it
names. Treat the server's environment like code, for example when it comes from
a Kubernetes ConfigMap.

**stdout belongs to the protocol.** On stdio, any write to stdout corrupts the
MCP stream. The server logs to stderr. Never run a `PythonREPL` in the server
process: it redirects `sys.stdout` for the whole process. The server does not
expose `embedding_ssl_verify`, which patches `requests` for the whole process.

## Limitations

- When the first hit alone exceeds `MAX_RESPONSE_CHARS`, which takes very large
  metadata, `search` returns no hits and `truncated=true`. Raise the cap or
  deny the large metadata fields.

- `get_document` reads one indexed unit. It does not stitch a whole logical
  document; read those with `get_pages`.
- The built-in Elasticsearch client is not closed on shutdown, because
  `ElasticsearchRetriever` has no public `close()`. A custom retriever with a
  `close()` method is closed when the command exits.
- Each tool result carries its data twice: as structured content and, for
  clients that do not read structured content, as JSON text. This is how the
  MCP SDK builds results.
