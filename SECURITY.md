# Security Policy

## Supported versions

Only the latest release of `pyrlm-runtime` receives security fixes.

## Reporting a vulnerability

Report a vulnerability privately through GitHub's
[private vulnerability reporting](https://github.com/apenab/pyrlm-runtime/security/advisories/new).
Do not open a public issue. Include the version, the configuration involved and
the steps to reproduce.

## Scope

Most of the library runs inside your own process. Two parts deserve attention:

- **The REPL.** The default `"python"` backend runs model-written code with
  `exec()` behind a whitelist. It is not a security boundary. Use the `"monty"`
  backend when the model or its inputs are untrusted.
- **The MCP server** (`pyrlm_runtime.mcp_server`, extra `mcp`). It is a network
  service when you run it over streamable HTTP. It serves read-only retrieval
  and refuses to start on a non-loopback address without a bearer token and a
  Host allowlist. Its threat model and settings are in
  [docs/MCP.md](docs/MCP.md#security). Text it returns comes from your indexed
  documents and may try to instruct the client's model; treat it as untrusted.
