## v0.4.1 (2026-05-05)

### Feat

- **examples**: add demo for counting 'r' in generated fruit names and interactive trace viewer
- **trace-viewer**: add interactive TUI for visualizing execution traces

### Fix

- **azure-openai, generic-chat**: update temperature parameter to be optional and improve payload handling
- **azure-openai**: enhance check_azure_connection to support multiple models and improve error handling
- **azure-openai**: handle unsupported temperature parameter and retry with default value
- **tests**: skip trace viewer tests if optional dependencies are not installed
- **rlm**: enhance prompt summary truncation and add root query support
- **ci**: push tag explicitly in release workflow

## v0.4.0 (2026-04-08)

### Feat

- **doctools**: Implement document access tools and policies
- extend retrieval with page-level ops, REPL extensions, diagnostics, and hardening
- add retrieval functionality to RLM with Elasticsearch support
- Update rich REPL demo for Azure GPT-5.1 and enhance environment variable handling
- Add AzureOpenAIAdapter and connectivity check for Azure environment variables
- Implemented llm_batch for parallel processing of prompts and enhance traceability

### Fix

- enhance RLM to validate synchronous retriever methods and update code pattern matching
- enable TLS certificate verification for ElasticsearchRetriever and update AsyncElasticsearchRetriever documentation
- enhance search functions to support deduplication and improve prompt handling
- code review fixes for generic_chat, env, context, retrieval
- pin pydantic-monty<0.0.8 and skip tests for unimplemented functions
- updated agents.md
- pass log_truncate_code into _run_recursive_subcall to fix F821 lint errors
- multiple corrections

### Perf

- Added Oolong benchmark script for baseline vs RLM evaluation

## v0.3.0 (2026-03-09)

### Feat

- add live rich trace functionality and event listener for RLM execution
- enhance REPL environment and prompts, add diagnostic run example
- add Vertex AI integration and testing framework
- enhanced RLM functionality and improve prompt handling
- add GitHub Actions workflow for automated release process
- add multi-turn conversation history to RLM loop (#13)
- add .claude to .gitignore

## v0.2.0 (2026-02-15)

### Feat

- integrate pydantic-monty as secure REPL backend (#9)
- improve adapters (#8)
- added dotenv and removed unnecessary examples
- added ai skills

### Refactor

- published package to pip and renaming the package in order to publish on pip (#4)

## v0.1.0 (2026-01-21)

### Feat

- add result export functionality and enhance README with usage details
- add demo for RLM vs baseline comparison and enhance benchmark documentation
- enhance rlm_vs_baseline benchmark for MIT RLM paper demo
- Add SmartRouter for automatic baseline/RLM selection
- enhance Context and RLM functionality with document support and recursive subcalls
- Implement core functionality for Recursive Language Model (RLM)
