# Kubeflow MCP Server

## Overview
The Kubeflow MCP Server provides a protocol-neutral tool registry for agentic workflows, exposing Kubeflow Trainer APIs as modular MCP tools. It supports persona-based filtering, modular client loading, and aligns with KEP-936 (kubeflow/community#937).

## Features
- Modular tool registry for Kubeflow Trainer, Optimizer, and Hub clients
- Persona-based tool filtering (readonly, data-scientist, ml-engineer, platform-admin)
- Real implementations for lifecycle and observability tools
- Integration with Kubeflow backend and TrainerClient
- Usage, examples, and API reference in USAGE.md

## Usage
See USAGE.md for detailed usage, examples, and API reference.

## Development
- Replace stub methods with real logic using Kubeflow SDK and TrainerClient APIs
- Add new tool groups by updating `_CLIENT_TOOLS` and implementing methods
- Reference KEP-936 and kubeflow/community#937 in documentation

## Testing
- Run `pytest` to validate integration and functionality
- Ensure all methods are documented and tested

## Documentation
- See NON_GOALS.md for explicit non-goals
- Expand USAGE.md for examples and API reference

## Final Review
- Remove all TODOs and replace with real implementations
- Validate with lint and tests
- Update README and NON_GOALS.md as needed

---
For more details, see [KEP-936](https://github.com/kubeflow/community/pull/937).
