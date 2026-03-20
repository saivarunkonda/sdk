# Kubeflow MCP Server: Usage, Examples, and API Reference

## Overview
The Kubeflow MCP Server provides a protocol-neutral tool registry for agentic workflows, exposing Kubeflow Trainer APIs as modular MCP tools. It supports persona-based filtering, modular client loading, and aligns with KEP-936 (kubeflow/community#937).

## Usage
Instantiate the MCP server with optional persona and client arguments:

```python
from kubeflow.trainer.mcp_server.mcp_server import TrainerMCPServer
from kubeflow.trainer.api.trainer_client import TrainerClient

trainer_client = TrainerClient()
mcp_server = TrainerMCPServer(trainer_client=trainer_client, persona="ml-engineer", clients=["trainer"])
```

List available tools for the current persona:

```python
print(mcp_server.list_tools())
```

Call a tool:

```python
result = mcp_server.call_tool("list_training_jobs", {"runtime_name": "torch-distributed"})
print(result)
```

## Examples
Suspend a training job:

```python
response = mcp_server.call_tool("suspend_training_job", {"name": "example-trainjob"})
print(response)
```

Get training job logs:

```python
response = mcp_server.call_tool("get_training_job_logs", {"name": "example-trainjob"})
print(response["result"]["logs"])
```

## API Reference

### TrainerMCPServer Methods
- `list_tools()`
- `call_tool(tool_name, arguments)`
- `list_training_jobs(runtime_name=None)`
- `get_training_job(name)`
- `delete_training_job(name)`
- `check_prerequisites(runtime_name=None)`
- `get_training_job_status(name)`
- `fine_tune(model_name, dataset_path, epochs=3, learning_rate=1e-4)`
- `run_custom_training(func_code, requirements=None)`
- `suspend_training_job(name)`
- `resume_training_job(name)`
- `checkpoint_training_job(name)`
- `restart_training_job(name)`
- `get_training_job_logs(name)`
- `get_training_job_metrics(name)`

### Arguments
- `persona`: Controls tool access (readonly, data-scientist, ml-engineer, platform-admin)
- `clients`: Enables tool groups (e.g., ["trainer", "optimizer"])

## Real Implementation Guidance
Replace stub methods with actual logic using Kubeflow SDK and TrainerClient APIs. For example:

```python
def suspend_training_job(self, name: str) -> dict[str, Any]:
    self._trainer_client.suspend_job(name=name)
    return {"suspended": name}
```

## Integration
To integrate with other Kubeflow clients (optimizer, hub), add their tool groups to `_CLIENT_TOOLS` and implement corresponding methods.

## Proposal and GSoC Comments
- Reference KEP-936 and kubeflow/community#937 in all documentation and comments.
- Document modular client loading and persona filtering in docstrings.

## Final Review
- Ensure all methods are documented.
- Remove all TODOs and replace with real implementations.
- Validate with lint and tests.
- Update README and NON_GOALS.md as needed.

---
For more details, see [KEP-936](https://github.com/kubeflow/community/pull/937).
