from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.types import types


class TrainerMCPServer:
    """MCP-oriented tool wrapper for Kubeflow Trainer APIs.

    This class provides a protocol-neutral scaffold for exposing Trainer APIs as
    MCP tools, aligned with [KEP-936](https://github.com/kubeflow/community/pull/937).
    It implements modular client loading (via `--clients` argument) and persona-based tool filtering,
    as described in the official Kubeflow MCP proposal.

    References:
        - KEP-936: https://github.com/kubeflow/community/pull/937
        - Proposal: kubeflow/community#937

    Features:
        - Modular client loading: tools are grouped by client (e.g., trainer, optimizer, hub).
        - Persona filtering: tool access is filtered by persona (readonly, data-scientist, ml-engineer, platform-admin).
        - Tool registry: tool names and signatures are aligned with the proposal.
        - Protocol-neutral: no transport layer, agentic tool registry only.
    """

    def __init__(self, trainer_client: TrainerClient | None = None, persona: str = "ml-engineer", clients: list[str] = None):
        """Initialize the MCP server wrapper.

        Args:
            trainer_client: Optional ``TrainerClient`` instance. If not provided,
                a default ``TrainerClient`` instance is created.
            persona: Persona string (readonly, data-scientist, ml-engineer, platform-admin).
            clients: List of enabled clients (e.g., ["trainer", "optimizer"]).
        """
        self._trainer_client = trainer_client or TrainerClient()
        self._persona = persona
        self._clients = set(clients) if clients else {"trainer"}

    # Modular tool registry: tools grouped by client
    _CLIENT_TOOLS = {
        "trainer": [
            "list_training_jobs",
            "get_training_job",
            "get_training_job_status",
            "check_prerequisites",
            "delete_training_job",
            "fine_tune",
            "run_custom_training",
            "suspend_training_job",
            "resume_training_job",
            "checkpoint_training_job",
            "restart_training_job",
            "get_training_job_logs",
            "get_training_job_metrics",
        ],
        # "optimizer": [ ... ],
        # "hub": [ ... ],
    }

    # Persona tool map based on KEP-936 (tools must also be enabled by client)
    _PERSONA_TOOLS = {
        "readonly": [
            "list_training_jobs",
            "get_training_job",
            "get_training_job_status",
            "check_prerequisites",
        ],
        "data-scientist": [
            "list_training_jobs",
            "get_training_job",
            "get_training_job_status",
            "check_prerequisites",
            "fine_tune",
            "run_custom_training",
            "get_training_job_logs",
            "get_training_job_metrics",
        ],
        "ml-engineer": [
            "list_training_jobs",
            "get_training_job",
            "get_training_job_status",
            "check_prerequisites",
            "delete_training_job",
            "fine_tune",
            "run_custom_training",
            "suspend_training_job",
            "resume_training_job",
            "checkpoint_training_job",
            "restart_training_job",
            "get_training_job_logs",
            "get_training_job_metrics",
        ],
        "platform-admin": [
            "list_training_jobs",
            "get_training_job",
            "get_training_job_status",
            "check_prerequisites",
            "delete_training_job",
            "fine_tune",
            "run_custom_training",
            "suspend_training_job",
            "resume_training_job",
            "checkpoint_training_job",
            "restart_training_job",
            "get_training_job_logs",
            "get_training_job_metrics",
        ],
    }
    def suspend_training_job(self, name: str) -> dict[str, Any]:
        """Stub for suspending a training job."""
        # TODO: Implement actual suspend logic
        return {"suspended": name}

    def resume_training_job(self, name: str) -> dict[str, Any]:
        """Stub for resuming a training job."""
        # TODO: Implement actual resume logic
        return {"resumed": name}

    def checkpoint_training_job(self, name: str) -> dict[str, Any]:
        """Stub for checkpointing a training job."""
        # TODO: Implement actual checkpoint logic
        return {"checkpointed": name}

    def restart_training_job(self, name: str) -> dict[str, Any]:
        """Stub for restarting a training job."""
        # TODO: Implement actual restart logic
        return {"restarted": name}

    def get_training_job_logs(self, name: str) -> dict[str, Any]:
        """Stub for retrieving training job logs."""
        # TODO: Implement actual log retrieval
        return {"logs": f"Logs for job {name} (Stub)"}

    def get_training_job_metrics(self, name: str) -> dict[str, Any]:
        """Stub for retrieving training job metrics."""
        # TODO: Implement actual metrics retrieval
        return {"metrics": {"accuracy": 0.95, "loss": 0.1}}
    def fine_tune(self, model_name: str, dataset_path: str, epochs: int = 3, learning_rate: float = 1e-4) -> dict[str, Any]:
        """Stub for fine-tuning a model. Replace with actual SDK logic."""
        # TODO: Implement using TrainerClient fine-tune API
        return {
            "message": f"Fine-tuning model '{model_name}' on dataset '{dataset_path}' for {epochs} epochs at lr={learning_rate}. (Stub)"
        }

    def run_custom_training(self, func_code: str, requirements: list[str] = None) -> dict[str, Any]:
        """Stub for running custom training code. Replace with secure execution logic per KEP-936."""
        # TODO: Implement secure code execution, AST checks, temp file handling
        return {
            "message": "Custom training job submitted (Stub)."
        }

    def list_tools(self) -> list[str]:
        """List available Trainer MCP tools for the current persona and enabled clients, preserving test order."""
        # Use the order from _CLIENT_TOOLS["trainer"] as canonical
        persona_tools = set(self._PERSONA_TOOLS.get(self._persona, self._PERSONA_TOOLS["ml-engineer"]))
        client_tools = []
        for client in self._clients:
            client_tools.extend(self._CLIENT_TOOLS.get(client, []))
        # Preserve order as in client_tools, but only include those allowed by persona
        seen = set()
        ordered_tools = []
        for tool in client_tools:
            if tool in persona_tools and tool not in seen:
                ordered_tools.append(tool)
                seen.add(tool)
        return ordered_tools

    def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke a supported Trainer MCP tool by name.

        Args:
            tool_name: Tool name to invoke.
            arguments: Optional dictionary of tool arguments.

        Returns:
            A structured response envelope containing either ``result`` on
            success or ``error`` details on failure.
        """
        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "error": {
                    "type": "TypeError",
                    "message": "Tool arguments must be a dictionary.",
                },
            }

        try:
            allowed_tools = self.list_tools()
            if tool_name not in allowed_tools:
                raise PermissionError(f"Persona '{self._persona}' is not allowed to use tool '{tool_name}'.")

            if tool_name == "list_training_jobs":
                runtime_name = arguments.get("runtime_name")
                result = self.list_training_jobs(runtime_name=runtime_name)
                return {"ok": True, "result": result}

            if tool_name == "get_training_job":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'get_training_job' requires argument 'name'.")
                result = self.get_training_job(name=name)
                return {"ok": True, "result": result}

            if tool_name == "delete_training_job":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'delete_training_job' requires argument 'name'.")
                result = self.delete_training_job(name=name)
                return {"ok": True, "result": result}

            if tool_name == "check_prerequisites":
                runtime_name = arguments.get("runtime_name")
                result = self.check_prerequisites(runtime_name=runtime_name)
                return {"ok": True, "result": result}

            if tool_name == "get_training_job_status":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'get_training_job_status' requires argument 'name'.")
                result = self.get_training_job_status(name=name)
                return {"ok": True, "result": result}

            if tool_name == "fine_tune":
                model_name = arguments.get("model_name")
                dataset_path = arguments.get("dataset_path")
                epochs = arguments.get("epochs", 3)
                learning_rate = arguments.get("learning_rate", 1e-4)
                if not model_name or not dataset_path:
                    raise ValueError("Tool 'fine_tune' requires arguments 'model_name' and 'dataset_path'.")
                result = self.fine_tune(model_name, dataset_path, epochs, learning_rate)
                return {"ok": True, "result": result}

            if tool_name == "run_custom_training":
                func_code = arguments.get("func_code")
                requirements = arguments.get("requirements")
                if not func_code:
                    raise ValueError("Tool 'run_custom_training' requires argument 'func_code'.")
                result = self.run_custom_training(func_code, requirements)
                return {"ok": True, "result": result}

            if tool_name == "suspend_training_job":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'suspend_training_job' requires argument 'name'.")
                result = self.suspend_training_job(name=name)
                return {"ok": True, "result": result}

            if tool_name == "resume_training_job":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'resume_training_job' requires argument 'name'.")
                result = self.resume_training_job(name=name)
                return {"ok": True, "result": result}

            if tool_name == "checkpoint_training_job":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'checkpoint_training_job' requires argument 'name'.")
                result = self.checkpoint_training_job(name=name)
                return {"ok": True, "result": result}

            if tool_name == "restart_training_job":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'restart_training_job' requires argument 'name'.")
                result = self.restart_training_job(name=name)
                return {"ok": True, "result": result}

            if tool_name == "get_training_job_logs":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'get_training_job_logs' requires argument 'name'.")
                result = self.get_training_job_logs(name=name)
                return {"ok": True, "result": result}

            if tool_name == "get_training_job_metrics":
                name = arguments.get("name")
                if not name:
                    raise ValueError("Tool 'get_training_job_metrics' requires argument 'name'.")
                result = self.get_training_job_metrics(name=name)
                return {"ok": True, "result": result}

            raise ValueError(f"Unsupported tool '{tool_name}'")
        except Exception as error:
            return {
                "ok": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }

    def get_training_job_status(self, name: str) -> dict[str, Any]:
        """Observability tool: get the status and progress of a training job."""
        train_job = self._trainer_client.get_job(name=name)
        # Minimal status/progress info; can be expanded as needed
        return {
            "name": train_job.name,
            "status": getattr(train_job, "status", None),
            "progress": getattr(train_job, "progress", None),
            "last_update": getattr(train_job, "last_update", None),
        }

    def check_prerequisites(self, runtime_name: str | None = None) -> dict[str, Any]:
        """Pre-flight validation: check runtime availability and basic cluster connectivity."""
        result = {}
        # Check runtime availability
        runtimes = self._trainer_client.list_runtimes()
        result["available_runtimes"] = [r.name for r in runtimes]
        if runtime_name:
            result["runtime_exists"] = runtime_name in result["available_runtimes"]
        # TODO: Add more checks (GPU, storage, etc.) as needed
        result["cluster_connectivity"] = True  # Assume True if runtimes can be listed
        return result
    def delete_training_job(self, name: str) -> dict[str, Any]:
        """Delete a training job by name using ``TrainerClient``.

        Args:
            name: TrainJob name.

        Returns:
            Dictionary indicating deletion success.
        """
        self._trainer_client.delete_job(name=name)
        return {"deleted": name}

    def list_training_jobs(self, runtime_name: str | None = None) -> dict[str, Any]:
        """List training jobs using ``TrainerClient``.

        Args:
            runtime_name: Optional runtime name used to filter jobs.

        Returns:
            A dictionary with serialized TrainJob entries and a total count.
        """
        runtime: types.Runtime | None = None
        if runtime_name is not None:
            runtime = self._trainer_client.get_runtime(runtime_name)

        jobs = self._trainer_client.list_jobs(runtime=runtime)
        serialized_jobs = [self._serialize_train_job(job) for job in jobs]

        return {
            "jobs": serialized_jobs,
            "count": len(serialized_jobs),
        }

    def get_training_job(self, name: str) -> dict[str, Any]:
        """Get a single training job by name using ``TrainerClient``.

        Args:
            name: TrainJob name.

        Returns:
            Serialized TrainJob dictionary.
        """
        train_job = self._trainer_client.get_job(name=name)
        return self._serialize_train_job(train_job)

    def _serialize_train_job(self, train_job: types.TrainJob) -> dict[str, Any]:
        """Serialize a ``TrainJob`` to a JSON-compatible dictionary."""
        return self._serialize_value(asdict(train_job))

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serialize values to JSON-compatible structures."""
        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return self._serialize_value(asdict(value))

        if isinstance(value, dict):
            return {key: self._serialize_value(item) for key, item in value.items()}

        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]

        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]

        return value
