from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.types import types


class TrainerMCPServer:
    """MCP-oriented tool wrapper for Kubeflow Trainer APIs.

    This class provides a protocol-neutral scaffold for exposing Trainer APIs as
    MCP tools. It starts with a first tool, ``list_training_jobs``, and can be
    expanded incrementally with additional TrainJob lifecycle methods.
    """

    def __init__(self, trainer_client: TrainerClient | None = None):
        """Initialize the MCP server wrapper.

        Args:
            trainer_client: Optional ``TrainerClient`` instance. If not provided,
                a default ``TrainerClient`` instance is created.
        """
        self._trainer_client = trainer_client or TrainerClient()

    def list_tools(self) -> list[str]:
        """List available Trainer MCP tools."""
        return ["list_training_jobs", "get_training_job", "delete_training_job"]

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

            raise ValueError(f"Unsupported tool '{tool_name}'")
        except Exception as error:
            return {
                "ok": False,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
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
