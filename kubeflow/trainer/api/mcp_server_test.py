from datetime import datetime
from unittest.mock import Mock

from kubeflow.trainer.api.mcp_server import TrainerMCPServer
from kubeflow.trainer.types import types


def _get_runtime() -> types.Runtime:
    runtime_trainer = types.RuntimeTrainer(
        trainer_type=types.TrainerType.CUSTOM_TRAINER,
        framework="pytorch",
        image="pytorch/pytorch:2.6.0",
        num_nodes=1,
    )
    runtime_trainer.set_command(("python", "train.py"))

    return types.Runtime(name="torch-distributed", trainer=runtime_trainer)


def _get_train_job() -> types.TrainJob:
    runtime = _get_runtime()
    steps = [
        types.Step(
            name="node-0",
            status="Running",
            pod_name="trainjob-node-0",
            device="gpu",
            device_count="1",
        )
    ]

    return types.TrainJob(
        name="example-trainjob",
        runtime=runtime,
        steps=steps,
        num_nodes=1,
        creation_timestamp=datetime(2026, 3, 20, 10, 0, 0),
        status="Running",
    )


def test_list_tools_returns_supported_tools():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)

    assert mcp_server.list_tools() == ["list_training_jobs", "get_training_job", "delete_training_job"]
def test_call_tool_delete_training_job_success_response_envelope():
    trainer_client = Mock()
    trainer_client.delete_job.return_value = None
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("delete_training_job", {"name": "example-trainjob"})
    trainer_client.delete_job.assert_called_once_with(name="example-trainjob")
    assert response["ok"] is True
    assert response["result"] == {"deleted": "example-trainjob"}

def test_call_tool_delete_training_job_missing_name_argument():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("delete_training_job", {})
    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"


def test_list_training_jobs_without_runtime_filter():
    trainer_client = Mock()
    trainer_client.list_jobs.return_value = [_get_train_job()]

    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    result = mcp_server.list_training_jobs()

    trainer_client.list_jobs.assert_called_once_with(runtime=None)
    assert result["count"] == 1
    assert result["jobs"][0]["name"] == "example-trainjob"
    assert result["jobs"][0]["creation_timestamp"] == "2026-03-20T10:00:00"
    assert result["jobs"][0]["runtime"]["trainer"]["trainer_type"] == types.TrainerType.CUSTOM_TRAINER.value


def test_list_training_jobs_with_runtime_filter():
    trainer_client = Mock()
    runtime = _get_runtime()
    trainer_client.get_runtime.return_value = runtime
    trainer_client.list_jobs.return_value = []

    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    result = mcp_server.list_training_jobs(runtime_name="torch-distributed")

    trainer_client.get_runtime.assert_called_once_with("torch-distributed")
    trainer_client.list_jobs.assert_called_once_with(runtime=runtime)
    assert result == {"jobs": [], "count": 0}


def test_call_tool_success_response_envelope():
    trainer_client = Mock()
    trainer_client.get_runtime.return_value = _get_runtime()
    trainer_client.list_jobs.return_value = [_get_train_job()]

    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("list_training_jobs", {"runtime_name": "torch-distributed"})

    assert response["ok"] is True
    assert response["result"]["count"] == 1


def test_call_tool_get_training_job_success_response_envelope():
    trainer_client = Mock()
    train_job = _get_train_job()
    trainer_client.get_job.return_value = train_job

    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("get_training_job", {"name": "example-trainjob"})

    trainer_client.get_job.assert_called_once_with(name="example-trainjob")
    assert response["ok"] is True
    assert response["result"]["name"] == "example-trainjob"


def test_call_tool_get_training_job_missing_name_argument():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)

    response = mcp_server.call_tool("get_training_job", {})

    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"


def test_call_tool_returns_error_for_unknown_tool():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)

    response = mcp_server.call_tool("unknown_tool", {})

    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"


def test_call_tool_returns_error_for_invalid_arguments_type():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)

    response = mcp_server.call_tool("list_training_jobs", arguments="invalid")

    assert response["ok"] is False
    assert response["error"]["type"] == "TypeError"
