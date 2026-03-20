def test_call_tool_suspend_training_job():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("suspend_training_job", {"name": "example-trainjob"})
    assert response["ok"] is True
    assert response["result"] == {"suspended": "example-trainjob"}

def test_call_tool_resume_training_job():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("resume_training_job", {"name": "example-trainjob"})
    assert response["ok"] is True
    assert response["result"] == {"resumed": "example-trainjob"}

def test_call_tool_checkpoint_training_job():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("checkpoint_training_job", {"name": "example-trainjob"})
    assert response["ok"] is True
    assert response["result"] == {"checkpointed": "example-trainjob"}

def test_call_tool_restart_training_job():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("restart_training_job", {"name": "example-trainjob"})
    assert response["ok"] is True
    assert response["result"] == {"restarted": "example-trainjob"}

def test_call_tool_get_training_job_logs():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("get_training_job_logs", {"name": "example-trainjob"})
    assert response["ok"] is True
    assert "Logs for job example-trainjob" in response["result"]["logs"]

def test_call_tool_get_training_job_metrics():
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("get_training_job_metrics", {"name": "example-trainjob"})
    assert response["ok"] is True
    metrics = response["result"]["metrics"]
    assert "accuracy" in metrics
    assert "loss" in metrics
from datetime import datetime
import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from kubeflow.trainer.mcp_server.mcp_server import TrainerMCPServer
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
    mcp_server = TrainerMCPServer(trainer_client=trainer_client, persona="readonly")
    assert mcp_server.list_tools() == [
        "list_training_jobs",
        "get_training_job",
        "get_training_job_status",
        "check_prerequisites",
    ]

def test_call_tool_get_training_job_status_returns_status_and_progress():
    trainer_client = Mock()
    mock_job = Mock()
    mock_job.name = "example-trainjob"
    mock_job.status = "Running"
    mock_job.progress = 0.5
    mock_job.last_update = datetime(2026, 3, 21, 12, 0, 0)
    trainer_client.get_job.return_value = mock_job
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("get_training_job_status", {"name": "example-trainjob"})
    assert response["ok"] is True
    result = response["result"]
    assert result["name"] == "example-trainjob"
    assert result["status"] == "Running"
    assert result["progress"] == 0.5
    assert result["last_update"] == datetime(2026, 3, 21, 12, 0, 0)

def test_call_tool_check_prerequisites_lists_runtimes_and_connectivity():
    trainer_client = Mock()
    torch_runtime = Mock()
    torch_runtime.name = "torch-distributed"
    tf_runtime = Mock()
    tf_runtime.name = "tf-job"
    trainer_client.list_runtimes.return_value = [torch_runtime, tf_runtime]
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("check_prerequisites", {})
    assert response["ok"] is True
    result = response["result"]
    assert "available_runtimes" in result
    assert set(result["available_runtimes"]) == {"torch-distributed", "tf-job"}
    assert result["cluster_connectivity"] is True

def test_call_tool_check_prerequisites_with_runtime_name():
    trainer_client = Mock()
    torch_runtime = Mock()
    torch_runtime.name = "torch-distributed"
    tf_runtime = Mock()
    tf_runtime.name = "tf-job"
    trainer_client.list_runtimes.return_value = [torch_runtime, tf_runtime]
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("check_prerequisites", {"runtime_name": "torch-distributed"})
    assert response["ok"] is True
    result = response["result"]
    assert result["runtime_exists"] is True
    response2 = mcp_server.call_tool("check_prerequisites", {"runtime_name": "not-a-runtime"})
    assert response2["ok"] is True
    assert response2["result"]["runtime_exists"] is False

def test_call_tool_delete_training_job_success_response_envelope():
    trainer_client = Mock()
    trainer_client.delete_job.return_value = None
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("delete_training_job", {"name": "example-trainjob"})
    trainer_client.delete_job.assert_called_once_with(name="example-trainjob")
    assert response["ok"] is True
    assert response["result"] == {"deleted": "example-trainjob"}

def test_call_tool_delete_training_job_missing_name_argument():
    def test_call_tool_fine_tune_stub():
        trainer_client = Mock()
        mcp_server = TrainerMCPServer(trainer_client=trainer_client)
        response = mcp_server.call_tool("fine_tune", {
            "model_name": "bert-base",
            "dataset_path": "/data/train.csv",
            "epochs": 5,
            "learning_rate": 2e-5
        })
        assert response["ok"] is True
        result = response["result"]
        assert "Fine-tuning model" in result["message"]

    def test_call_tool_fine_tune_missing_args():
        trainer_client = Mock()
        mcp_server = TrainerMCPServer(trainer_client=trainer_client)
        response = mcp_server.call_tool("fine_tune", {"model_name": "bert-base"})
        assert response["ok"] is False
        assert response["error"]["type"] == "ValueError"

    def test_call_tool_run_custom_training_stub():
        trainer_client = Mock()
        mcp_server = TrainerMCPServer(trainer_client=trainer_client)
        response = mcp_server.call_tool("run_custom_training", {
            "func_code": "def train(): pass",
            "requirements": ["torch"]
        })
        assert response["ok"] is True
        result = response["result"]
        assert "Custom training job submitted" in result["message"]

    def test_call_tool_run_custom_training_missing_func_code():
        trainer_client = Mock()
        mcp_server = TrainerMCPServer(trainer_client=trainer_client)
        response = mcp_server.call_tool("run_custom_training", {})
        assert response["ok"] is False
        assert response["error"]["type"] == "ValueError"
    trainer_client = Mock()
    mcp_server = TrainerMCPServer(trainer_client=trainer_client)
    response = mcp_server.call_tool("delete_training_job", {})
    assert response["ok"] is False
    assert response["error"]["type"] == "ValueError"
