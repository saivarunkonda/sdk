# MCP Server Non-Goals

This document outlines the explicit non-goals for the Kubeflow MCP server implementation. These are features, behaviors, or responsibilities that are intentionally excluded from the scope of the MCP server, either due to architectural boundaries, security concerns, or project focus.

## Non-Goals

1. **Direct Model Training Logic**
   - The MCP server does not implement or manage the actual training logic for ML models. All training operations are delegated to underlying Kubeflow Trainer APIs and clients.

2. **End-to-End Workflow Orchestration**
   - The MCP server does not orchestrate full ML pipelines or workflows. It exposes tool interfaces for job lifecycle management, but workflow orchestration is handled by other Kubeflow components.

3. **User Authentication and Authorization**
   - The MCP server does not provide user authentication, authorization, or access control. These concerns are managed by upstream services or platform infrastructure.

4. **Resource Scheduling and Allocation**
   - The MCP server does not schedule or allocate compute resources (e.g., GPUs, nodes). Resource management is handled by Kubeflow and Kubernetes.

5. **Custom Code Execution Security**
   - The MCP server does not guarantee secure execution of arbitrary user code. Custom training tools are stubs and require additional sandboxing and validation for production use.

6. **Persistent Storage Management**
   - The MCP server does not manage persistent storage, data volumes, or artifact repositories. Storage is handled by Kubeflow and external systems.

7. **Monitoring and Observability**
   - The MCP server does not provide built-in monitoring, logging, or observability dashboards. It exposes status tools, but observability is handled by other Kubeflow services.

8. **Backward Compatibility Guarantees**
   - The MCP server does not guarantee backward compatibility across Kubeflow versions. It tracks the current SDK and KEP-936 proposal.

9. **Protocol/Transport Layer Implementation**
   - The MCP server does not implement protocol or transport layers (e.g., HTTP, gRPC). It is protocol-neutral and intended for agentic tool registry use.

10. **UI/UX or Frontend Components**
    - The MCP server does not provide any user interface or frontend components. It is backend-only and tool-centric.

---

For full project scope and goals, see the README and KEP-936 proposal reference.
