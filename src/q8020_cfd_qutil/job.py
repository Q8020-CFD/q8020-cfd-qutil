"""
Asynchronous job submission and retrieval for IBM Quantum hardware.

submit_job  — transpile and fire a SamplerV2 job; return the job ID.
get_job_result — fetch status and counts for a previously submitted job.
"""

from typing import Any

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler

from q8020_cfd_qutil.backend import get_backend, get_service
from q8020_cfd_qutil.circuit import transpile_circuit


def submit_job(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend_name: str,
    shots: int = 4096,
    optimization_level: int = 1,
    token: str | None = None,
    channel: str = "ibm_cloud",
    instance: str | None = None,
) -> str:
    """
    Transpile and submit circuit(s) to a named IBM Quantum backend.

    Auth token resolution order: argument → IBM_QUANTUM_TOKEN env var
    → previously saved QiskitRuntimeService credentials.

    Args:
        circuits: One QuantumCircuit or a list of them.
        backend_name: IBM backend name (e.g., 'ibm_brisbane').
        shots: Shots per circuit pub.
        optimization_level: Transpilation optimization level (0-3).
        token: IBM Quantum API token.
        channel: Service channel ('ibm_quantum' or 'ibm_cloud').
        instance: Hub/group/project instance string, if required.

    Returns:
        Job ID string.
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]

    backend = get_backend(
        name=backend_name,
        backend_type="hardware",
        token=token,
        channel=channel,
        instance=instance,
    )

    transpiled = [
        transpile_circuit(qc, backend, optimization_level=optimization_level)[0]
        for qc in circuits
    ]

    sampler = Sampler(backend)
    job = sampler.run(transpiled, shots=shots)
    return job.job_id()


def get_job_result(
    job_id: str,
    token: str | None = None,
    channel: str = "ibm_cloud",
    instance: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve status and results for a previously submitted IBM Quantum job.

    Does not block when the job is still running; inspect the 'status'
    field before using 'results'.

    Args:
        job_id: IBM Quantum job ID returned by submit_job.
        token: IBM Quantum API token.
        channel: Service channel ('ibm_quantum' or 'ibm_cloud').
        instance: Hub/group/project instance string, if required.

    Returns:
        Dict with keys:
          - job_id  : str — echoed back for convenience
          - status  : str — e.g. 'DONE', 'RUNNING', 'QUEUED', 'ERROR'
          - results : list of per-circuit dicts (counts keyed by register
                      name), or None when the job has not completed.
    """
    service = get_service(token=token, channel=channel, instance=instance)
    job = service.job(job_id)

    status_raw = job.status()
    # qiskit-ibm-runtime < 0.20 returns a JobStatus enum; >= 0.20 a plain str.
    status_str: str = (
        status_raw.name if hasattr(status_raw, "name") else str(status_raw)
    )

    results: list[dict[str, Any]] | None = None
    if status_str == "DONE":
        results = _extract_counts(job.result())

    out: dict[str, Any] = {
        "job_id": job_id,
        "status": status_str,
        "results": results,
    }

    # Attach execution metrics (timestamps, quantum_seconds usage)
    try:
        out["metrics"] = job.metrics()
    except Exception:
        pass

    # Attach backend calibration snapshot from execution time
    try:
        backend = job.backend()
        if backend is not None:
            out["backend_name"] = backend.name
            from q8020_cfd_metautil.meta_fragment import make_backend_meta
            out["backend"] = make_backend_meta(backend)
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_counts(
    primitive_result: Any,
) -> list[dict[str, Any]]:
    """
    Pull shot counts out of a SamplerV2 PrimitiveResult.

    Each element of the returned list corresponds to one circuit pub and
    contains a 'counts' dict mapping register name → {bitstring: count}.
    """
    out: list[dict[str, Any]] = []
    for pub_result in primitive_result:
        counts_by_reg: dict[str, dict[str, int]] = {}
        for attr_name in dir(pub_result.data):
            if attr_name.startswith("_"):
                continue
            attr = getattr(pub_result.data, attr_name)
            if hasattr(attr, "get_counts"):
                counts_by_reg[attr_name] = attr.get_counts()
        out.append({"counts": counts_by_reg})
    return out
