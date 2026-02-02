"""
Circuit utilities for transpilation and execution.

Provides functions for:
- Extracting circuit statistics (depth, gate counts)
- Transpiling circuits for backends
- Executing circuits and collecting counts
"""

import time
from typing import Any

from qiskit import QuantumCircuit, transpile


def get_circuit_info(qc: QuantumCircuit) -> dict:
    """
    Extract circuit statistics.
    
    Args:
        qc: Any QuantumCircuit (original or transpiled).
    
    Returns:
        Dict with depth, num_qubits, gate_counts.
    """
    return {
        "depth": qc.depth(),
        "num_qubits": qc.num_qubits,
        "gate_counts": dict(qc.count_ops()),
    }


def transpile_circuit(
    qc: QuantumCircuit,
    backend: Any,
    optimization_level: int = 1
) -> tuple[QuantumCircuit, dict]:
    """
    Transpile a circuit for a backend.
    
    Args:
        qc: The quantum circuit to transpile.
        backend: Backend object (AerSimulator, FakeBackendV2, or IBMBackend).
        optimization_level: Transpilation optimization level (0-3).
    
    Returns:
        Tuple of (transpiled_circuit, transpile_info_dict).
    """
    transpile_start = time.time()
    qc_transpiled = transpile(qc, backend=backend, optimization_level=optimization_level)
    transpile_time = time.time() - transpile_start
    
    transpile_info = {
        "wall_time": transpile_time,
        "optimization_level": optimization_level,
        "before": get_circuit_info(qc),
        "after": get_circuit_info(qc_transpiled),
    }
    
    return qc_transpiled, transpile_info


def execute_circuit_counts(
    qc_transpiled: QuantumCircuit,
    backend: Any,
    shots: int = 1024
) -> tuple[dict[str, int], dict]:
    """
    Execute a transpiled circuit on a backend.
    
    Args:
        qc_transpiled: The transpiled quantum circuit to execute.
        backend: Backend object (AerSimulator, FakeBackendV2, or IBMBackend).
        shots: Number of shots.
    
    Returns:
        Tuple of (counts_dict, execution_info_dict).
    """
    # Detect backend type: AerSimulator uses backend.run(), others use SamplerV2
    try:
        from qiskit_aer import AerSimulator
        is_aer = isinstance(backend, AerSimulator)
    except ImportError:
        is_aer = False

    execute_start = time.time()

    if is_aer:
        # Aer path: use backend.run() API
        result = backend.run(qc_transpiled, shots=shots).result()
        execute_time = time.time() - execute_start

        status = result.results[0].status
        exec_info = {
            "wall_time": execute_time,
            "backend_time": result.results[0].time_taken,
            "shots_requested": shots,
            "shots_executed": result.results[0].shots,
            "status": status.name if hasattr(status, 'name') else str(status),
        }
        counts = result.get_counts()
    else:
        # V2 Sampler path: for IBMBackend, FakeBackendV2, etc.
        from qiskit_ibm_runtime import SamplerV2 as Sampler

        sampler = Sampler(backend)
        job = sampler.run([qc_transpiled], shots=shots)
        result = job.result()
        execute_time = time.time() - execute_start

        pub_result = result[0]

        # Get counts from the first available classical register
        if hasattr(pub_result.data, "meas"):
            counts = pub_result.data.meas.get_counts()
        else:
            # Find first BitArray in data
            counts = None
            for attr_name in dir(pub_result.data):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(pub_result.data, attr_name)
                if hasattr(attr, "get_counts"):
                    counts = attr.get_counts()
                    break
            if counts is None:
                raise RuntimeError("No countable register found in result")

        exec_info = {
            "wall_time": execute_time,
            "job_id": job.job_id(),
            "shots_requested": shots,
        }

    return counts, exec_info
