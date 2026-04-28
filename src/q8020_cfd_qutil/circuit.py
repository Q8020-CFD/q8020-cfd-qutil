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
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
) -> tuple[QuantumCircuit, dict]:
    """
    Transpile a circuit for a backend.
    
    Args:
        qc: The quantum circuit to transpile.
        backend: Backend object (AerSimulator, FakeBackendV2, or IBMBackend).
        optimization_level: Transpilation optimization level (0-3).
        seed_transpiler: Seed for reproducible layout/routing (default: None).
    
    Returns:
        Tuple of (transpiled_circuit, transpile_info_dict).
    """
    transpile_start = time.time()
    kwargs = {"backend": backend, "optimization_level": optimization_level}
    if seed_transpiler is not None:
        kwargs["seed_transpiler"] = seed_transpiler
    qc_transpiled = transpile(qc, **kwargs)
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
    shots: int = 1024,
    seed: int | None = None,
) -> tuple[dict[str, int], dict]:
    """Execute a transpiled circuit and return joint counts.

    Returns (counts, exec_info).  ``counts`` keys are single bitstrings
    with **no spaces**, regardless of how many classical registers the
    circuit has.  For multi-register circuits Aer normally space-joins
    the registers; this function strips the space so all callers see a
    uniform contract.  Single-creg circuits are unaffected (no space to
    strip).

    Bitstring layout for multi-creg: most-recently-added register on the
    LEFT, matching Aer convention.  V2 Sampler's ``join_data()`` produces
    the same layout natively.
    """
    try:
        from qiskit_aer import AerSimulator
        is_aer = isinstance(backend, AerSimulator)
    except ImportError:
        is_aer = False

    t0 = time.time()

    if is_aer:
        kwargs = {"shots": shots}
        if seed is not None:
            kwargs["seed_simulator"] = seed
        result = backend.run(qc_transpiled, **kwargs).result()
        raw = result.get_counts()
        counts = {k.replace(" ", ""): v for k, v in raw.items()}
        exec_info = {
            "wall_time": time.time() - t0,
            "shots_requested": shots,
            "shots_executed": result.results[0].shots,
            "backend_time": result.results[0].time_taken,
        }
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        sampler = Sampler(backend)
        if seed is not None:
            sampler.options.simulator.seed_simulator = seed
        job = sampler.run([qc_transpiled], shots=shots)
        pub_result = job.result()[0]
        counts = pub_result.join_data().get_counts()
        exec_info = {
            "wall_time": time.time() - t0,
            "shots_requested": shots,
            "job_id": job.job_id(),
        }

    return counts, exec_info
