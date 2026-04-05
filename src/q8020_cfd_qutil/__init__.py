"""
Qiskit utilities for quantum circuit execution and backend management.

This package provides:
- Backend connection and creation (simulator, fake, hardware)
- Circuit transpilation and execution utilities
- Asynchronous job submission and retrieval for IBM Quantum hardware
"""

from q8020_cfd_qutil.backend import get_backend, get_service
from q8020_cfd_qutil.circuit import (
    get_circuit_info,
    transpile_circuit,
    execute_circuit_counts,
)
from q8020_cfd_qutil.job import get_job_result, submit_job

__all__ = [
    "get_backend",
    "get_service",
    "get_circuit_info",
    "transpile_circuit",
    "execute_circuit_counts",
    "submit_job",
    "get_job_result",
]
