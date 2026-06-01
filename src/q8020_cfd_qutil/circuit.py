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


def circuit_stats_in_basis(
    qc: QuantumCircuit,
    basis_gates: list[str],
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
) -> dict:
    """Logical depth/gate counts after decomposing to a fixed gate basis.

    Transpiles all-to-all (no backend/coupling map, so no routing swaps)
    purely to measure how the circuit lowers onto ``basis_gates`` -- e.g.
    forcing a dense ``UnitaryGate`` to decompose into cx + 1-qubit gates.
    Use for hardware-cost reporting; it does not affect execution.

    Returns the same dict as ``get_circuit_info`` (depth, num_qubits,
    gate_counts), measured on the basis-decomposed circuit.
    """
    kwargs = {
        "basis_gates": basis_gates,
        "optimization_level": optimization_level,
    }
    if seed_transpiler is not None:
        kwargs["seed_transpiler"] = seed_transpiler
    return get_circuit_info(transpile(qc, **kwargs))


# Default basis for hardware-cost reporting: canonical IBM
# superconducting set (cx is the dominant-cost metric, comparable
# across methods).
DEFAULT_METRIC_BASIS = ["cx", "rz", "sx", "x"]


def _stats_worker(conn, qc, basis_gates, strategy, opt_level, seed, reps):
    """Child-process body: decompose qc and ship back circuit stats.

    Wrapped so any *catchable* error returns a reason; an *uncatchable*
    crash (segfault) just kills this child -- the parent detects the
    non-zero exit code.
    """
    try:
        if strategy == "transpile":
            kwargs = {
                "basis_gates": basis_gates,
                "optimization_level": opt_level,
            }
            if seed is not None:
                kwargs["seed_transpiler"] = seed
            out = transpile(qc, **kwargs)
        else:  # progressive gate-definition expansion (no unitary synth)
            out = qc
            for _ in range(reps):
                out = out.decompose()
        info = get_circuit_info(out)
        info["available"] = True
        info["method"] = strategy
        conn.send(info)
    except BaseException as exc:  # noqa: BLE001 - report, never raise
        conn.send({
            "available": False,
            "reason": f"{strategy}: {type(exc).__name__}: {exc}",
        })
    finally:
        conn.close()


def safe_circuit_stats_in_basis(
    qc: QuantumCircuit,
    basis_gates: list[str],
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
    timeout: float = 180.0,
    try_decompose: bool = True,
) -> dict:
    """Subprocess-isolated basis-decomposition stats.

    Runs the basis transpile in a child process so a transpiler crash
    (e.g. the qs_decomposition segfault on LCU multi-controlled SELECT
    gates) or a hang can never take down the caller -- the simulation
    keeps running and just records that metrics were unavailable here.

    Tries a full ``transpile`` first; if that child crashes/times out and
    ``try_decompose`` is set, falls back to progressive ``.decompose()``
    (gate-definition expansion, which avoids unitary synthesis) in a
    fresh child.  Each attempt is fully isolated.

    Returns ``{available: True, depth, num_qubits, gate_counts, method}``
    on success, else ``{available: False, reason}``.
    """
    import multiprocessing as mp

    strategies = [("transpile", optimization_level, 0)]
    if try_decompose:
        strategies.append(("decompose", 0, 8))

    # 'spawn', not 'fork': the caller has live Aer/BLAS threads, and
    # forking a multithreaded process deadlocks the child.  spawn starts
    # a clean interpreter (costs a re-import, but it's crash/hang safe).
    ctx = mp.get_context("spawn")
    reason = "not attempted"
    for strat, ol, reps in strategies:
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(
            target=_stats_worker,
            args=(child_conn, qc, basis_gates, strat, ol,
                  seed_transpiler, reps),
        )
        proc.start()
        child_conn.close()

        result = None
        capped = timeout is not None and timeout > 0
        deadline = time.time() + timeout if capped else None
        while True:
            if parent_conn.poll(0.2):
                try:
                    result = parent_conn.recv()
                except EOFError:
                    result = None
                break
            if not proc.is_alive():
                break  # crashed without sending (segfault)
            if capped and time.time() >= deadline:
                break  # exceeded the cap; handled below

        # A received result wins, even if the child hasn't fully exited
        # yet (it sends, then tears down).  Only treat still-alive +
        # no-result as a genuine timeout (capped runs only).
        if result is None and proc.is_alive():
            proc.terminate()
            proc.join(5)
            parent_conn.close()
            reason = f"{strat}: timeout>{timeout:.0f}s"
            continue

        proc.join(5)
        parent_conn.close()
        if result is None:
            reason = f"{strat}: crashed (exitcode={proc.exitcode})"
            continue
        if result.get("available"):
            return result
        reason = result.get("reason", f"{strat}: unknown")

    return {"available": False, "reason": reason}


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
