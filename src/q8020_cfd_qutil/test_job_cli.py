"""
CLI tool for submitting and retrieving IBM Quantum jobs.

submit  — build a Bell state circuit and submit it via submit_job.
getresult — poll a job by ID, write metautil fragments for results
            and backend calibration.
"""

import argparse
import json
from pathlib import Path

from qiskit import QuantumCircuit

from q8020_cfd_qutil.job import get_job_result, submit_job


def _bell_circuit() -> QuantumCircuit:
    """2-qubit Bell state circuit with measurement on both qubits."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test IBM Quantum job submission and retrieval."
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["submit", "getresult"],
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="IBM backend name (required for submit).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Shots per circuit (submit only).",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Job ID to retrieve (required for getresult).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Write metautil fragments to this directory.",
    )
    args = parser.parse_args()

    if args.action == "submit":
        if args.backend is None:
            parser.error("--backend is required for --action submit")
        job_id = submit_job(_bell_circuit(), args.backend, shots=args.shots)

        submit_data = {
            "job_id": job_id,
            "backend": args.backend,
            "shots": args.shots,
            "action": "submit",
        }
        outdir = Path(args.outdir) if args.outdir else None
        if outdir and outdir.is_dir():
            from q8020_cfd_metautil.meta_fragment import write_results
            write_results(outdir, submit_data)
        print(json.dumps(submit_data, default=str))

    else:  # getresult
        if args.job_id is None:
            parser.error("--job-id is required for --action getresult")
        result = get_job_result(args.job_id)

        outdir = Path(args.outdir) if args.outdir else None
        if outdir and outdir.is_dir():
            _write_fragments(result, outdir)
        else:
            # No outdir: print compact counts-only JSON
            print(json.dumps({
                "job_id": result["job_id"],
                "status": result["status"],
                "results": result.get("results"),
            }, indent=2, default=str))


def _write_fragments(result: dict, outdir: Path) -> None:
    """Split get_job_result output into metautil fragment files."""
    from q8020_cfd_metautil.meta_fragment import (
        write_backend,
        write_results,
    )

    # Results fragment: counts + metrics + status
    results_data = {
        "job_id": result["job_id"],
        "status": result["status"],
        "results": result.get("results"),
        "metrics": result.get("metrics"),
    }
    write_results(outdir, results_data)

    # Backend calibration fragment
    backend_data = result.get("backend")
    if backend_data:
        write_backend(outdir, backend_data)

    # Compact stdout for the harvester
    print(json.dumps({
        "job_id": result["job_id"],
        "status": result["status"],
        "results": result.get("results"),
    }, default=str))


if __name__ == "__main__":
    main()
