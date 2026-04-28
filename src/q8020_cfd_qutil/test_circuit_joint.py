"""Tests for execute_circuit_counts joint-counts upgrade (§7a of SPEC-shots-backend)."""

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator

from q8020_cfd_qutil.backend import get_backend
from q8020_cfd_qutil.circuit import (
    execute_circuit_counts,
    transpile_circuit,
)

# Any small fake backend works; tests only need >= 3 qubits.
_FAKE_BACKEND_NAME = "lima"


def _bell_two_creg_circuit() -> QuantumCircuit:
    """Two-creg circuit: Bell pair across data(1) and anc(1).

    Only '00' and '11' should appear in joint counts.
    Marginal-product would predict uniform over {00,01,10,11}.
    """
    cr_data = ClassicalRegister(1, "data")
    cr_anc = ClassicalRegister(1, "anc")
    qc = QuantumCircuit(2)
    qc.add_register(cr_data)
    qc.add_register(cr_anc)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, cr_data[0])
    qc.measure(1, cr_anc[0])
    return qc


def _simple_two_creg_circuit(q: int = 2) -> QuantumCircuit:
    """Known-state two-creg circuit: data=|11>, anc=|0>."""
    cr_data = ClassicalRegister(q, "data")
    cr_anc = ClassicalRegister(1, "anc")
    qc = QuantumCircuit(q + 1)
    qc.add_register(cr_data)
    qc.add_register(cr_anc)
    for i in range(q):
        qc.x(i)
    for i in range(q):
        qc.measure(i, cr_data[i])
    qc.measure(q, cr_anc[0])
    return qc


class TestJointCountsAer:

    def test_counts_key_shape(self):
        """Joint dict keys have q+anc total bits, no spaces."""
        qc = _simple_two_creg_circuit(q=2)
        backend = AerSimulator()
        qc_t, _ = transpile_circuit(qc, backend, optimization_level=0)
        counts, _ = execute_circuit_counts(
            qc_t, backend, shots=1000, seed=42,
        )
        for key in counts:
            assert " " not in key, f"space in key: {key!r}"
            assert len(key) == 3, f"expected 3-bit key, got {key!r}"

    def test_counts_recover_correlation(self):
        """Bell pair: joint counts show only correlated bitstrings."""
        qc = _bell_two_creg_circuit()
        backend = AerSimulator()
        qc_t, _ = transpile_circuit(qc, backend, optimization_level=0)
        counts, _ = execute_circuit_counts(
            qc_t, backend, shots=10000, seed=42,
        )
        # Only '00' and '11' should appear (data+anc, no space)
        assert set(counts.keys()).issubset({"00", "11"}), (
            f"unexpected keys: {set(counts.keys())}"
        )
        # Both outcomes present
        assert "00" in counts and "11" in counts
        # Marginal product would predict 4 outcomes; joint has only 2
        assert len(counts) == 2

    def test_seed_reproducibility_aer(self):
        """Same seed, same circuit -> identical counts dicts."""
        qc = _simple_two_creg_circuit(q=2)
        backend = AerSimulator()
        qc_t, _ = transpile_circuit(qc, backend, optimization_level=0)
        c1, _ = execute_circuit_counts(
            qc_t, backend, shots=2048, seed=99,
        )
        c2, _ = execute_circuit_counts(
            qc_t, backend, shots=2048, seed=99,
        )
        assert c1 == c2


class TestJointCountsV2Fake:

    def test_counts_aer_vs_v2_match_distribution(self):
        """Aer and V2-fake produce same key shape and similar distribution."""
        qc = _simple_two_creg_circuit(q=2)
        shots = 20000

        # Aer path
        backend_aer = AerSimulator()
        qc_t_aer, _ = transpile_circuit(
            qc, backend_aer, optimization_level=0,
        )
        counts_aer, _ = execute_circuit_counts(
            qc_t_aer, backend_aer, shots=shots, seed=42,
        )

        # V2 fake path
        backend_fake = get_backend(
            _FAKE_BACKEND_NAME, backend_type="fake",
        )
        qc_t_fake, _ = transpile_circuit(
            qc, backend_fake, optimization_level=0,
        )
        counts_fake, _ = execute_circuit_counts(
            qc_t_fake, backend_fake, shots=shots, seed=42,
        )

        # Same key lengths
        for key in counts_aer:
            assert len(key) == 3
        for key in counts_fake:
            assert len(key) == 3

        # Dominant outcome present in both (noisy fake may have extras)
        dominant_aer = max(counts_aer, key=counts_aer.get)
        assert dominant_aer in counts_fake

    def test_seed_reproducibility_v2_fake(self):
        """Same seed on V2 fake backend -> identical counts."""
        qc = _simple_two_creg_circuit(q=2)
        backend = get_backend(
            _FAKE_BACKEND_NAME, backend_type="fake",
        )
        qc_t, _ = transpile_circuit(qc, backend, optimization_level=0)
        c1, _ = execute_circuit_counts(
            qc_t, backend, shots=2048, seed=77,
        )
        c2, _ = execute_circuit_counts(
            qc_t, backend, shots=2048, seed=77,
        )
        assert c1 == c2
