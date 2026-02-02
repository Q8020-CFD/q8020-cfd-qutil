"""
Backend connection and creation utilities for Qiskit.

Three backend modes:
- "sim": AerSimulator with optional fake backend topology and noise modeling.
         Use for local development and testing with realistic noise.
- "fake": FakeBackendV2 directly (not wrapped in AerSimulator).
         Use when you need the actual fake backend object.
- "hardware": Real IBM Quantum hardware via QiskitRuntimeService.
         Requires IBM Quantum credentials.
"""

import os
from typing import Any

from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_ibm_runtime import QiskitRuntimeService, fake_provider


def _get_fake_backend(name: str):
    """Get a fake backend by name (internal helper)."""
    normalized = name.strip().capitalize()
    class_name = f"Fake{normalized}V2"
    
    if hasattr(fake_provider, class_name):
        backend_class = getattr(fake_provider, class_name)
        return backend_class()
    
    available = [attr.replace("Fake", "").replace("V2", "").lower() 
                 for attr in dir(fake_provider) 
                 if attr.startswith("Fake") and attr.endswith("V2")]
    raise ValueError(f"Unknown backend '{name}'. Available: {available}")


def _build_thermal_noise(t1: float, t2: float) -> NoiseModel:
    """Build thermal relaxation noise model from T1/T2 (internal helper)."""
    # Clamp T2 to valid range: T2 must be <= 2*T1
    if t2 > 2 * t1:
        t2 = 2 * t1
    
    noise_model = NoiseModel()
    
    # Gate times in microseconds (typical for superconducting qubits)
    gate_time_1q = 0.05  # 50 ns for single-qubit gates
    gate_time_2q = 0.3   # 300 ns for two-qubit gates
    
    error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
    error_2q = thermal_relaxation_error(t1, t2, gate_time_2q).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q))
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x', 'y', 'z', 'ry', 'rz', 'rx', 'sx'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    
    return noise_model


def get_service(
    token: str | None = None,
    channel: str = "ibm_quantum",
    instance: str | None = None,
) -> QiskitRuntimeService:
    """
    Connect to IBM Quantum Runtime Service.
    
    Token resolution order:
    1. token argument
    2. IBM_QUANTUM_TOKEN environment variable
    3. Previously saved credentials
    
    Args:
        token: IBM Quantum API token. If None, uses env var or saved credentials.
        channel: Service channel ("ibm_quantum" or "ibm_cloud").
        instance: Optional instance (hub/group/project) for ibm_quantum channel.
    
    Returns:
        QiskitRuntimeService instance.
    
    Raises:
        RuntimeError: If connection fails.
    """
    # Resolve token
    resolved_token = token or os.environ.get("IBM_QUANTUM_TOKEN")
    
    try:
        if resolved_token:
            # Use explicit token
            kwargs = {"channel": channel, "token": resolved_token}
            if instance:
                kwargs["instance"] = instance
            return QiskitRuntimeService(**kwargs)
        else:
            # Try saved credentials
            kwargs = {"channel": channel}
            if instance:
                kwargs["instance"] = instance
            return QiskitRuntimeService(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to IBM Quantum: {e}. "
            "Provide token via argument, IBM_QUANTUM_TOKEN env var, "
            "or save credentials with QiskitRuntimeService.save_account()."
        ) from e


def get_backend(
    name: str | None = None,
    backend_type: str = "sim",
    t1: float | None = None,
    t2: float | None = None,
    coupling_map: str = "default",
    token: str | None = None,
    channel: str = "ibm_quantum",
    instance: str | None = None,
) -> Any:
    """
    Get a configured backend for circuit execution.
    
    Three modes:
    - "sim" (default): AerSimulator with optional fake backend topology and noise.
    - "fake": FakeBackendV2 directly (not wrapped in AerSimulator).
    - "hardware": Real IBM Quantum hardware via QiskitRuntimeService.
    
    Args:
        name: Backend name. For "sim"/"fake": fake backend name (e.g., 'manila').
              For "hardware": IBM backend name, or None for least_busy.
        backend_type: One of "sim", "fake", "hardware".
        t1: T1 relaxation time in µs (sim mode only). Overrides backend noise.
        t2: T2 dephasing time in µs (sim mode only). Must be <= 2*T1.
        coupling_map: "default" or "all-to-all" (sim mode only).
        token: IBM Quantum token (hardware mode only).
        channel: IBM channel (hardware mode only).
        instance: IBM instance (hardware mode only).
    
    Returns:
        Backend object (AerSimulator, FakeBackendV2, or IBMBackend).
    
    Examples:
        get_backend("manila")                    # Sim with Manila topology + noise
        get_backend("manila", t1=50, t2=70)      # Sim with custom noise
        get_backend()                            # Ideal simulator, no noise
        get_backend("manila", backend_type="fake")  # FakeBackendV2 directly
        get_backend("ibm_brisbane", backend_type="hardware")  # Real hardware
        get_backend(backend_type="hardware")     # Least busy real hardware
    """
    if backend_type == "fake":
        if name is None:
            raise ValueError("name is required for backend_type='fake'")
        return _get_fake_backend(name)
    
    if backend_type == "hardware":
        service = get_service(token=token, channel=channel, instance=instance)
        if name is None:
            return service.least_busy(operational=True, simulator=False)
        return service.backend(name)
    
    if backend_type == "sim":
        fake_backend = None
        noise_model = None
        coupling_map_obj = None
        
        # Get fake backend if name specified
        if name is not None:
            fake_backend = _get_fake_backend(name)
            
            # Handle coupling map
            if coupling_map == "all-to-all":
                coupling_map_obj = CouplingMap.from_full(fake_backend.num_qubits)
            else:
                coupling_map_obj = fake_backend.coupling_map
            
            # Use backend's noise unless t1/t2 override
            if t1 is None or t2 is None:
                noise_model = NoiseModel.from_backend(fake_backend)
        
        # Build custom noise model if t1/t2 provided
        if t1 is not None and t2 is not None and t1 > 0 and t2 > 0:
            noise_model = _build_thermal_noise(t1, t2)
        
        # Create simulator
        if fake_backend is not None:
            simulator = AerSimulator.from_backend(fake_backend)
            if coupling_map == "all-to-all":
                # Override coupling map for all-to-all
                simulator.set_options(coupling_map=list(coupling_map_obj.get_edges()))
        else:
            simulator = AerSimulator()
        
        # Attach noise model if any
        if noise_model is not None:
            simulator.set_options(noise_model=noise_model)
        
        return simulator
    
    raise ValueError(f"Unknown backend_type '{backend_type}'. Use 'sim', 'fake', or 'hardware'.")
