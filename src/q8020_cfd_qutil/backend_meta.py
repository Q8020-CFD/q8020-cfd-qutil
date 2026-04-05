"""Look up IBM Quantum backend specs and emit structured metadata JSON.

Install (once):
    pip install -e q8020-cfd-qutil        # registers the CLI entry point

Three backend modes:

  sim        Local AerSimulator.  When a backend name is given, the
             simulator inherits that backend's topology, coupling map,
             and noise model.  Without a name you get an ideal (noiseless)
             simulator.

  fake       Raw FakeBackendV2 object — the Qiskit-provided emulation of
             a retired IBM device.  Requires a name.

  hardware   Real IBM Quantum hardware via QiskitRuntimeService.
             Credentials are resolved in order: --token flag,
             IBM_QUANTUM_TOKEN env var, ~/.qiskit/qiskit-ibm.json.
             Omit the name to auto-select the least-busy device.

Available backend names (IBM legacy emulations):
    1q:  armonk
    5q:  athens, belem, bogota, casablanca, essex, lima, london,
         manila, ourense, quito, rome, santiago, valencia, vigo, yorktown
    7q:  jakarta, lagos, nairobi
   14q:  melbourne
   15q:  guadalupe
   16q:  almaden, singapore
   20q:  boeblingen, johannesburg, poughkeepsie
   27q:  cairo, cambridge, hanoi, kolkata, montreal, mumbai,
         paris, sydney, toronto
   53q:  rochester
   65q:  brooklyn, manhattan, washington

CLI usage:

  # Print full backend metadata JSON to stdout
  q8020-backend-meta sim melbourne
  q8020-backend-meta sim                          # ideal, no noise
  q8020-backend-meta fake manila

  # Write a q8020_backend_000.json fragment to a directory
  q8020-backend-meta sim melbourne -d /tmp/my_case

  # Real hardware (token required)
  q8020-backend-meta hardware ibm_brisbane --token $IBM_QUANTUM_TOKEN
  q8020-backend-meta hardware                     # least-busy device

  # Pipe into jq, save, etc.
  q8020-backend-meta sim guadalupe | jq .coupling_map
  q8020-backend-meta sim cairo > cairo_backend.json

Python API:

  from q8020_cfd_qutil.backend_meta import lookup
  meta = lookup(backend_type="sim", name="melbourne")
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from q8020_cfd_qutil.backend import get_backend
from q8020_cfd_metautil.meta_fragment import make_backend_meta, write_backend

_QISKIT_IBM_JSON = Path.home() / ".qiskit" / "qiskit-ibm.json"


def _load_qiskit_ibm_creds() -> dict[str, Any] | None:
    """Read the first profile from ~/.qiskit/qiskit-ibm.json, if it exists."""
    if not _QISKIT_IBM_JSON.is_file():
        return None
    try:
        profiles = json.loads(_QISKIT_IBM_JSON.read_text(encoding="utf-8"))
        if not profiles:
            return None
        # Use the first (or only) profile
        name = next(iter(profiles))
        return profiles[name]
    except Exception:
        return None


def lookup(
    backend_type: str = "sim",
    name: str | None = None,
    token: str | None = None,
    channel: str = "ibm_cloud",
    instance: str | None = None,
) -> dict[str, Any]:
    """Instantiate a backend and return its metadata dict.

    Args:
        backend_type: One of "sim", "fake", "hardware".
        name: Backend name (e.g. "melbourne", "ibm_brisbane").
        token: IBM Quantum token (hardware mode).
        channel: IBM channel (hardware mode).
        instance: IBM instance (hardware mode).

    Returns:
        Backend metadata dict.
    """
    backend = get_backend(
        name=name,
        backend_type=backend_type,
        token=token,
        channel=channel,
        instance=instance,
    )
    return make_backend_meta(backend)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Look up backend details and emit structured metadata JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:

  # Noisy simulator (inherits Melbourne topology + noise)
  q8020-backend-meta sim melbourne

  # Ideal simulator (no noise, no topology)
  q8020-backend-meta sim

  # Raw fake backend object
  q8020-backend-meta fake manila

  # Write fragment file to a case directory
  q8020-backend-meta sim guadalupe -d ~/q8020/2026-02-07/abc123

  # Real IBM hardware (needs token or IBM_QUANTUM_TOKEN env var)
  q8020-backend-meta hardware ibm_brisbane --token $IBM_QUANTUM_TOKEN

  # Least-busy real device
  q8020-backend-meta hardware --token $IBM_QUANTUM_TOKEN

  # Pipe to jq
  q8020-backend-meta sim cairo | jq '.coupling_map | length'

Available backend names (IBM legacy emulations):
   1q: armonk
   5q: athens belem bogota casablanca essex lima london manila
       ourense quito rome santiago valencia vigo yorktown
   7q: jakarta lagos nairobi
  14q: melbourne
  15q: guadalupe
  16q: almaden singapore
  20q: boeblingen johannesburg poughkeepsie
  27q: cairo cambridge hanoi kolkata montreal mumbai paris
       sydney toronto
  53q: rochester
  65q: brooklyn manhattan washington
""",
    )
    parser.add_argument(
        "backend_type",
        choices=["sim", "fake", "hardware"],
        help="Backend mode: sim, fake, or hardware",
    )
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Backend name (e.g. melbourne, ibm_brisbane). "
             "Omit for ideal sim or least-busy hardware.",
    )
    parser.add_argument(
        "--outdir", "-d",
        type=str,
        default=None,
        help="Write a q8020_backend fragment to this directory instead of stdout",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="IBM Quantum API token (hardware mode; also reads IBM_QUANTUM_TOKEN env var)",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="ibm_cloud",
        help="IBM channel (default: ibm_cloud)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="IBM instance hub/group/project (hardware mode)",
    )

    args = parser.parse_args()

    # Auto-fill hardware credentials from ~/.qiskit/qiskit-ibm.json
    token = args.token
    channel = args.channel
    instance = args.instance
    if args.backend_type == "hardware" and token is None:
        creds = _load_qiskit_ibm_creds()
        if creds:
            token = creds.get("token", token)
            channel = creds.get("channel", channel)
            instance = creds.get("instance", instance)
            print(f"Using saved credentials from {_QISKIT_IBM_JSON}",
                  file=sys.stderr)

    data = lookup(
        backend_type=args.backend_type,
        name=args.name,
        token=token,
        channel=channel,
        instance=instance,
    )

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
        if not outdir.exists():
            print(f"Error: directory does not exist: {outdir}", file=sys.stderr)
            sys.exit(1)
        write_backend(outdir, data)
        print(f"✅ Backend metadata written to: {outdir}", file=sys.stderr)
    else:
        json.dump(data, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
