"""
InspectCore — Trust & Audit Core for AI decisions.

Minimal, Termux-friendly module that extracts the provable-decision
layer from TurboLLM and TOTAL-Neuro:

    input (prompt + hidden states)
        -> G-Space Inspector   (safety decision + metrics)
        -> PQ Signer           (ECDSA + Dilithium3 hybrid, with fallback)
        -> PoI package
        -> Ledger              (append-only, hash-chained, SQLite)
        -> output: decision + proof + block_id

Design goals:
    * no GPU, no vLLM, no torch required
    * runs on aarch64 / Termux
    * CLI-first, library-first; HTTP API is optional and comes later
    * every piece has a graceful fallback (simulation mode)

Public API:
    from inspectcore import Inspector, Signer, Ledger, Attestation, Agent

See README.md for usage.
"""

__version__ = "0.1.0"
__all__ = [
    "Inspector",
    "Signer",
    "Ledger",
    "Attestation",
    "Agent",
    "__version__",
]


def __getattr__(name):
    """
    Lazy imports so that `import inspectcore` is cheap and does not
    pull in numpy / cryptography until the user actually needs them.
    This keeps CLI startup fast on Termux.
    """
    if name == "Inspector":
        from .inspector import Inspector
        return Inspector
    if name == "Signer":
        from .signer import Signer
        return Signer
    if name == "Ledger":
        from .ledger import Ledger
        return Ledger
    if name == "Attestation":
        from .attestation import Attestation
        return Attestation
    if name == "Agent":
        from .agent import Agent
        return Agent
    raise AttributeError(f"module 'inspectcore' has no attribute {name!r}")
