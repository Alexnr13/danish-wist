"""Tools for evaluating and training bots. See LEARNING.md.

The engine (`danish_wist`) never imports this package; the web game imports
only `learn.inference` (NumPy, no PyTorch) to play its trained bot. Modules
that need PyTorch import it themselves, so evaluation works without it.
"""
