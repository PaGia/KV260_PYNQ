#!/usr/bin/env python3
"""
3-Frequency Signal Test

Records 3 separate 10-second sessions, one per unknown input frequency.
Each session produces a .bin file for PC-side FFT analysis.

Usage on KV260 (via JupyterClient):
    from test_3freq import record_freq
    record_freq(1)   # connect signal 1, then run
    record_freq(2)   # connect signal 2, then run
    record_freq(3)   # connect signal 3, then run
"""

import sys
sys.path.insert(0, "/root/jupyter_notebooks/RHS_SPI")
from rhs2116_driver import record

DURATION = 10  # seconds per recording
BITSTREAM = "/root/jupyter_notebooks/RHS_SPI/rhs2116_system.bit"

def record_freq(n):
    """Record frequency test N (1, 2, or 3)."""
    print("=== Recording frequency %d/3 (%d seconds) ===" % (n, DURATION))
    result = record(
        duration_sec=DURATION,
        output_file="/root/jupyter_notebooks/RHS_SPI/freq_%d.bin" % n,
        bitstream_path=BITSTREAM,
    )
    print()
    if 'error' in result:
        print("ERROR: %s" % result['error'])
    else:
        print("Saved: %s (%d frames)" % (result['output_file'], result['frames_written']))
    return result
