#!/usr/bin/env python3
"""
RHS2116 Standard Recording Driver for KV260

Usage (on KV260 via JupyterClient):
    from rhs2116_driver import record
    result = record(duration_sec=10)

Usage (custom parameters):
    result = record(
        duration_sec=60,
        output_file='/root/jupyter_notebooks/RHS_SPI/my_recording.bin',
        lower_bw=0.5,
        upper_bw=10000.0,
    )

Output: raw .bin file (112 bytes/frame, continuous)
    Parse on PC with struct.unpack('<28I', frame_bytes)
"""

import time
import gc
import struct
import numpy as np
from datetime import datetime

# =============================================================================
# Constants
# =============================================================================

# Frame parameters
FRAME_SIZE = 112        # bytes per frame (14 beats x 8 bytes)
FRAME_WORDS = 28        # uint32 per frame
BD_SIZE = 64            # bytes per BD descriptor
BD_WORDS = 16           # uint32 per BD

# Magic number
MAGIC_LOW = 0x49712F0B
MAGIC_HIGH = 0x8D542C8A

# AXI-Lite register addresses
ADDR_CTRL_REG       = 0x00
ADDR_MAX_TIMESTEP   = 0x04
ADDR_MISO_DELAY     = 0x0C
ADDR_DATA_STREAM_EN = 0x10
ADDR_AUX_CMD_CFG    = 0x14
ADDR_TRIGGER_REG    = 0x80
ADDR_AUX_RAM_ADDR   = 0x84
ADDR_AUX_RAM_DATA   = 0x88
ADDR_AUX_RAM_BANK   = 0x8C
ADDR_STATUS_REG     = 0xA0
ADDR_TIMESTAMP      = 0xA8
ADDR_BOARD_ID       = 0xB0

# DMA register addresses (S2MM)
S2MM_DMACR       = 0x30
S2MM_DMASR       = 0x34
S2MM_CURDESC     = 0x38
S2MM_CURDESC_MSB = 0x3C
S2MM_TAILDESC    = 0x40
S2MM_TAILDESC_MSB = 0x44

# Recording defaults
DEFAULT_SAMPLE_RATE = 30000
DEFAULT_LOWER_BW = 1.0
DEFAULT_UPPER_BW = 7500.0
DEFAULT_PHASE_SELECT = 3
DEFAULT_NUM_BD = 65536
DEFAULT_BITSTREAM = "/root/jupyter_notebooks/RHS_SPI/rhs2116_system.bit"
DEFAULT_OUTPUT_DIR = "/root/jupyter_notebooks/RHS_SPI"

BATCH_SIZE = 4096

# =============================================================================
# SPI Command Functions
# =============================================================================

def CLEAR_CMD():
    """CLEAR command - initialize ADC (must execute once after power-on)"""
    return 0x6A000000

def WRITE_CMD(reg_addr, data, u=False, m=False):
    """WRITE command - write to RHS2116 register
    Format: 10UM0000_RRRRRRRR_DDDDDDDDDDDDDDDD
    """
    cmd = 0x80000000 | ((reg_addr & 0xFF) << 16) | (data & 0xFFFF)
    if u: cmd |= 0x20000000
    if m: cmd |= 0x10000000
    return cmd

def READ_CMD(reg_addr, u=False, m=False):
    """READ command - read from RHS2116 register
    Format: 11UM0000_RRRRRRRR_0000000000000000
    Result returned 2 commands later (pipeline delay)
    """
    cmd = 0xC0000000 | ((reg_addr & 0xFF) << 16)
    if u: cmd |= 0x20000000
    if m: cmd |= 0x10000000
    return cmd

def DUMMY_CMD():
    """DUMMY command - no operation"""
    return 0xFFFFFFFF

# =============================================================================
# Bandwidth Lookup Tables (from RHS2116 Datasheet)
# =============================================================================
# Verified: 100% match with Intan STM32 rhsregisters.c formula output.
# Previous driver used curve-fit formulas with incorrect coefficients
# (copied from RHD2000, not RHS2116), causing wrong Reg4/6/7 values.

# Upper bandwidth: freq_Hz -> (reg4_value, reg5_value)
# Reg4 = (RH1_sel2 << 6) | RH1_sel1
# Reg5 = (RH2_sel2 << 6) | RH2_sel1
UPPER_BW_TABLE = {
    20000: (0x0008, 0x0004),
    15000: (0x000B, 0x0008),
    10000: (0x0011, 0x0010),
    7500:  (0x0016, 0x0017),
    5000:  (0x0021, 0x0025),
    3000:  (0x0043, 0x004D),
    2500:  (0x004D, 0x0059),
    2000:  (0x005B, 0x006C),
    1500:  (0x0081, 0x0097),
    1000:  (0x00AE, 0x00DE),
    750:   (0x00E9, 0x0124),
    500:   (0x015E, 0x01AB),
    300:   (0x0246, 0x02C2),
    250:   (0x02AA, 0x034D),
    200:   (0x0358, 0x0407),
    150:   (0x046C, 0x0548),
    100:   (0x0686, 0x07C5),
}

# Lower bandwidth: freq_Hz -> reg_value (same format for Reg 6 and Reg 7)
# reg_value = (RL_sel3 << 13) | (RL_sel2 << 7) | RL_sel1
LOWER_BW_TABLE = {
    1000.0: 0x000A,
    500.0:  0x000D,
    300.0:  0x000F,
    250.0:  0x0011,
    200.0:  0x0012,
    150.0:  0x0015,
    100.0:  0x0019,
    75.0:   0x001C,
    50.0:   0x0022,
    30.0:   0x002C,
    25.0:   0x0030,
    20.0:   0x0036,
    15.0:   0x003E,
    10.0:   0x0085,
    7.5:    0x0092,
    5.0:    0x00A8,
    3.0:    0x0114,
    2.5:    0x012A,
    2.0:    0x0188,
    1.5:    0x0209,
    1.0:    0x032C,
    0.75:   0x04B1,
    0.50:   0x08A3,
    0.30:   0x1401,
    0.25:   0x1B38,
    0.10:   0x3E10,
}

def lookup_upper_bandwidth(freq_hz):
    """Look up Reg4, Reg5 values from datasheet table.
    Selects nearest available frequency.
    """
    best = min(UPPER_BW_TABLE.keys(), key=lambda f: abs(f - freq_hz))
    return UPPER_BW_TABLE[best]

def lookup_lower_bandwidth(freq_hz):
    """Look up Reg6/Reg7 value from datasheet table.
    Selects nearest available frequency.
    """
    best = min(LOWER_BW_TABLE.keys(), key=lambda f: abs(f - freq_hz))
    return LOWER_BW_TABLE[best]

# =============================================================================
# Init Sequence Generation
# =============================================================================

def generate_init_sequence(sample_rate=30000, lower_bw=1.0, upper_bw=7500.0):
    """Generate RHS2116 complete initialization command sequence.

    Returns: list of 128 commands (real commands + DUMMY padding)
    """
    cmds = []

    # Step 1: Dummy reads to stabilize chip
    cmds.append(READ_CMD(255))
    cmds.append(READ_CMD(255))

    # Step 2: CLEAR command (initialize ADC)
    cmds.append(CLEAR_CMD())

    # Step 3: ADC bias (Reg 0) - from RHS2116 Datasheet (8-level table)
    # Note: Intan STM32 rhsregisters.c splits the >= 440 kS/s tier into two:
    #   <= 525 kS/s -> (3, 7) and > 525 kS/s -> (2, 4).
    #   These values do NOT appear in the datasheet. We follow the datasheet.
    adc_rate = 20 * sample_rate
    if adc_rate <= 120000:
        adc_buffer_bias, mux_bias = 32, 40
    elif adc_rate <= 140000:
        adc_buffer_bias, mux_bias = 16, 40
    elif adc_rate <= 175000:
        adc_buffer_bias, mux_bias = 8, 40
    elif adc_rate <= 220000:
        adc_buffer_bias, mux_bias = 8, 32
    elif adc_rate <= 280000:
        adc_buffer_bias, mux_bias = 8, 26
    elif adc_rate <= 350000:
        adc_buffer_bias, mux_bias = 4, 18
    elif adc_rate <= 440000:
        adc_buffer_bias, mux_bias = 3, 16
    else:  # >= 440 kS/s (600 kS/s @ 30 kHz sample rate)
        adc_buffer_bias, mux_bias = 3, 5
    reg0_val = ((adc_buffer_bias & 0x3F) << 6) | (mux_bias & 0x3F)
    cmds.append(WRITE_CMD(0, reg0_val))

    # Step 4: ADC output format (Reg 1) - twoscomp=1, DSPen=1, weak_miso=1, DSP_cutoff=12
    # 0x00DC: weak_miso=1, twoscomp=1, DSPen=1, DSP cutoff freq=12 (f_c ≈ 1.17 Hz @ 30kHz)
    cmds.append(WRITE_CMD(1, 0x00DC))

    # Step 5: Impedance test (Reg 2-3) - disabled
    cmds.append(WRITE_CMD(2, 0x0000))
    cmds.append(WRITE_CMD(3, 0x0080))

    # Step 6: Bandwidth (Reg 4-7) - from RHS2116 Datasheet lookup tables
    reg4, reg5 = lookup_upper_bandwidth(upper_bw)
    reg6 = lookup_lower_bandwidth(1000.0)  # RL_A: 1000 Hz (stim recovery)
    reg7 = lookup_lower_bandwidth(lower_bw)  # RL_B: normal recording
    cmds.append(WRITE_CMD(4, reg4))
    cmds.append(WRITE_CMD(5, reg5))
    cmds.append(WRITE_CMD(6, reg6))
    cmds.append(WRITE_CMD(7, reg7))

    # Step 7: AC amplifier power (Reg 8) - all 16 channels on
    cmds.append(WRITE_CMD(8, 0xFFFF))

    # Step 8: Trigger registers (Reg 10, 12)
    cmds.append(WRITE_CMD(10, 0x0000))  # amp fast settle = 0
    cmds.append(WRITE_CMD(12, 0x0000))  # amp fL select = 0 (use RL_B)

    # Step 9: Stimulation control init (Reg 32-38)
    cmds.append(WRITE_CMD(32, 0x0000))  # stim enable A = 0
    cmds.append(WRITE_CMD(33, 0x0000))  # stim enable B = 0
    cmds.append(WRITE_CMD(34, 0x0080))  # stim step size = 1.0 uA
    cmds.append(WRITE_CMD(35, 0x0000))  # stim bias
    cmds.append(WRITE_CMD(36, 0x0080))  # charge recovery DAC
    cmds.append(WRITE_CMD(37, 0x0000))  # charge recovery limit
    cmds.append(WRITE_CMD(38, 0xFFFF))  # DC amp power: all on

    # Step 10: Stimulation trigger registers (Reg 42, 44, 46, 48)
    cmds.append(WRITE_CMD(42, 0x0000))
    cmds.append(WRITE_CMD(44, 0x0000))
    cmds.append(WRITE_CMD(46, 0x0000))
    cmds.append(WRITE_CMD(48, 0x0000))

    # Pad to 128 with DUMMY
    while len(cmds) < 128:
        cmds.append(DUMMY_CMD())

    return cmds

# =============================================================================
# BRAM Operations
# =============================================================================

def write_aux_bram(mmio, slot_id, index, data):
    """Write single command to AUX BRAM.
    Pulse sequence: addr -> data -> WE assert -> WE deassert
    """
    mmio.write(ADDR_AUX_RAM_ADDR, index & 0x7F)
    mmio.write(ADDR_AUX_RAM_DATA, data)
    mmio.write(ADDR_AUX_RAM_BANK, (slot_id & 0x03) | 0x100)  # WE=1
    mmio.write(ADDR_AUX_RAM_BANK, slot_id & 0x03)             # WE=0

def write_aux_bram_slot(mmio, slot_id, commands):
    """Write command sequence to one AUX BRAM slot (max 128 entries)."""
    for i, cmd in enumerate(commands[:128]):
        write_aux_bram(mmio, slot_id, i, cmd)

# =============================================================================
# SG Mode DMA Setup
# =============================================================================

def setup_sg_dma(dma_mmio, mem, bd_base, num_bd):
    """Initialize SG Mode DMA BD ring and start DMA engine.

    Args:
        dma_mmio: DMA MMIO object
        mem: allocated memory array (uint32)
        bd_base: physical address of BD ring start
        num_bd: number of buffer descriptors
    """
    total_bd_bytes = num_bd * BD_SIZE
    data_base = bd_base + total_bd_bytes

    # Initialize BD ring
    for i in range(num_bd):
        bd_offset = i * BD_WORDS
        next_bd_addr = bd_base + ((i + 1) % num_bd) * BD_SIZE
        buf_addr = data_base + i * FRAME_SIZE

        mem[bd_offset + 0] = next_bd_addr & 0xFFFFFFFF  # NXTDESC
        mem[bd_offset + 1] = 0                           # NXTDESC_MSB
        mem[bd_offset + 2] = buf_addr & 0xFFFFFFFF       # BUFFER_ADDR
        mem[bd_offset + 3] = 0                           # BUFFER_ADDR_MSB
        mem[bd_offset + 4] = 0
        mem[bd_offset + 5] = 0
        mem[bd_offset + 6] = FRAME_SIZE                  # CONTROL (length)
        mem[bd_offset + 7] = 0                           # STATUS

    # Reset DMA
    dma_mmio.write(S2MM_DMACR, 0x00000004)
    time.sleep(0.01)

    # Set CURDESC
    dma_mmio.write(S2MM_CURDESC, bd_base & 0xFFFFFFFF)
    dma_mmio.write(S2MM_CURDESC_MSB, 0)

    # Run (SG Mode)
    dma_mmio.write(S2MM_DMACR, 0x00000001)
    time.sleep(0.001)

    # Set TAILDESC to last BD
    tail_bd = bd_base + (num_bd - 1) * BD_SIZE
    dma_mmio.write(S2MM_TAILDESC, tail_bd & 0xFFFFFFFF)
    dma_mmio.write(S2MM_TAILDESC_MSB, 0)

# =============================================================================
# Moving Tail Recording to .bin
# =============================================================================

def _recording_loop(mmio, dma_mmio, mem, bd_base, num_bd,
                    duration_sec, output_file):
    """Moving Tail recording loop - writes raw frames to .bin file.

    Args:
        mmio: SPI IP MMIO object
        dma_mmio: DMA MMIO object
        mem: allocated memory array (uint32)
        bd_base: physical address of BD ring start
        num_bd: number of buffer descriptors
        duration_sec: recording duration in seconds
        output_file: output .bin file path

    Returns:
        dict with recording stats
    """
    total_bd_bytes = num_bd * BD_SIZE
    data_base_words = total_bd_bytes // 4
    bd_mask = num_bd - 1  # for & instead of %

    written = 0
    read_idx = 0
    wrap_count = 0
    max_pending = 0

    batch_buffer = np.zeros(BATCH_SIZE * FRAME_WORDS, dtype=np.uint32)
    start_time = time.perf_counter()
    last_report = start_time

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()

    try:
        with open(output_file, 'wb', buffering=4 * 1024 * 1024) as f:
            while True:
                elapsed = time.perf_counter() - start_time
                if elapsed >= duration_sec:
                    break

                # Batch check BD completion
                completed = 0
                for i in range(BATCH_SIZE):
                    bd_idx = (read_idx + i) & bd_mask
                    if (mem[bd_idx * BD_WORDS + 7] >> 31) & 1:
                        completed += 1
                    else:
                        break

                if completed > 0:
                    # Batch read frame data (handle ring wrap)
                    end_idx = read_idx + completed
                    if end_idx <= num_bd:
                        src_start = data_base_words + read_idx * FRAME_WORDS
                        src_end = src_start + completed * FRAME_WORDS
                        batch_buffer[:completed * FRAME_WORDS] = mem[src_start:src_end]
                    else:
                        first_part = num_bd - read_idx
                        src_start = data_base_words + read_idx * FRAME_WORDS
                        src_end = data_base_words + num_bd * FRAME_WORDS
                        batch_buffer[:first_part * FRAME_WORDS] = mem[src_start:src_end]
                        second_part = completed - first_part
                        src_start = data_base_words
                        src_end = src_start + second_part * FRAME_WORDS
                        batch_buffer[first_part * FRAME_WORDS:completed * FRAME_WORDS] = mem[src_start:src_end]

                    # Write to file
                    f.write(batch_buffer[:completed * FRAME_WORDS].tobytes())

                    # Clear BD status
                    for i in range(completed):
                        bd_idx = (read_idx + i) & bd_mask
                        mem[bd_idx * BD_WORDS + 7] = 0

                    # Update TAILDESC
                    new_tail_idx = (read_idx + completed - 1) & bd_mask
                    new_tail_addr = bd_base + new_tail_idx * BD_SIZE
                    dma_mmio.write(S2MM_TAILDESC, new_tail_addr & 0xFFFFFFFF)
                    dma_mmio.write(S2MM_TAILDESC_MSB, 0)

                    # Advance read_idx
                    old_idx = read_idx
                    read_idx = (read_idx + completed) & bd_mask
                    if read_idx <= old_idx and completed > 0:
                        wrap_count += 1
                    if completed > max_pending:
                        max_pending = completed

                    written += completed
                else:
                    time.sleep(0.00001)  # 10 us

                # Periodic report (every 10s)
                now = time.perf_counter()
                if (now - last_report) >= 10.0:
                    rate = written / elapsed if elapsed > 0 else 0
                    print(f"  [{elapsed:.0f}s] frames={written:,}, rate={rate:.0f} fps, wraps={wrap_count}")
                    last_report = now

    finally:
        if gc_was_enabled:
            gc.enable()

    total_elapsed = time.perf_counter() - start_time
    return {
        'frames_written': written,
        'duration_actual': total_elapsed,
        'wrap_count': wrap_count,
        'max_pending': max_pending,
    }

# =============================================================================
# Verification
# =============================================================================

def verify_bin_file(filepath, num_frames=10):
    """Read and verify first N frames from .bin file.

    Returns: list of (frame_index, timestamp, magic_ok)
    """
    results = []
    try:
        with open(filepath, 'rb') as f:
            for i in range(num_frames):
                data = f.read(FRAME_SIZE)
                if len(data) < FRAME_SIZE:
                    break
                words = struct.unpack('<28I', data)
                magic_ok = (words[0] == MAGIC_LOW and words[1] == MAGIC_HIGH)
                ts = words[2]
                results.append((i, ts, magic_ok))
    except Exception as e:
        print(f"  Verify error: {e}")
    return results

# =============================================================================
# Main Entry: record()
# =============================================================================

def record(duration_sec, output_file=None,
           sample_rate=DEFAULT_SAMPLE_RATE,
           lower_bw=DEFAULT_LOWER_BW, upper_bw=DEFAULT_UPPER_BW,
           phase_select=DEFAULT_PHASE_SELECT,
           num_bd=DEFAULT_NUM_BD,
           bitstream_path=DEFAULT_BITSTREAM):
    """Complete recording flow: connect -> init -> record -> save -> cleanup.

    Args:
        duration_sec: recording duration in seconds
        output_file: .bin output path (auto-generated if None)
        sample_rate: sampling rate in Hz (default 30000)
        lower_bw: lower cutoff frequency in Hz (default 1.0)
        upper_bw: upper cutoff frequency in Hz (default 7500.0)
        phase_select: MISO phase compensation 0-9 (default 3)
        num_bd: number of BD descriptors (default 65536, must be power of 2)
        bitstream_path: Overlay bitstream path

    Returns:
        dict: {output_file, frames_written, duration_actual, sample_rate, errors}
    """
    from pynq import Overlay, allocate

    if output_file is None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{DEFAULT_OUTPUT_DIR}/recording_{ts_str}.bin"

    print(f"=== RHS2116 Recording Driver ===")
    print(f"  Duration: {duration_sec}s, Sample rate: {sample_rate} Hz")
    print(f"  BW: {lower_bw}-{upper_bw} Hz, Phase: {phase_select}")
    print(f"  Output: {output_file}")
    print(f"  BD count: {num_bd}")

    mem = None
    ol = None
    mmio = None
    dma_mmio = None

    try:
        # =====================================================================
        # 1. Connect
        # =====================================================================
        print("\n[1/7] Loading overlay...")
        ol = Overlay(bitstream_path)
        spi = ol.rhs2000_spi_ip_0
        dma = ol.axi_dma_0
        mmio = spi.mmio
        dma_mmio = dma.mmio

        board_id = mmio.read(ADDR_BOARD_ID)
        print(f"  Board ID: {board_id}", end="")
        if board_id == 900:
            print(" OK")
        else:
            print(f" UNEXPECTED (expected 900)")

        # =====================================================================
        # 2. Init RHS2116
        # =====================================================================
        print("[2/7] Initializing RHS2116...")

        # Stop SPI
        mmio.write(ADDR_CTRL_REG, 0x00)
        time.sleep(0.01)

        # Set MISO phase
        mmio.write(ADDR_MISO_DELAY, phase_select)

        # Generate and write init sequence to all 4 BRAM slots
        init_cmds = generate_init_sequence(sample_rate, lower_bw, upper_bw)
        n_init_cmds = sum(1 for c in init_cmds if c != DUMMY_CMD())
        print(f"  Init commands: {n_init_cmds}")

        for slot in range(4):
            write_aux_bram_slot(mmio, slot, init_cmds)

        # Set AUX_CMD_CFG (CRITICAL!)
        max_idx = min(n_init_cmds + 5, 127)
        loop_idx = max_idx  # Stay at DUMMY after init completes
        aux_cmd_cfg = ((loop_idx & 0x7F) << 16) | (max_idx & 0x7F)
        mmio.write(ADDR_AUX_CMD_CFG, aux_cmd_cfg)
        print(f"  AUX_CMD_CFG: max={max_idx}, loop={loop_idx}")

        # Execute init: SPI start -> wait -> stop
        mmio.write(ADDR_CTRL_REG, 0x02)  # spi_run = 1
        mmio.write(ADDR_TRIGGER_REG, 0x01)  # trigger
        time.sleep(0.1)
        mmio.write(ADDR_CTRL_REG, 0x00)  # stop
        time.sleep(0.01)

        init_ts = mmio.read(ADDR_TIMESTAMP)
        print(f"  Init done, timestamp={init_ts}")

        # =====================================================================
        # 3. Switch to Recording mode
        # =====================================================================
        print("[3/7] Switching BRAM to recording mode...")
        dummy_cmds = [DUMMY_CMD()] * 128
        for slot in range(4):
            write_aux_bram_slot(mmio, slot, dummy_cmds)
        print("  All 4 slots filled with DUMMY")

        # =====================================================================
        # 4. Setup SG Mode DMA
        # =====================================================================
        print("[4/7] Setting up SG Mode DMA...")
        total_bd_bytes = num_bd * BD_SIZE
        total_data_bytes = num_bd * FRAME_SIZE
        total_bytes = total_bd_bytes + total_data_bytes
        mem = allocate(shape=(total_bytes // 4,), dtype=np.uint32)
        mem[:] = 0

        bd_base = mem.physical_address
        setup_sg_dma(dma_mmio, mem, bd_base, num_bd)
        print(f"  Allocated {total_bytes / 1024 / 1024:.1f} MB, BD ring ready")

        # =====================================================================
        # 5. Record (Moving Tail)
        # =====================================================================
        print(f"[5/7] Recording {duration_sec}s...")
        mmio.write(ADDR_CTRL_REG, 0x02)  # spi_run = 1
        mmio.write(ADDR_TRIGGER_REG, 0x01)  # trigger

        stats = _recording_loop(mmio, dma_mmio, mem, bd_base, num_bd,
                                duration_sec, output_file)

        # Stop SPI
        mmio.write(ADDR_CTRL_REG, 0x00)
        time.sleep(0.01)

        frames = stats['frames_written']
        elapsed = stats['duration_actual']
        rate = frames / elapsed if elapsed > 0 else 0
        print(f"  Done: {frames:,} frames in {elapsed:.2f}s ({rate:.0f} fps)")

        # =====================================================================
        # 6. Cleanup DMA
        # =====================================================================
        print("[6/7] Cleaning up...")
        dma_mmio.write(S2MM_DMACR, 0x00000004)  # Reset DMA
        time.sleep(0.01)
        mem.freebuffer()
        mem = None
        gc.collect()
        print("  DMA stopped, memory freed")

        # =====================================================================
        # 7. Verify
        # =====================================================================
        print("[7/7] Verifying output file...")
        results = verify_bin_file(output_file)
        all_ok = all(r[2] for r in results)
        if results:
            ts_list = [r[1] for r in results]
            print(f"  First {len(results)} frames: magic={'OK' if all_ok else 'FAIL'}, ts={ts_list}")
        else:
            print("  No frames to verify")

        import os
        file_size = os.path.getsize(output_file)
        expected_frames = file_size // FRAME_SIZE
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB ({expected_frames:,} frames)")

        print(f"\n=== Recording Complete ===")
        print(f"  Output: {output_file}")

        return {
            'output_file': output_file,
            'frames_written': frames,
            'duration_actual': elapsed,
            'sample_rate': sample_rate,
            'wrap_count': stats['wrap_count'],
            'max_pending': stats['max_pending'],
            'magic_ok': all_ok,
        }

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

    finally:
        # Ensure cleanup even on error
        try:
            if mmio is not None:
                mmio.write(ADDR_CTRL_REG, 0x00)
        except:
            pass
        try:
            if dma_mmio is not None:
                dma_mmio.write(S2MM_DMACR, 0x00000004)
        except:
            pass
        try:
            if mem is not None:
                mem.freebuffer()
        except:
            pass
        gc.collect()
