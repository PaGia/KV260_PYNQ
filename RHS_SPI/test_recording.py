#!/usr/bin/env python3
"""
RHS2116 Recording Test (T13)
============================
根據 CLAUDE.md T13 測試計畫執行 RHS2116 基本 Recording 測試

測試項目:
1. 使用正確的頻寬設定初始化 RHS2116
2. 執行 Recording 並收集 ADC 資料
3. 驗證資料非固定值

重要: Lower Cutoff Frequency (Reg 7) 必須設為 1 Hz 才能錄製 7 Hz 訊號

Issue #11 修復: 使用 "Raw dump + 事後分析" 方案
- fast_raw_dump(): 高速原始資料儲存 (僅記憶體複製)
- offline_analysis(): 離線 NumPy 向量化分析
"""

import time
import math
import numpy as np

# ============================================================
# 頻寬計算函數 (移植自 Intan rhsregisters.c)
# ============================================================

def rH1_from_upper_bandwidth(upper_bandwidth):
    """計算上限頻率對應的 RH1 電阻值 (ohms)"""
    log10f = math.log10(upper_bandwidth)
    return 0.9370 * math.pow(10, 0.04767 * log10f * log10f - 1.1892 * log10f + 8.0968)

def rH2_from_upper_bandwidth(upper_bandwidth):
    """計算上限頻率對應的 RH2 電阻值 (ohms)"""
    log10f = math.log10(upper_bandwidth)
    return 1.0191 * math.pow(10, 0.03383 * log10f * log10f - 1.0821 * log10f + 8.1009)

def rL_from_lower_bandwidth(lower_bandwidth):
    """計算下限頻率對應的 RL 電阻值 (ohms)"""
    log10f = math.log10(lower_bandwidth)
    if lower_bandwidth < 4.0:
        return 1.0061 * math.pow(10, 0.065 * log10f * log10f - 0.9043 * log10f + 5.7186)
    else:
        return 1.0061 * math.pow(10, -0.04659 * log10f * log10f - 0.4596 * log10f + 5.5880)

def calc_upper_bandwidth_regs(upper_bandwidth):
    """計算上限頻率暫存器值 (Reg 4 和 5)"""
    RH1Base, RH1Dac1Unit, RH1Dac2Unit = 2200.0, 600.0, 29400.0
    RH2Base, RH2Dac1Unit, RH2Dac2Unit = 8700.0, 763.0, 38400.0
    RH1Dac1Steps, RH1Dac2Steps = 63, 31
    RH2Dac1Steps, RH2Dac2Steps = 63, 31

    if upper_bandwidth > 30000.0:
        upper_bandwidth = 30000.0

    # 計算 RH1 DAC 值
    rH1_target = rH1_from_upper_bandwidth(upper_bandwidth)
    rH1_sel1, rH1_sel2 = 0, 0
    rH1_actual = RH1Base

    for _ in range(RH1Dac2Steps):
        if rH1_actual < rH1_target - (RH1Dac2Unit - RH1Dac1Unit / 2):
            rH1_actual += RH1Dac2Unit
            rH1_sel2 += 1
    for _ in range(RH1Dac1Steps):
        if rH1_actual < rH1_target - (RH1Dac1Unit / 2):
            rH1_actual += RH1Dac1Unit
            rH1_sel1 += 1

    # 計算 RH2 DAC 值
    rH2_target = rH2_from_upper_bandwidth(upper_bandwidth)
    rH2_sel1, rH2_sel2 = 0, 0
    rH2_actual = RH2Base

    for _ in range(RH2Dac2Steps):
        if rH2_actual < rH2_target - (RH2Dac2Unit - RH2Dac1Unit / 2):
            rH2_actual += RH2Dac2Unit
            rH2_sel2 += 1
    for _ in range(RH2Dac1Steps):
        if rH2_actual < rH2_target - (RH2Dac1Unit / 2):
            rH2_actual += RH2Dac1Unit
            rH2_sel1 += 1

    reg4 = (rH1_sel2 << 6) | rH1_sel1
    reg5 = (rH2_sel2 << 6) | rH2_sel1
    return reg4, reg5

def calc_lower_bandwidth_reg(lower_bandwidth):
    """計算下限頻率暫存器值 (Reg 6 或 7)"""
    RLBase = 3500.0
    RLDac1Unit, RLDac2Unit, RLDac3Unit = 175.0, 12700.0, 3000000.0
    RLDac1Steps, RLDac2Steps = 127, 63

    if lower_bandwidth > 1500.0:
        lower_bandwidth = 1500.0

    rL_target = rL_from_lower_bandwidth(lower_bandwidth)
    rL_sel1, rL_sel2, rL_sel3 = 0, 0, 0
    rL_actual = RLBase

    if lower_bandwidth < 0.15:
        rL_actual += RLDac3Unit
        rL_sel3 = 1

    for _ in range(RLDac2Steps):
        if rL_actual < rL_target - (RLDac2Unit - RLDac1Unit / 2):
            rL_actual += RLDac2Unit
            rL_sel2 += 1

    for _ in range(RLDac1Steps):
        if rL_actual < rL_target - (RLDac1Unit / 2):
            rL_actual += RLDac1Unit
            rL_sel1 += 1

    return (rL_sel3 << 13) | (rL_sel2 << 7) | rL_sel1

# ============================================================
# SPI 命令產生函數
# ============================================================

def CLEAR_CMD():
    """CLEAR 命令 - 初始化 ADC"""
    return 0x6A000000

def WRITE_CMD(reg_addr, data, u=False, m=False):
    """WRITE 命令 - 寫入暫存器"""
    cmd = 0x80000000 | ((reg_addr & 0xFF) << 16) | (data & 0xFFFF)
    if u: cmd |= 0x20000000
    if m: cmd |= 0x10000000
    return cmd

def READ_CMD(reg_addr, u=False, m=False):
    """READ 命令 - 讀取暫存器"""
    cmd = 0xC0000000 | ((reg_addr & 0xFF) << 16)
    if u: cmd |= 0x20000000
    if m: cmd |= 0x10000000
    return cmd

def DUMMY_CMD():
    """DUMMY 命令 - 無操作"""
    return 0xFFFFFFFF

# ============================================================
# 初始化命令序列產生
# ============================================================

def generate_init_sequence(sample_rate=30000, lower_bw=1.0, upper_bw=7500.0):
    """產生 RHS2116 完整初始化命令序列"""
    cmds = []

    # Step 1: Dummy 命令
    cmds.append(READ_CMD(255))
    cmds.append(READ_CMD(255))

    # Step 2: CLEAR 命令
    cmds.append(CLEAR_CMD())

    # Step 3: ADC Bias 設定 (Reg 0)
    adc_rate = 20 * sample_rate
    if adc_rate <= 120000:
        adc_buffer_bias, mux_bias = 32, 40
    elif adc_rate <= 350000:
        adc_buffer_bias, mux_bias = 4, 18
    elif adc_rate <= 525000:
        adc_buffer_bias, mux_bias = 3, 7
    else:
        adc_buffer_bias, mux_bias = 2, 4

    reg0_val = ((adc_buffer_bias & 0x3F) << 6) | (mux_bias & 0x3F)
    cmds.append(WRITE_CMD(0, reg0_val))

    # Step 4: ADC 輸出格式 (Reg 1)
    # Bit 位置:
    #   D[3:0] = DSP cutoff freq (1 = 啟用 DSP 高通濾波)
    #   D[4]   = DSPen (1 = 啟用 DSP 偏移移除)
    #   D[5]   = absmode (0 = 正常模式)
    #   D[6]   = twoscomp (1 = 二補數格式, 0 = offset binary)
    #   D[7]   = weak_miso (1 = CS 高時弱驅動)
    #
    # 0x00D1 = 0b 1101 0001
    #   D[3:0] = 0001 (DSP cutoff = 1)
    #   D[4]   = 1 (DSPen = 啟用)
    #   D[5]   = 0 (absmode = 關閉)
    #   D[6]   = 1 (twoscomp = 二補數) ← Issue #10 修復
    #   D[7]   = 1 (weak_miso = 啟用)
    reg1_val = 0x00D1  # weak_miso=1, twoscomp=1, DSPen=1, cutoff=1
    cmds.append(WRITE_CMD(1, reg1_val))

    # Step 5: 阻抗測試關閉 (Reg 2-3)
    cmds.append(WRITE_CMD(2, 0x0000))
    cmds.append(WRITE_CMD(3, 0x0080))

    # Step 6: 頻寬設定 (Reg 4-7) - 關鍵
    reg4, reg5 = calc_upper_bandwidth_regs(upper_bw)
    reg6 = calc_lower_bandwidth_reg(1000.0)  # RL_A: 1000 Hz
    reg7 = calc_lower_bandwidth_reg(lower_bw) # RL_B: 1 Hz

    print(f"[Bandwidth] Upper: {upper_bw} Hz -> Reg4=0x{reg4:04X}, Reg5=0x{reg5:04X}")
    print(f"[Bandwidth] Lower (RL_A): 1000 Hz -> Reg6=0x{reg6:04X}")
    print(f"[Bandwidth] Lower (RL_B): {lower_bw} Hz -> Reg7=0x{reg7:04X}")

    cmds.append(WRITE_CMD(4, reg4))
    cmds.append(WRITE_CMD(5, reg5))
    cmds.append(WRITE_CMD(6, reg6))
    cmds.append(WRITE_CMD(7, reg7))

    # Step 7: AC 放大器電源 (Reg 8)
    cmds.append(WRITE_CMD(8, 0xFFFF))

    # Step 8: 觸發暫存器初始化 (Reg 10, 12)
    cmds.append(WRITE_CMD(10, 0x0000))
    cmds.append(WRITE_CMD(12, 0x0000))  # 使用 RL_B (Reg 7)

    # Step 9: 刺激控制初始化 (Reg 32-38)
    cmds.append(WRITE_CMD(32, 0x0000))
    cmds.append(WRITE_CMD(33, 0x0000))
    cmds.append(WRITE_CMD(34, 0x0080))
    cmds.append(WRITE_CMD(35, 0x0000))
    cmds.append(WRITE_CMD(36, 0x0080))
    cmds.append(WRITE_CMD(37, 0x0000))
    cmds.append(WRITE_CMD(38, 0xFFFF))

    # Step 10: 刺激觸發暫存器
    cmds.append(WRITE_CMD(42, 0x0000))
    cmds.append(WRITE_CMD(44, 0x0000))
    cmds.append(WRITE_CMD(46, 0x0000))
    cmds.append(WRITE_CMD(48, 0x0000))

    # 填充 DUMMY
    while len(cmds) < 128:
        cmds.append(DUMMY_CMD())

    return cmds

# ============================================================
# BRAM 操作函數
# ============================================================

def write_aux_bram(mmio, slot_id, index, data):
    """寫入單一命令到 AUX BRAM"""
    ADDR_AUX_RAM_ADDR = 0x84
    ADDR_AUX_RAM_DATA = 0x88
    ADDR_AUX_RAM_BANK = 0x8C

    mmio.write(ADDR_AUX_RAM_ADDR, index & 0x7F)
    mmio.write(ADDR_AUX_RAM_DATA, data)
    mmio.write(ADDR_AUX_RAM_BANK, (slot_id & 0x03) | 0x100)
    mmio.write(ADDR_AUX_RAM_BANK, slot_id & 0x03)

def write_aux_bram_slot(mmio, slot_id, commands):
    """將命令序列寫入 AUX BRAM Slot"""
    for i, cmd in enumerate(commands[:128]):
        write_aux_bram(mmio, slot_id, i, cmd)

# ============================================================
# Issue #11 修復: Raw Dump + 離線分析
# ============================================================

def fast_raw_dump(mmio, dma, mem, duration_sec=1.0, num_bd=8192):
    """
    高速原始資料儲存 (Phase 1) - 等待完成版本

    核心策略:
    1. 使用足夠大的 BD Ring (num_bd >= target_frames)
    2. 只等待 DMA 完成，不在收集時處理資料
    3. SPI 停止後，一次性從 DMA 緩衝區提取所有資料

    Args:
        mmio: MMIO 物件
        dma: DMA 物件
        mem: 預分配的記憶體緩衝區 (包含 BD Ring + Data)
        duration_sec: 錄製時間 (秒)
        num_bd: BD Ring 大小 (必須 >= target_frames)

    Returns:
        (timestamps, adc_raw, frame_count, elapsed_time)
    """
    BD_SIZE = 64
    FRAME_SIZE = 112
    TARGET_FRAMES = int(30000 * duration_sec)

    # 確保 BD Ring 夠大
    if num_bd < TARGET_FRAMES:
        print(f"    WARNING: num_bd ({num_bd}) < target_frames ({TARGET_FRAMES})")
        print(f"    Data may be lost due to BD ring wrap-around")

    # BD Ring 參數
    bd_words = BD_SIZE // 4
    frame_words = FRAME_SIZE // 4
    total_bd_bytes = num_bd * BD_SIZE
    data_base_words = total_bd_bytes // 4

    start_time = time.perf_counter()

    # 等待指定時間 (讓 DMA 連續收集資料)
    time.sleep(duration_sec)

    elapsed = time.perf_counter() - start_time

    # 計算實際收集的 frame 數量 (掃描 BD status)
    # 注意: DMA 會從 BD[0] 開始連續寫入，所以找連續完成的 BD 數量
    collected = 0
    for i in range(min(num_bd, TARGET_FRAMES + 100)):  # 多掃描一些
        bd_offset = i * bd_words
        status = int(mem[bd_offset + 7])
        if (status >> 31) & 1:  # CMPLT bit
            collected += 1
        else:
            break  # 連續計數

    # 限制為目標數量
    collected = min(collected, TARGET_FRAMES)

    # 預分配結果陣列
    timestamps = np.zeros(collected, dtype=np.uint32)
    adc_raw = np.zeros((collected, 16), dtype=np.uint32)

    # Magic Number 常數
    MAGIC_LOW = 0x49712F0B
    MAGIC_HIGH = 0x8D542C8A

    # 一次性提取所有資料 (SPI 已停止，無時間壓力)
    valid_count = 0
    for i in range(collected):
        data_offset = data_base_words + i * frame_words

        # 驗證 Magic Number
        magic_l = int(mem[data_offset + 0])
        magic_h = int(mem[data_offset + 1])
        if magic_l != MAGIC_LOW or magic_h != MAGIC_HIGH:
            continue  # 跳過無效 frame

        # 提取 timestamp
        timestamps[valid_count] = mem[data_offset + 2]

        # 提取 ADC 原始值 (word[6:22])
        for ch in range(16):
            adc_raw[valid_count, ch] = mem[data_offset + 6 + ch]

        valid_count += 1

    # 裁剪到有效數量
    timestamps = timestamps[:valid_count]
    adc_raw = adc_raw[:valid_count]

    return timestamps, adc_raw, valid_count, elapsed


def offline_analysis(timestamps, adc_raw, frame_count):
    """
    離線分析原始資料 (Phase 2)

    將 fast_raw_dump 收集的原始資料轉換為分析格式。
    使用 NumPy 向量化操作，無時間壓力。

    Args:
        timestamps: fast_raw_dump 返回的 timestamp 陣列
        adc_raw: fast_raw_dump 返回的 ADC 原始值陣列 (uint32)
        frame_count: frame 數量

    Returns:
        dict 包含:
        - timestamps: numpy array of uint32
        - adc_data: dict {ch: numpy array of int16}
        - gap_count: int
    """
    if frame_count == 0:
        return {
            'timestamps': np.array([], dtype=np.uint32),
            'adc_data': {i: np.array([], dtype=np.int16) for i in range(16)},
            'gap_count': 0
        }

    # 計算 Timestamp 間隙
    if frame_count > 1:
        ts_diff = np.diff(timestamps)
        gap_count = int(np.sum(ts_diff != 1))
    else:
        gap_count = 0

    # 轉換 ADC 資料 (高 16 位，twoscomp 格式)
    # adc_raw shape: (frame_count, 16), dtype=uint32
    adc_data = {}
    for ch in range(16):
        # 提取高 16 位
        raw_high = (adc_raw[:, ch] >> 16).astype(np.uint16)
        # 轉換為 int16 (twoscomp)
        adc_data[ch] = raw_high.view(np.int16)

    return {
        'timestamps': timestamps,
        'adc_data': adc_data,
        'gap_count': gap_count
    }


# ============================================================
# Recording 測試主程式
# ============================================================

def run_recording_test(duration_sec=1.0):
    """
    執行 T13 Recording 測試

    Issue #11 修復版本: 使用 raw dump + 離線分析方案
    - Phase 1: fast_raw_dump() - 高速資料收集，不做解析
    - Phase 2: offline_analysis() - 事後 NumPy 向量化分析

    Args:
        duration_sec: 錄製時間 (秒)，預設 1 秒
    """
    from pynq import Overlay, allocate

    print("=" * 60)
    print("T13: RHS2116 Basic Recording Test (Issue #11 Fixed)")
    print("=" * 60)

    # 1. 載入 Overlay
    print("\n[1] Loading Overlay...")
    overlay = Overlay('/root/jupyter_notebooks/RHS_SPI/rhs2116_system.bit')
    mmio = overlay.rhs2000_spi_ip_0.mmio
    dma = overlay.axi_dma_0

    # 驗證 Board ID
    board_id = mmio.read(0xB0)
    print(f"    Board ID: {board_id} (expected: 900)")
    if board_id != 900:
        print("    ERROR: Board ID mismatch!")
        return False

    # 2. 設定 MISO Phase
    print("\n[2] Setting MISO Phase = 3...")
    mmio.write(0x0C, 3)

    # 3. 產生初始化命令
    print("\n[3] Generating initialization sequence...")
    init_cmds = generate_init_sequence(sample_rate=30000, lower_bw=1.0, upper_bw=7500.0)
    n_valid = sum(1 for c in init_cmds if c != DUMMY_CMD())
    print(f"    Valid commands: {n_valid}")

    # 4. 寫入 AUX BRAM
    print("\n[4] Writing to AUX BRAM (all 4 slots)...")
    for slot in range(4):
        write_aux_bram_slot(mmio, slot_id=slot, commands=init_cmds)
    print("    Done")

    # 5. 執行初始化
    print("\n[5] Executing initialization...")
    mmio.write(0x00, 0x00)
    time.sleep(0.01)
    mmio.write(0x00, 0x02)
    mmio.write(0x80, 0x01)

    init_time = (n_valid / 30000) + 0.05
    print(f"    Waiting {init_time*1000:.0f} ms...")
    time.sleep(init_time)

    ts_after_init = mmio.read(0xA8)
    print(f"    Timestamp after init: {ts_after_init}")

    mmio.write(0x00, 0x00)
    time.sleep(0.01)

    # 6. 切換到 Recording 模式 (DUMMY 命令)
    print("\n[6] Switching to Recording mode...")
    dummy_cmds = [DUMMY_CMD()] * 128
    for slot in range(4):
        write_aux_bram_slot(mmio, slot_id=slot, commands=dummy_cmds)
    print("    AUX BRAM filled with DUMMY commands")

    # 7. DMA 設定
    print("\n[7] Setting up DMA (SG Mode)...")

    NUM_BD = 65536  # 足夠容納 2 秒以上 (30000 frames @ 30 kHz)
    BD_SIZE = 64
    FRAME_SIZE = 112
    TARGET_FRAMES = int(30000 * duration_sec)

    total_bd_bytes = NUM_BD * BD_SIZE
    total_data_bytes = NUM_BD * FRAME_SIZE
    total_bytes = total_bd_bytes + total_data_bytes

    mem = allocate(shape=(total_bytes // 4,), dtype=np.uint32)
    mem[:] = 0

    bd_base = mem.physical_address
    data_base = bd_base + total_bd_bytes

    # 初始化 BD Ring
    for i in range(NUM_BD):
        bd_offset = (i * BD_SIZE) // 4
        next_bd_addr = bd_base + ((i + 1) % NUM_BD) * BD_SIZE
        buf_addr = data_base + i * FRAME_SIZE

        mem[bd_offset + 0] = next_bd_addr & 0xFFFFFFFF
        mem[bd_offset + 1] = 0
        mem[bd_offset + 2] = buf_addr & 0xFFFFFFFF
        mem[bd_offset + 3] = 0
        mem[bd_offset + 4] = 0
        mem[bd_offset + 5] = 0
        mem[bd_offset + 6] = FRAME_SIZE
        mem[bd_offset + 7] = 0

    # 設定 DMA
    dma.mmio.write(0x30, 0x00000004)  # Reset
    time.sleep(0.01)
    dma.mmio.write(0x38, bd_base & 0xFFFFFFFF)
    dma.mmio.write(0x3C, 0)
    dma.mmio.write(0x30, 0x00000001)  # Run
    time.sleep(0.001)

    tail_bd = bd_base + (NUM_BD - 1) * BD_SIZE
    dma.mmio.write(0x40, tail_bd & 0xFFFFFFFF)
    dma.mmio.write(0x44, 0)

    print(f"    BD Ring: {NUM_BD} entries")
    print(f"    Target frames: {TARGET_FRAMES}")

    # 8. 啟動 SPI 並執行 Raw Dump
    print(f"\n[8] Starting Recording ({duration_sec}s) - Raw Dump Phase...")

    # 清除所有 BD status (確保乾淨的起始狀態)
    bd_words = BD_SIZE // 4
    for i in range(NUM_BD):
        mem[i * bd_words + 7] = 0

    # 重新設定 DMA (確保從頭開始)
    dma.mmio.write(0x30, 0x00000004)  # Reset
    time.sleep(0.001)
    dma.mmio.write(0x38, bd_base & 0xFFFFFFFF)
    dma.mmio.write(0x3C, 0)
    dma.mmio.write(0x30, 0x00000001)  # Run
    time.sleep(0.001)
    dma.mmio.write(0x40, tail_bd & 0xFFFFFFFF)
    dma.mmio.write(0x44, 0)

    # 啟動 SPI
    mmio.write(0x00, 0x02)
    mmio.write(0x80, 0x01)

    # Phase 1: Raw Dump (立即複製關鍵資料)
    timestamps_raw, adc_raw, frame_count, elapsed = fast_raw_dump(
        mmio, dma, mem,
        duration_sec=duration_sec,
        num_bd=NUM_BD
    )

    # 停止 SPI
    mmio.write(0x00, 0x00)

    print(f"    Collected: {frame_count} frames in {elapsed:.2f}s")
    print(f"    Collection rate: {frame_count/elapsed:.1f} fps")
    print(f"    Expected: {TARGET_FRAMES} frames")
    print(f"    Efficiency: {100*frame_count/TARGET_FRAMES:.1f}%")

    # 9. 離線分析 (轉換資料格式)
    print("\n[9] Offline Analysis Phase...")
    analysis_start = time.perf_counter()

    result = offline_analysis(timestamps_raw, adc_raw, frame_count)

    analysis_time = time.perf_counter() - analysis_start
    print(f"    Analysis completed in {analysis_time*1000:.1f} ms")
    print(f"    Timestamp gaps: {result['gap_count']}")

    timestamps = result['timestamps']
    adc_data = result['adc_data']

    if len(timestamps) > 1:
        print(f"    Timestamp range: {timestamps[0]} - {timestamps[-1]}")

    # 10. 顯示 ADC 統計
    print("\n[10] Channel Statistics (ADC values, int16 twoscomp):")
    print("    " + "-" * 56)
    print(f"    {'Ch':>4} | {'Min':>8} | {'Max':>8} | {'Mean':>10} | {'StdDev':>8}")
    print("    " + "-" * 56)

    all_constant = True
    for ch in range(14):  # Ch0-13 (有效資料)
        arr = adc_data[ch]
        if len(arr) > 0:
            ch_min = int(np.min(arr))
            ch_max = int(np.max(arr))
            ch_mean = np.mean(arr)
            ch_std = np.std(arr)
            print(f"    {ch:>4} | {ch_min:>8} | {ch_max:>8} | {ch_mean:>10.1f} | {ch_std:>8.2f}")
            if ch_std > 1.0:
                all_constant = False

    # 11. 判斷結果
    print("\n" + "=" * 60)

    # 判斷是否資料丟失 (Issue #11 核心指標)
    data_loss = frame_count < TARGET_FRAMES * 0.99  # 允許 1% 誤差

    if data_loss:
        loss_rate = 100 * (1 - frame_count / TARGET_FRAMES)
        print(f"RESULT: DATA LOSS DETECTED ({loss_rate:.1f}%)")
        print(f"        Collected {frame_count}/{TARGET_FRAMES} frames")
    elif result['gap_count'] > 0:
        print(f"RESULT: TIMESTAMP GAPS DETECTED ({result['gap_count']})")
        print("        Data may have discontinuities")
    elif all_constant:
        print("RESULT: FAIL - All channels show constant values")
        print("        This suggests initialization or bandwidth issue")
    else:
        print("RESULT: PASS - Recording successful, no data loss")
        # 額外分析 Ch0
        if len(adc_data[0]) > 0:
            arr = adc_data[0]
            print(f"\n    Ch0 details:")
            print(f"      First 10 values: {arr[:10].tolist()}")
            print(f"      Last 10 values: {arr[-10:].tolist()}")

    print("=" * 60)

    mem.freebuffer()
    return not data_loss and result['gap_count'] == 0

if __name__ == "__main__":
    run_recording_test()
