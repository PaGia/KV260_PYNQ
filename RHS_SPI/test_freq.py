#!/usr/bin/env python3
"""
CDC Fix後記錄功能驗證測試
記錄信號並分析頻率
"""

import sys
sys.path.insert(0, '.')
from jupyter_remote import JupyterClient
import numpy as np
import time

def test_recording_with_cdc():
    """執行記錄功能測試並分析頻率"""
    
    client = JupyterClient(
        base_url='http://120.126.83.227:9090',
        password='xilinx'
    )

    if not client.login():
        print('❌ 無法連接到 Jupyter')
        return False

    print('\n=== CDC修復後記錄功能驗證測試 ===')
    client.find_or_create_session('RHS_SPI/test_recording_cdc.ipynb')

    # 記錄測試程式碼
    code = '''
import os
os.chdir("/root/jupyter_notebooks/RHS_SPI")

from pynq import Overlay, allocate
import numpy as np
import time
from datetime import datetime

# 載入 overlay
ol = Overlay("rhs2116_system.bit")
mmio = ol.rhs2000_spi_ip_0.mmio
dma = ol.axi_dma_0

print("🎤 CDC修復後記錄功能測試")
print("=" * 60)

# 系統檢查
board_id = mmio.read(0xB0)
status = mmio.read(0xA0)
print(f"Board ID: {board_id}")
print(f"Status: 0x{status:08X}")

# RHS2116初始化
print("\\n📝 初始化RHS2116...")

def WRITE_CMD(reg, data):
    return 0x80000000 | ((reg & 0xFF) << 16) | (data & 0xFFFF)

def DUMMY_CMD():
    return 0xFFFFFFFF

def CLEAR_CMD():
    return 0x6A000000

def write_aux_bram(mmio, slot_id, index, data):
    mmio.write(0x84, index & 0x7F)
    mmio.write(0x88, data)
    mmio.write(0x8C, (slot_id & 0x03) | 0x100)
    mmio.write(0x8C, slot_id & 0x03)

# 初始化命令序列
init_cmds = [
    CLEAR_CMD(),
    WRITE_CMD(0, 0x0820),   # ADC buffer bias
    WRITE_CMD(1, 0x00D1),   # twoscomp=1
    WRITE_CMD(4, 0x0015),   # Upper BW
    WRITE_CMD(5, 0x0017),   # Upper BW
    WRITE_CMD(7, 0x0044),   # Lower BW = 1 Hz
    WRITE_CMD(8, 0xFFFF),   # All amps on
]

# 填充到128個命令
while len(init_cmds) < 128:
    init_cmds.append(DUMMY_CMD())

# 寫入所有4個slot
for slot in range(4):
    for i, cmd in enumerate(init_cmds):
        write_aux_bram(mmio, slot, i, cmd)

print("初始化命令已寫入BRAM")

# 設定MISO相位 (RHS2116通常需要phase=3)
mmio.write(0x0C, 3)
print("MISO Phase = 3")

# 執行初始化
mmio.write(0x00, 0x02)
mmio.write(0x80, 0x01)
time.sleep(0.1)
mmio.write(0x00, 0x00)
print("初始化完成")

# 切換到DUMMY命令進行記錄
dummy_cmds = [DUMMY_CMD()] * 128
for slot in range(4):
    for i, cmd in enumerate(dummy_cmds):
        write_aux_bram(mmio, slot, i, cmd)

print("\\n📊 開始記錄...")

# DMA設定
S2MM_DMACR = 0x30
S2MM_DMASR = 0x34
S2MM_CURDESC = 0x38
S2MM_CURDESC_MSB = 0x3C
S2MM_TAILDESC = 0x40
S2MM_TAILDESC_MSB = 0x44

FRAME_SIZE = 112
BD_SIZE = 64
NUM_BD = 4096  # 減少BD數量以加快測試
RECORDING_SECONDS = 5  # 記錄5秒

# 分配記憶體
total_bd_bytes = NUM_BD * BD_SIZE
total_data_bytes = NUM_BD * FRAME_SIZE
total_bytes = total_bd_bytes + total_data_bytes

mem = allocate(shape=(total_bytes // 4,), dtype=np.uint32)
mem[:] = 0

bd_base = mem.physical_address
data_base = bd_base + total_bd_bytes

# 初始化BD Ring
bd_words = BD_SIZE // 4
for i in range(NUM_BD):
    bd_offset = i * bd_words
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

# 設定DMA
dma.mmio.write(S2MM_DMACR, 0x00000004)  # Reset
time.sleep(0.01)
dma.mmio.write(S2MM_CURDESC, bd_base & 0xFFFFFFFF)
dma.mmio.write(S2MM_CURDESC_MSB, 0)
dma.mmio.write(S2MM_DMACR, 0x00000001)  # Run
time.sleep(0.001)
tail_bd = bd_base + (NUM_BD - 1) * BD_SIZE
dma.mmio.write(S2MM_TAILDESC, tail_bd & 0xFFFFFFFF)
dma.mmio.write(S2MM_TAILDESC_MSB, 0)

# 啟動SPI記錄
mmio.write(0x10, 0x01)  # DATA_STREAM_EN
mmio.write(0x00, 0x02)  # SPI_RUN
mmio.write(0x80, 0x01)  # TRIGGER

print(f"記錄中... {RECORDING_SECONDS}秒")

# 收集資料
frame_words = FRAME_SIZE // 4
data_base_words = (bd_base + NUM_BD * BD_SIZE - mem.physical_address) // 4
all_frames = []
read_idx = 0
start_time = time.perf_counter()

while time.perf_counter() - start_time < RECORDING_SECONDS:
    # 檢查BD完成
    for check_idx in range(32):  # 每次檢查32個BD
        bd_idx = (read_idx + check_idx) % NUM_BD
        bd_offset = bd_idx * bd_words
        
        if (mem[bd_offset + 7] >> 31) & 1:  # CMPLT bit
            # 讀取資料
            data_offset = data_base_words + bd_idx * frame_words
            frame_data = mem[data_offset:data_offset + frame_words].copy()
            all_frames.append(frame_data)
            
            # 清除BD status
            mem[bd_idx * bd_words + 7] = 0
            
            # 更新tail
            new_tail_addr = bd_base + bd_idx * BD_SIZE
            dma.mmio.write(S2MM_TAILDESC, new_tail_addr & 0xFFFFFFFF)
            
            read_idx = (bd_idx + 1) % NUM_BD
        else:
            break
    
    time.sleep(0.001)

# 停止SPI
mmio.write(0x00, 0x00)

elapsed = time.perf_counter() - start_time
frames_collected = len(all_frames)
print(f"\\n記錄完成: {frames_collected} frames in {elapsed:.2f}秒")
print(f"採樣率: {frames_collected/elapsed:.1f} fps")

# 分析資料
print("\\n🔍 分析記錄資料...")

if frames_collected > 100:
    # 提取ADC資料
    MAGIC_LOW = 0x49712F0B
    MAGIC_HIGH = 0x8D542C8A
    
    adc_data = []
    for frame in all_frames[:min(30000, frames_collected)]:  # 分析前1秒資料
        # 驗證Magic Number
        if frame[0] == MAGIC_LOW and frame[1] == MAGIC_HIGH:
            # 提取Ch0資料 (beat 2, low word)
            ch0_raw = frame[4]  # Beat 2, Ch0
            ch0_adc = (ch0_raw >> 16) & 0xFFFF
            
            # Two's complement轉換
            if ch0_adc & 0x8000:
                ch0_signed = ch0_adc - 65536
            else:
                ch0_signed = ch0_adc
            
            adc_data.append(ch0_signed)
    
    adc_array = np.array(adc_data, dtype=np.float32)
    print(f"有效樣本: {len(adc_array)}")
    print(f"平均值: {np.mean(adc_array):.1f}")
    print(f"標準差: {np.std(adc_array):.1f}")
    print(f"峰峰值: {np.max(adc_array) - np.min(adc_array):.1f}")
    
    # FFT分析
    if len(adc_array) > 1000:
        # 移除DC
        adc_array = adc_array - np.mean(adc_array)
        
        # 計算FFT
        fft_data = np.fft.rfft(adc_array)
        fft_mag = np.abs(fft_data)
        fft_freq = np.fft.rfftfreq(len(adc_array), 1/30000)  # 30kHz採樣率
        
        # 找峰值頻率 (忽略DC和超低頻)
        freq_range = (fft_freq > 10) & (fft_freq < 5000)
        if np.any(freq_range):
            peak_idx = np.argmax(fft_mag[freq_range])
            peak_freq = fft_freq[freq_range][peak_idx]
            peak_mag = fft_mag[freq_range][peak_idx]
            
            # 計算SNR
            signal_power = peak_mag ** 2
            noise_idx = (fft_freq > 10) & (fft_freq < 5000) & (np.abs(fft_freq - peak_freq) > 50)
            if np.any(noise_idx):
                noise_power = np.mean(fft_mag[noise_idx] ** 2)
                snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            else:
                snr_db = 0
            
            print(f"\\n🎯 檢測到主頻率: {peak_freq:.1f} Hz")
            print(f"   振幅: {peak_mag:.1f}")
            print(f"   SNR: {snr_db:.1f} dB")
            
            # 檢查諧波
            harmonics = []
            for h in [2, 3, 4, 5]:
                h_freq = peak_freq * h
                if h_freq < 5000:
                    h_idx = np.argmin(np.abs(fft_freq - h_freq))
                    if fft_mag[h_idx] > peak_mag * 0.05:  # 諧波大於基頻的5%
                        harmonics.append(h)
            
            if harmonics:
                print(f"   檢測到諧波: {harmonics}")
        else:
            print("\\n未檢測到明顯頻率成分")
    
    # 保存一小段原始資料供進一步分析
    np.save('/root/jupyter_notebooks/RHS_SPI/cdc_test_recording.npy', adc_array[:30000])
    print("\\n資料已保存到 cdc_test_recording.npy")

else:
    print("❌ 記錄資料不足")

del mem
print("\\n✅ CDC修復後記錄功能測試完成")
    '''

    success, outputs = client.execute_code(code, timeout=120)
    if success:
        for output in outputs:
            if 'text' in output:
                print(output['text'])
    else:
        print('❌ 記錄測試失敗')
        for output in outputs:
            if 'text' in output:
                print(output['text'])
        return False

    return True

if __name__ == '__main__':
    success = test_recording_with_cdc()
    if success:
        print('\n🎯 記錄測試完成，請查看頻率分析結果')
    else:
        print('\n💥 記錄測試失敗')
        sys.exit(1)