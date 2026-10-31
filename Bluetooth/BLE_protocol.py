"""
PowerThru BLE protocol v2 -- the band is a DATA PIPE. No model on the device: it streams
raw sensor data and the phone app does all feature extraction / analysis.
Pure Python (struct only), so the same file runs on the phone/PC and in MicroPython.

GATT: one custom service, three characteristics
  EMG     (notify): batches of raw sEMG samples (int16), e.g. 50 samples every 50 ms @ 1 kHz
  STATUS  (notify): 1 Hz slow sensors (accel mean/std, skin temp, sweat conductance, battery)
  CONTROL (write) : app -> band commands (start/stop streaming, set sample rate, ping)

Throughput: 1 kHz x 16-bit = 2 kB/s (~16 kbps), well within BLE. Batching 50 samples = 7 + 100 + 1
= 108-byte packets, so the app must request a larger ATT MTU (>= 111). Android/iOS negotiate
185-247 normally. If MTU can't be raised, set EMG_BATCH to 6 (fits the default 20-byte payload).
Replace the UUIDs with your own (uuid.uuid4()) before shipping.
"""
import struct

SERVICE_UUID = "6e5a0001-b5a3-f393-e0a9-e50e24dcca9e"
EMG_UUID     = "6e5a0002-b5a3-f393-e0a9-e50e24dcca9e"
STATUS_UUID  = "6e5a0004-b5a3-f393-e0a9-e50e24dcca9e"
CONTROL_UUID = "6e5a0003-b5a3-f393-e0a9-e50e24dcca9e"
DEVICE_NAME = "PowerBand"

EMG_FS = 1000
EMG_BATCH = 50
FLAG_SENSOR_FAULT, FLAG_LOW_BATT, FLAG_STREAMING = 1, 2, 4
CMD_START, CMD_STOP, CMD_SET_RATE, CMD_PING = 0x01, 0x02, 0x03, 0x04


def crc8(data, poly=0x07):
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


# ---- EMG batch: seq u16 | t_ms u32 (time of first sample) | n u8 | n x int16 | crc8
def pack_emg(seq, t_ms, samples):
    body = struct.pack("<HIB", seq & 0xFFFF, t_ms & 0xFFFFFFFF, len(samples)) + \
           struct.pack("<%dh" % len(samples), *samples)
    return body + bytes([crc8(body)])


def parse_emg(pkt):
    pkt = bytes(pkt)
    if len(pkt) < 8 or crc8(pkt[:-1]) != pkt[-1]:
        raise ValueError("bad EMG packet (length/CRC)")
    seq, t_ms, n = struct.unpack("<HIB", pkt[:7])
    if len(pkt) != 7 + 2 * n + 1:
        raise ValueError("EMG length mismatch")
    return {"seq": seq, "t_ms": t_ms, "samples": list(struct.unpack("<%dh" % n, pkt[7:7 + 2 * n]))}


# ---- STATUS (1 Hz): seq u16 | t_ms u32 | acc_mean u16 (mg) | acc_sd u16 (mg) | temp i16 (0.01C)
#                     | sweat u16 (0.01 uS) | batt u8 | flags u8 | crc8   = 17 bytes
_ST = "<HIHHhHBB"


def pack_status(seq, t_ms, acc_mean_g, acc_sd_g, temp_c, sweat_us, batt_pct, flags=0):
    body = struct.pack(_ST, seq & 0xFFFF, t_ms & 0xFFFFFFFF, int(acc_mean_g * 1000),
                       int(acc_sd_g * 1000), int(round(temp_c * 100)), int(sweat_us * 100),
                       batt_pct, flags)
    return body + bytes([crc8(body)])


def parse_status(pkt):
    pkt = bytes(pkt)
    if len(pkt) != struct.calcsize(_ST) + 1 or crc8(pkt[:-1]) != pkt[-1]:
        raise ValueError("bad STATUS packet (length/CRC)")
    seq, t, am, asd, temp, sw, batt, fl = struct.unpack(_ST, pkt[:-1])
    return {"seq": seq, "t_ms": t, "acc_mean_g": am / 1000, "acc_sd_g": asd / 1000,
            "temp_c": temp / 100, "sweat_us": sw / 100, "batt_pct": batt, "flags": fl}


def pack_set_rate(hz):
    return bytes([CMD_SET_RATE]) + struct.pack("<H", hz)
