"""
Phone/PC side: receives the band's raw data stream, detects dropped packets, saves the raw
signal, and computes the per-second features (the same ones tinyml_model.py trains on).
A real iOS/Android app would do this with CoreBluetooth / Android BLE and the same packet layout.

  python ble_client.py --simulate --label fatigue    no hardware needed
  python ble_client.py --scan
  python ble_client.py --label normal                 connect to first PowerBand and record
Outputs: session_raw.npz (emg, t_ms), session_status.csv, session_epochs.csv (8 features/sec + label)
"""
import argparse, asyncio, csv, random
import numpy as np
import ble_protocol as P
from powerthru_common import extract_emg_features, synth_emg, FEATURES


class Session:
    def __init__(self, label=""):
        self.label = label
        self.emg, self.t0 = [], None          # raw samples (int16 counts)
        self.expect_seq, self.dropped, self.packets = None, 0, 0
        self.status, self.epochs = [], []
        self.last_status = None
        self.next_epoch = P.EMG_FS            # samples needed for the next 1-s epoch

    def on_emg(self, raw):
        try:
            d = P.parse_emg(raw)
        except ValueError as e:
            print("dropped bad EMG packet:", e); return
        if self.expect_seq is not None and d["seq"] != self.expect_seq:
            lost = (d["seq"] - self.expect_seq) & 0xFFFF
            self.dropped += lost
            # keep the time axis honest: pad lost batches with NaN rather than silently splicing
            self.emg.extend([np.nan] * lost * P.EMG_BATCH)
        self.expect_seq = (d["seq"] + 1) & 0xFFFF
        self.packets += 1
        if self.t0 is None: self.t0 = d["t_ms"]
        self.emg.extend(d["samples"])
        while len(self.emg) >= self.next_epoch:
            self._epoch(np.array(self.emg[self.next_epoch - P.EMG_FS:self.next_epoch], dtype=float))
            self.next_epoch += P.EMG_FS

    def _epoch(self, x):
        if np.isnan(x).any():                 # skip epochs with gaps instead of faking features
            return
        rms, mdf, zc, wl = extract_emg_features(x, P.EMG_FS)
        s = self.last_status or {}
        row = [rms, mdf, zc, wl, s.get("acc_mean_g", np.nan), s.get("acc_sd_g", np.nan),
               s.get("temp_c", np.nan), s.get("sweat_us", np.nan)]
        self.epochs.append(row)
        print(f"  t={len(self.epochs):4d}s rms={rms:7.1f} mdf={mdf:5.1f}Hz  "
              f"batt={s.get('batt_pct', '?')}%  dropped={self.dropped}")

    def on_status(self, raw):
        try:
            self.last_status = P.parse_status(raw)
            self.status.append(self.last_status)
        except ValueError as e:
            print("dropped bad STATUS packet:", e)

    def save(self):
        np.savez_compressed("session_raw.npz", emg=np.array(self.emg, dtype=np.float32), fs=P.EMG_FS)
        with open("session_epochs.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(FEATURES + ["label"])
            for r in self.epochs: w.writerow(list(r) + [self.label])
        with open("session_status.csv", "w", newline="") as f:
            if self.status:
                w = csv.DictWriter(f, fieldnames=list(self.status[0])); w.writeheader(); w.writerows(self.status)
        print(f"saved {len(self.emg)} EMG samples, {len(self.epochs)} epochs, {len(self.status)} status; "
              f"{self.dropped} samples-batches dropped of {self.packets + self.dropped}")


# ------------------------------------------------------------------ simulator
async def simulated_band(sess, seconds=20, drop_prob=0.01, delay=0.0):
    rng = np.random.default_rng(0)
    seq = st_seq = 0
    for sec in range(seconds):
        phase = sec / seconds
        sig = synth_emg(P.EMG_FS, P.EMG_FS, 80 + 120 * phase, 105 - 35 * phase, rng)
        for b in range(P.EMG_FS // P.EMG_BATCH):
            pkt = P.pack_emg(seq, sec * 1000 + b * P.EMG_BATCH,
                             [int(v) for v in sig[b * P.EMG_BATCH:(b + 1) * P.EMG_BATCH]])
            seq += 1
            if random.random() > drop_prob:      # emulate occasional BLE packet loss
                sess.on_emg(pkt)
        sess.on_status(P.pack_status(st_seq, sec * 1000, 1 + 0.3 * phase, 0.1, 33 + phase, 2 + 3 * phase, 96 - sec // 10))
        st_seq += 1
        await asyncio.sleep(delay)


# ------------------------------------------------------------------ real BLE
async def run_live(address, sess):
    from bleak import BleakScanner, BleakClient
    dev = (address and await BleakScanner.find_device_by_address(address, timeout=10)) \
        or await BleakScanner.find_device_by_name(P.DEVICE_NAME, timeout=10)
    if not dev:
        print("PowerBand not found. Powered on? Already connected to another phone?"); return
    gone = asyncio.Event()
    async with BleakClient(dev, disconnected_callback=lambda c: gone.set()) as c:
        print("connected:", dev.address, "| MTU:", getattr(c, "mtu_size", "?"))
        await c.start_notify(P.EMG_UUID, lambda _, d: sess.on_emg(d))
        await c.start_notify(P.STATUS_UUID, lambda _, d: sess.on_status(d))
        await c.write_gatt_char(P.CONTROL_UUID, bytes([P.CMD_START]), response=True)
        print("streaming... Ctrl+C to stop")
        try:
            await gone.wait()
        finally:
            if c.is_connected:
                await c.write_gatt_char(P.CONTROL_UUID, bytes([P.CMD_STOP]), response=True)


async def scan():
    from bleak import BleakScanner
    for d in await BleakScanner.discover(timeout=8):
        if d.name and "Power" in d.name: print(d.address, d.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true"); ap.add_argument("--scan", action="store_true")
    ap.add_argument("--address"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    s = Session(a.label)
    try:
        if a.scan: asyncio.run(scan())
        elif a.simulate: asyncio.run(simulated_band(s))
        else: asyncio.run(run_live(a.address, s))
    except KeyboardInterrupt:
        pass
    finally:
        if not a.scan: s.save()
