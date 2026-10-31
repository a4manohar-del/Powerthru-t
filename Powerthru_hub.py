#!/usr/bin/env python3
"""
PowerThru Hub -- ONE script: band data in -> fast processing -> iPhone app out.

  band --BLE--> [this script on a laptop/Pi]: checks packets, extracts features,
                applies the user's baseline, scores fatigue/overexertion risk
            --Wi-Fi (HTTP + Server-Sent Events)--> iPhone app, once per second

  python powerthru_hub.py --simulate                 no hardware: fake band, full pipeline
  python powerthru_hub.py                            connect to the first "PowerBand" over BLE
  python powerthru_hub.py --profile user_profile.json --token SECRET --log session.csv --label fatigue

iPhone app endpoints (same Wi-Fi network; URL is printed at startup):
  GET /stream?token=...   live JSON event every second (SSE; read with URLSession.bytes lines)
  GET /latest?token=...   most recent JSON event
  GET /health             {"ok": true}   (no token)
Event JSON: t, state (calibrating|normal|fatigue|high_risk), risk 0-100, features{...},
            z{...} (vs. user's baseline), status{acc,temp,sweat,batt}, dropped_batches

Dependencies: numpy (+ bleak only for real BLE). Everything else is standard library.

Optional trained model: save an MLP as .npz (keys W0,b0,W1,b1,...; x@W+b, ReLU between layers,
softmax output over [normal, fatigue, high_risk]). From scikit-learn:
    np.savez("powerthru_mlp.npz", **{f"{k}{i}": a for i,(w,b) in enumerate(zip(m.coefs_, m.intercepts_))
                                      for k,a in (("W",w),("b",b))})
then run with --weights powerthru_mlp.npz. Without it, a simple rule-based score is used, which is
an UNCALIBRATED PLACEHOLDER (not validated for cramp or overexertion detection).
"""
import argparse, asyncio, csv, json, queue, random, socket, struct, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np

# ------------------------------------------------------------------ constants / protocol
FS, BATCH = 1000, 50
SERVICE_UUID = "6e5a0001-b5a3-f393-e0a9-e50e24dcca9e"
EMG_UUID     = "6e5a0002-b5a3-f393-e0a9-e50e24dcca9e"
CONTROL_UUID = "6e5a0003-b5a3-f393-e0a9-e50e24dcca9e"
STATUS_UUID  = "6e5a0004-b5a3-f393-e0a9-e50e24dcca9e"
DEVICE_NAME, CMD_START, CMD_STOP = "PowerBand", 0x01, 0x02
FEATURES = ["emg_rms", "emg_mdf", "emg_zc", "emg_wl", "acc_mag_mean", "acc_mag_std", "temp_c", "sweat_us"]
STATES = ["normal", "fatigue", "high_risk"]
WINDOW = 30                      # seconds of trend the classifier looks at


def crc8(data, poly=0x07):
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
    return c


def pack_emg(seq, t_ms, samples):            # used by the simulator
    body = struct.pack("<HIB", seq & 0xFFFF, t_ms & 0xFFFFFFFF, len(samples)) + struct.pack("<%dh" % len(samples), *samples)
    return body + bytes([crc8(body)])


def parse_emg(pkt):
    pkt = bytes(pkt)
    if len(pkt) < 8 or crc8(pkt[:-1]) != pkt[-1]:
        raise ValueError("bad EMG packet")
    seq, _, n = struct.unpack("<HIB", pkt[:7])
    if len(pkt) != 8 + 2 * n:
        raise ValueError("EMG length mismatch")
    return seq, np.frombuffer(pkt[7:7 + 2 * n], dtype="<i2").astype(np.float64)


_ST = "<HIHHhHBB"
def pack_status(seq, t_ms, acc_m, acc_sd, temp, sweat, batt, flags=0):   # simulator
    b = struct.pack(_ST, seq & 0xFFFF, t_ms & 0xFFFFFFFF, int(acc_m * 1000), int(acc_sd * 1000),
                    int(round(temp * 100)), int(sweat * 100), batt, flags)
    return b + bytes([crc8(b)])


def parse_status(pkt):
    pkt = bytes(pkt)
    if len(pkt) != struct.calcsize(_ST) + 1 or crc8(pkt[:-1]) != pkt[-1]:
        raise ValueError("bad STATUS packet")
    _, _, am, asd, t, sw, batt, fl = struct.unpack(_ST, pkt[:-1])
    return {"acc_mean_g": am / 1000, "acc_sd_g": asd / 1000, "temp_c": t / 100, "sweat_us": sw / 100,
            "batt_pct": batt, "flags": fl}


# ------------------------------------------------------------------ signal processing
_HANN = np.hanning(FS)
_FREQ = np.fft.rfftfreq(FS, 1.0 / FS)
_BAND = (_FREQ >= 20) & (_FREQ <= 450)       # precomputed once: keeps the per-second work tiny


def emg_features(x):
    """1 s of raw sEMG -> rms, median frequency, zero crossings, waveform length."""
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    wl = float(np.abs(np.diff(x)).sum())
    thr = 0.05 * rms
    zc = float(np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) >= thr)))
    p = np.abs(np.fft.rfft(x * _HANN)) ** 2
    cs = np.cumsum(p[_BAND])
    mdf = float(_FREQ[_BAND][np.searchsorted(cs, cs[-1] / 2.0)])
    return rms, mdf, zc, wl


def window_features(z):
    """(T,8) z-scored epochs -> 16 numbers (mean + total change per feature)."""
    t = np.arange(len(z), dtype=float); t -= t.mean()
    slope = (t[:, None] * (z - z.mean(0))).sum(0) / (np.sum(t * t) + 1e-9)
    return np.concatenate([z.mean(0), slope * len(t)])


class Classifier:
    def __init__(self, weights_path=None, sensitivity=1.0):
        self.w = None
        self.sens = sensitivity
        if weights_path:
            d = np.load(weights_path)
            self.w = []
            i = 0
            while f"W{i}" in d:
                self.w.append((d[f"W{i}"], d[f"b{i}"])); i += 1
            print(f"loaded model: {len(self.w)} layers from {weights_path}")

    def __call__(self, wf):
        if self.w:
            a = wf
            for i, (W, b) in enumerate(self.w):
                a = a @ W + b
                if i < len(self.w) - 1: a = np.maximum(a, 0)
            e = np.exp(a - a.max()); p = e / e.sum()
            risk = int(round(100 * (0.5 * p[1] + p[2])))
            return STATES[int(np.argmax(p))], min(100, risk)
        # rule-based placeholder: higher rms, lower median freq, warmer, sweatier, and trending that way
        m, s = wf[:8], wf[8:]
        score = 0.35 * m[0] - 0.35 * m[1] + 0.15 * m[6] + 0.15 * m[7] + 0.15 * (s[0] - s[1])
        risk = int(np.clip(score * 30, 0, 100))
        lo, hi = 35 / self.sens, 65 / self.sens          # conservative users (sens>1) alert earlier
        return ("normal" if risk < lo else "fatigue" if risk < hi else "high_risk"), risk


# ------------------------------------------------------------------ pipeline
class Analyzer:
    def __init__(self, hub, clf, profile=None, baseline_secs=60, log=None, label=""):
        self.hub, self.clf, self.label = hub, clf, label
        self.buf = np.empty(0)
        self.expect = None
        self.dropped = 0
        self.status = {}
        self.zs = deque(maxlen=WINDOW)
        self.t = 0
        self.base_rows, self.baseline_secs = [], baseline_secs
        self.mean = self.std = None
        if profile:
            self.mean = np.array(profile["baseline_mean"], float)
            self.std = self._floor(self.mean, np.array(profile["baseline_std"], float))
            print("baseline loaded from onboarding profile")
        self.log = log
        if log:
            self.csvf = open(log, "w", newline=""); self.w = csv.writer(self.csvf)
            self.w.writerow(["t"] + FEATURES + ["state", "risk", "label"])

    @staticmethod
    def _floor(mean, std):
        """A quiet baseline can have near-zero spread, which would blow z-scores up; floor at 10% of the mean."""
        return np.maximum(std, np.maximum(0.1 * np.abs(mean), 1e-3))

    def on_status(self, pkt):
        try: self.status = parse_status(pkt)
        except ValueError: pass

    def on_emg(self, pkt):
        try: seq, x = parse_emg(pkt)
        except ValueError: return
        if self.expect is not None and seq != self.expect:
            lost = (seq - self.expect) & 0xFFFF
            self.dropped += lost
            self.buf = np.concatenate([self.buf, np.full(lost * BATCH, np.nan)])
        self.expect = (seq + 1) & 0xFFFF
        self.buf = np.concatenate([self.buf, x])
        while len(self.buf) >= FS:
            sec, self.buf = self.buf[:FS], self.buf[FS:]
            self._epoch(sec)

    def _epoch(self, sec):
        self.t += 1
        if np.isnan(sec).any():                       # never invent features across a gap
            self.hub.publish({"t": self.t, "state": "gap", "risk": None, "dropped_batches": self.dropped}); return
        s = self.status
        row = np.array([*emg_features(sec), s.get("acc_mean_g", np.nan), s.get("acc_sd_g", np.nan),
                        s.get("temp_c", np.nan), s.get("sweat_us", np.nan)])
        msg = {"t": self.t, "features": dict(zip(FEATURES, np.round(row, 3).tolist())),
               "status": {k: s.get(k) for k in ("acc_mean_g", "temp_c", "sweat_us", "batt_pct")},
               "dropped_batches": self.dropped}
        if self.mean is None:                          # no profile: learn baseline from the first N seconds
            self.base_rows.append(row)
            if len(self.base_rows) >= self.baseline_secs:
                b = np.array(self.base_rows)
                self.mean = np.nanmean(b, 0); self.std = self._floor(self.mean, np.nanstd(b, 0))
                print(f"baseline learned from first {self.baseline_secs}s")
            msg.update(state="calibrating", risk=0, progress=round(len(self.base_rows) / self.baseline_secs, 2))
        else:
            z = np.clip(np.nan_to_num((row - self.mean) / self.std), -6, 6)
            self.zs.append(z)
            msg["z"] = dict(zip(FEATURES, np.round(z, 2).tolist()))
            if len(self.zs) >= 10:
                msg["state"], msg["risk"] = self.clf(window_features(np.array(self.zs)))
            else:
                msg.update(state="normal", risk=0)
        self.hub.publish(msg)
        if self.log: self.w.writerow([self.t, *row.tolist(), msg["state"], msg["risk"], self.label]); self.csvf.flush()


# ------------------------------------------------------------------ server -> iPhone
class Hub:
    def __init__(self, token=None):
        self.token, self.latest, self.clients, self.lock = token, {}, [], threading.Lock()

    def publish(self, msg):
        self.latest = msg
        with self.lock:
            for q in self.clients:
                try: q.put_nowait(msg)
                except queue.Full: pass
        print(f"t={msg.get('t'):4} {msg.get('state'):11s} risk={msg.get('risk')}  dropped={msg.get('dropped_batches')}")


def make_handler(hub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def _authed(self):
            if not hub.token: return True
            qs = parse_qs(urlparse(self.path).query)
            return qs.get("token", [""])[0] == hub.token or self.headers.get("Authorization") == f"Bearer {hub.token}"

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health": return self._json(200, {"ok": True})
            if not self._authed(): return self._json(401, {"error": "unauthorized"})
            if path == "/latest": return self._json(200, hub.latest)
            if path != "/stream": return self._json(404, {"error": "not found"})
            q = queue.Queue(maxsize=100)
            with hub.lock: hub.clients.append(q)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if hub.latest: q.put_nowait(hub.latest)
                while True:
                    try: self.wfile.write(f"data: {json.dumps(q.get(timeout=15))}\n\n".encode())
                    except queue.Empty: self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with hub.lock: hub.clients.remove(q)
    return H


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except OSError: return "127.0.0.1"
    finally: s.close()


# ------------------------------------------------------------------ data sources
def synth_emg(rng, amp, center):
    w = np.fft.rfft(rng.standard_normal(FS))
    shape = np.exp(-((_FREQ - center) / (0.6 * center)) ** 2) * _BAND
    x = np.fft.irfft(w * shape, FS)
    return x / (x.std() + 1e-12) * amp


async def simulate(an, speed):
    """Fake band: 70 s of steady effort, then a ramp of rising effort / falling median frequency."""
    rng, seq, st = np.random.default_rng(0), 0, 0
    t = 0
    while True:
        e = 0.0 if t % 240 < 70 else ((t % 240) - 70) / 170
        sig = synth_emg(rng, 80 + 120 * e, 105 - 35 * e).astype(int)
        for b in range(FS // BATCH):
            p = pack_emg(seq, t * 1000 + b * BATCH, sig[b * BATCH:(b + 1) * BATCH].tolist()); seq += 1
            if random.random() > 0.01: an.on_emg(p)          # ~1% packet loss, like real BLE
        an.on_status(pack_status(st, t * 1000, 1 + 0.3 * e, 0.1, 33 + 1.2 * e, 2 + 3 * e, 96 - (t // 60) % 90)); st += 1
        t += 1
        await asyncio.sleep(1.0 / speed)


async def ble_source(an, address):
    from bleak import BleakScanner, BleakClient
    while True:                                          # auto-reconnect with a short backoff
        try:
            dev = (address and await BleakScanner.find_device_by_address(address, timeout=10)) \
                or await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10)
            if not dev: print("PowerBand not found, retrying..."); await asyncio.sleep(3); continue
            gone = asyncio.Event()
            async with BleakClient(dev, disconnected_callback=lambda c: gone.set()) as c:
                print("connected:", dev.address)
                an.expect = None                         # new connection -> sequence restarts
                await c.start_notify(EMG_UUID, lambda _, d: an.on_emg(d))
                await c.start_notify(STATUS_UUID, lambda _, d: an.on_status(d))
                await c.write_gatt_char(CONTROL_UUID, bytes([CMD_START]), response=True)
                await gone.wait()
            print("band disconnected, retrying...")
        except Exception as e:
            print("BLE error:", e); await asyncio.sleep(3)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true"); ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--address"); ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token"); ap.add_argument("--profile"); ap.add_argument("--weights")
    ap.add_argument("--baseline-secs", type=int, default=60)
    ap.add_argument("--log"); ap.add_argument("--label", default="")
    a = ap.parse_args()

    prof = json.load(open(a.profile)) if a.profile else None
    hub = Hub(a.token)
    an = Analyzer(hub, Classifier(a.weights, prof.get("alert_sensitivity", 1.0) if prof else 1.0),
                  prof, a.baseline_secs, a.log, a.label)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(hub))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"Hub up: http://{lan_ip()}:{a.port}/stream" + (f"?token={a.token}" if a.token else "") +
          ("   (WARNING: no --token, anyone on your Wi-Fi can read it)" if not a.token else ""))
    try:
        asyncio.run(simulate(an, a.speed) if a.simulate else ble_source(an, a.address))
    except KeyboardInterrupt:
        print("stopped")
