# Powerthru-t
# PowerThru

**[Live demo (runs in your browser, simulated data)]([https://a4manohar-del.github.io/powerthru-t/](https://a4manohar-del.github.io/Powerthru-t/))**

> **Built with AI assistance.** Much of the code in this repo was written with Claude (Anthropic). See [AI disclosure](#ai-disclosure).

PowerThru is a wearable band for the forearm or calf that streams sEMG, motion, skin temperature and
sweat data to a phone app. The app learns each athlete's baseline, then tracks fatigue and overexertion
trends, with cramp detection as the flagship feature. The long-term target is coaches and sports programs
managing many athletes from one dashboard.

## Status (honest version)

| Part | State |
|---|---|
| Browser demo and Python hub with a simulated band | Working |
| Onboarding survey + 30 min calibration -> personal baseline and category | Working on simulated data |
| BLE packet format, phone-side recorder, test sketch | Written; hardware path untested |
| Risk score | Rule-based placeholder, **not validated** for cramps or overexertion |
| ML model | Pipeline works on synthetic data; no real labeled sEMG dataset yet |
| Custom PCB (KiCad) | In progress |

## AI disclosure

This project was built with the help of **Claude** (an AI assistant made by Anthropic). In the interest of being clear about who did what:

**Written with Claude:**
- `firmware/band_ble_test.ino`
- `onboarding_model.py`
- `tinyml_model.py`
- `ble_protocol.py`
- `ble_client.py`
- This README

**Mine (edit this list so it matches what you actually did):**
- The product idea, the design of the band, the onboarding flow and the target market
- The Arduino breadboard prototype and the hardware choices
- The WESAD training code and the decisions about which signals fit a forearm band
- Testing, directing the design, and deciding what to keep, change or drop

**Not verified:** AI-generated code can contain mistakes. The BLE firmware has not been tested on real hardware, the risk score is an untested placeholder rule, and all demo data is simulated.

## Try it

Browser: open the live demo above. Or run the full pipeline locally:

```bash
pip install numpy
python software/powerthru_hub.py --simulate --speed 20
```

In another terminal:

```bash
curl http://localhost:8765/latest      # latest JSON event
curl -N http://localhost:8765/stream   # live event stream the iPhone app reads
```

## How it works

```
band --BLE--> hub (features, baseline, score) --Wi-Fi: HTTP + SSE--> iPhone app
```

The band streams raw sEMG at 1 kHz plus a 1 Hz status packet. The hub extracts RMS and median frequency
per second, z-scores them against the user's own baseline, and scores the 30 s trend: rising RMS and a
falling median frequency are classic signs of muscle fatigue.

## Repo layout

| Path | What it is |
|---|---|
| `software/powerthru_hub.py` | One script: band (BLE or simulated) -> processing -> HTTP/SSE for the app |
| `software/onboarding_model.py` | Survey + calibration -> personal baseline and category |
| `software/tinyml_model.py` | Small MLP, int8 TFLite and C-array export (needs TensorFlow) |
| `software/ble_protocol.py`, `ble_client.py` | Packet format and a recorder that writes labeled feature CSVs |
| `software/WESAD_CNNGRU.py` | Uses WESAD data with a CNN + GRU Model |
| `firmware/band_ble_test.ino` | ArduinoBLE sketch streaming fake data in the same format (untested) |
| `firmware/firmware_micropython_sketch.py` | MicroPython peripheral sketch (untested) |
| `docs/powerthru_livd_viewer.html` | The browser demo (GitHub Pages) |
| `pcb/` | KiCad project (coming) |

## Real band

Flash `firmware/band_ble_test.ino` to a BLE board (Nano 33 BLE, Nano ESP32, ...), confirm "PowerBand"
appears in nRF Connect, then run `python software/powerthru_hub.py`. Packets are 108 bytes, so the BLE MTU
must be at least 111; otherwise set the batch size to 6 in both the sketch and the hub.

## Next

1. Collect labeled forearm sEMG sessions (`python software/ble_client.py --label fatigue`).
2. Train and validate across subjects (never test on people the model has seen).
3. Move from the breadboard to a BLE-capable PCB.

*Not a medical device.*
