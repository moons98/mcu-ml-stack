#!/usr/bin/env python3
"""Build, flash and collect results for TFLM benchmark variants.

Run inside the ESP-IDF environment:  . ~/esp/esp-idf/export.sh && python bench.py ...

  python bench.py build                       # bare_{ref,espnn} + tuned_{ref,espnn}_{160,240}_off
  python bench.py build --profile             # + per-op profiling builds (*_prof, +80 KB RAM)
  python bench.py build --wifi                # + wifi-on variants (needs cfg/wifi_secrets.defaults)
  python bench.py run -p /dev/ttyUSB0         # flash each built variant and collect serial RESULT lines
  python bench.py run -p /dev/ttyUSB0 --only bare_ref tuned_ref_240_off
  --model <name>                              # model in models/<name>/out (default: res_cnn_lstm)

Builds:  build/<model>/<variant>/
Results: <repo>/results/<model>/bench_<timestamp>.csv plus per-variant serial logs
         (static memory from idf_size is included in the CSV)
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "results"
KERNELS = ("ref", "espnn")
CPUS = ("160", "240")


def variants(with_wifi: bool, with_prof: bool = False):
    """Yield (name, [sdkconfig fragments]) applied on top of sdkconfig.defaults.

    bare_<kernel>             : ESP-IDF defaults (160 MHz, -Og, 2 MB DIO flash, ...) + kernel choice
    tuned_<kernel>_<cpu>_<wifi>: cfg/tuned.defaults (-O2, 16 MB QIO flash, ...) + axes
    """
    for k in KERNELS:
        yield f"bare_{k}", [f"cfg/{k}.defaults"]
    wifis = ("off", "on") if with_wifi else ("off",)
    for k, c, w in itertools.product(KERNELS, CPUS, wifis):
        base = ["cfg/tuned.defaults", f"cfg/{k}.defaults", f"cfg/cpu{c}.defaults", f"cfg/wifi_{w}.defaults"] + \
            (["cfg/wifi_secrets.defaults"] if w == "on" else [])
        yield f"tuned_{k}_{c}_{w}", base
        if with_prof and w == "off":
            yield f"tuned_{k}_{c}_{w}_prof", base + ["cfg/prof.defaults"]


def idf(args, build_dir: Path, defaults, model: str):
    sd = ";".join(["sdkconfig.defaults"] + defaults)
    cmd = ["idf.py", "-B", str(build_dir), f"-DSDKCONFIG={build_dir}/sdkconfig",
           f"-DSDKCONFIG_DEFAULTS={sd}", f"-DBENCH_MODEL={model}"] + args
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)


def static_mem(build_dir: Path) -> dict:
    r = subprocess.run([sys.executable, "-m", "esp_idf_size", "--format", "json", str(build_dir / "tflm_bench.map")],
                       capture_output=True, text=True)
    keys = ("diram_data", "diram_bss", "diram_text", "used_diram", "used_iram",
            "flash_code", "flash_rodata", "total_size")
    try:
        j = json.loads(r.stdout)
        return {k: j.get(k) for k in keys}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def cmd_build(a):
    if a.wifi and not (HERE / "cfg/wifi_secrets.defaults").exists():
        sys.exit("cfg/wifi_secrets.defaults not found. Create it with:\n  CONFIG_BENCH_WIFI_SSID=\"...\"\n  CONFIG_BENCH_WIFI_PASSWORD=\"...\"")
    if not (REPO / "models" / a.model / "out/model_data.cc").exists():
        sys.exit(f"models/{a.model}/out/model_data.cc not found; run models/{a.model}/export.py first")
    for name, defaults in variants(a.wifi, a.profile):
        bd = HERE / "build" / a.model / name
        if a.clean and bd.exists():
            subprocess.run(["rm", "-rf", str(bd)])
        t0 = time.time()
        r = idf(["build"], bd, defaults, a.model)
        ok = r.returncode == 0
        (bd).mkdir(parents=True, exist_ok=True)
        (bd / "build.log").write_text(r.stdout + r.stderr)
        mem = static_mem(bd) if ok else {}
        (bd / "static_mem.json").write_text(json.dumps(mem, indent=2))
        print(f"[{'OK' if ok else 'FAIL'}] {name:16s} {time.time()-t0:5.1f}s  {mem if ok else bd/'build.log'}")


def read_result(port: str, timeout: float) -> tuple[dict, str]:
    import serial  # pyserial (bundled with the ESP-IDF Python env)
    log = []
    res = {}
    with serial.Serial(port, 115200, timeout=0.5) as s:
        # Reset via DTR/RTS (works on the UART bridge port)
        s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False
        t_end = time.time() + timeout
        while time.time() < t_end:
            line = s.readline().decode(errors="replace").strip()
            if not line:
                continue
            log.append(line)
            if line.startswith(("CFG,", "RESULT,")):
                for kv in line.split(",")[1:]:
                    k, _, v = kv.partition("=")
                    res[k] = v
            if line.startswith("ERROR,"):
                res["error"] = line
                break
            if line == "DONE":
                break
    return res, "\n".join(log)


def cmd_run(a):
    out_dir = RESULTS / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rows = []
    for name, defaults in variants(True, True):
        bd = HERE / "build" / a.model / name
        if a.only and name not in a.only:
            continue
        if not (bd / "tflm_bench.bin").exists():
            continue
        print(f"== {name}: flash")
        r = idf(["-p", a.port, "flash"], bd, defaults, a.model)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            continue
        res, log = read_result(a.port, a.timeout)
        (out_dir / f"{ts}_{name}.log").write_text(log)
        mem = json.loads((bd / "static_mem.json").read_text())
        row = {"variant": name, **res, **{f"static_{k}": v for k, v in mem.items()}}
        rows.append(row)
        print(f"   {res}")
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "variant", k))
        out = out_dir / f"bench_{ts}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--wifi", action="store_true")
    b.add_argument("--clean", action="store_true")
    b.add_argument("--profile", action="store_true", help="also build per-op profiling variants (*_prof)")
    r = sub.add_parser("run")
    r.add_argument("-p", "--port", required=True)
    r.add_argument("--only", nargs="*")
    r.add_argument("--timeout", type=float, default=120)
    for p in (b, r):
        p.add_argument("--model", default="res_cnn_lstm", help="model in models/<model>/out")
    a = ap.parse_args()
    {"build": cmd_build, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
