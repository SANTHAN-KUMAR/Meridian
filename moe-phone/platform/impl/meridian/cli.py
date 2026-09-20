"""python -m meridian.cli profile --bin-dir <probes> --out-dir <results dir> [--serial <adb serial>]

Runs T0+T1 on the attached device and writes DeviceProfile.json + report.md
into --out-dir. Re-run with --from-existing to rebuild the profile from a
previously captured out-dir without touching the device again.
"""
from __future__ import annotations

import argparse
import os
import sys

from .adb import AdbDevice
from .profiler import run_probes, build_profile, save_profile
from .report import render


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    prof = sub.add_parser("profile")
    prof.add_argument("--bin-dir", required=False, help="dir with the cross-compiled probe binaries")
    prof.add_argument("--out-dir", required=True)
    prof.add_argument("--serial", default=None)
    prof.add_argument("--from-existing", action="store_true",
                       help="skip the device and rebuild from artifacts already in --out-dir")

    args = p.parse_args(argv)

    if args.cmd == "profile":
        device = AdbDevice(serial=args.serial)
        if not args.from_existing:
            if not args.bin_dir:
                print("error: --bin-dir is required unless --from-existing", file=sys.stderr)
                return 2
            devices = device.devices()
            if not devices:
                print("error: no adb device attached/authorized", file=sys.stderr)
                return 1
            run_probes(device, args.bin_dir, args.out_dir)
        profile = build_profile(args.out_dir)
        json_path = os.path.join(args.out_dir, "DeviceProfile.json")
        report_path = os.path.join(args.out_dir, "capability_report.md")
        save_profile(profile, json_path)
        with open(report_path, "w") as f:
            f.write(render(profile))
        print(f"wrote {json_path}")
        print(f"wrote {report_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
