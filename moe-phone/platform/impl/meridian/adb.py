"""Thin adb wrapper. No probe logic lives here -- only process plumbing."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


class AdbError(RuntimeError):
    pass


@dataclass
class AdbDevice:
    serial: str | None = None

    def _base(self) -> list:
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def shell(self, command: str, timeout: float = 60.0) -> str:
        cmd = self._base() + ["shell", command]
        # stdin=DEVNULL: without this, subprocess.run() inherits our stdin,
        # and adb forwards it to the remote shell. Over the wireless (TCP)
        # transport this reproducibly hung every command indefinitely --
        # something in g0_probe.sh's subshells blocks on a stdin read that
        # never gets EOF -- while the exact same command over USB returned
        # in under a second. Explicitly closing remote stdin fixed it and is
        # correct regardless of cause: nothing this profiler runs is
        # interactive.
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=timeout)
        if r.returncode != 0 and not r.stdout:
            raise AdbError(f"adb shell failed rc={r.returncode}: {command}\n{r.stderr}")
        return r.stdout

    def push(self, local: str, remote: str) -> None:
        cmd = self._base() + ["push", local, remote]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise AdbError(f"adb push failed: {r.stderr}")

    def pull(self, remote: str, local: str) -> None:
        cmd = self._base() + ["pull", remote, local]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise AdbError(f"adb pull failed: {r.stderr}")

    def devices(self) -> list:
        r = subprocess.run(["adb", "devices", "-l"], capture_output=True, text=True, timeout=15)
        lines = [l for l in r.stdout.splitlines()[1:] if l.strip()]
        return lines

    def ensure_awake(self) -> None:
        """Wake the screen and return to the launcher if the device is not
        Awake. Called before every probe stage so a run never silently
        crosses into the dozing regime (04_DEVICE_PROFILING.md section 4)."""
        out = self.shell("dumpsys power | grep -m1 mWakefulness=", timeout=20)
        if "Awake" not in out:
            self.shell("input keyevent KEYCODE_WAKEUP; input keyevent KEYCODE_HOME", timeout=20)
