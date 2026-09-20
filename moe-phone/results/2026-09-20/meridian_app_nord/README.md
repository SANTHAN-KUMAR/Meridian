# In-app device profile — OnePlus Nord (AC2001), 2026-09-20

`DeviceProfile.json` was produced by the Meridian Android app itself (`platform/impl/app`), running the bundled probes
as the app's own UID on the phone (unplugged, wireless adb only, screen held on by the app, app in the foreground).
The regime sampler recorded 122 samples during the run: all awake / foreground=self / unplugged, thermal status 0, so
`validity.conditions.in_regime` is true and the fields are `measured`. 3 dramprobe rows were rejected as descheduled.
Storage numbers are for the app's own internal path (f2fs). Read figures from the JSON, not from this file.
