# Retracted: this sweep's arms were not independent

Each arm was launched into the SAME app process, and `Probe.applyEnv` only ever called `setenv` for the
variables that arm named. Android keeps an app process alive after its activity finishes, so a knob set
for one arm was still set for the next: the arms accumulated instead of isolating.

The rows are consistent with exactly that, which is why they cannot be used:

    htp_default  44.01      (nothing set)
    htp_oppoll   43.60      (OPPOLL)
    htp_hostbuf  39.18      (OPPOLL + HOSTBUF, though labelled HOSTBUF)
    htp_both     39.11      (same two -- and it matches htp_hostbuf, which is the tell)
    htp_ndev2    36.71      (those two + DEVICES=2)
    htp_nobatch  32.08      (all of the above + OPBATCH=1)

`htp_hostbuf` and `htp_both` agreeing to 0.07 tok/s is not a coincidence: by then they were the same
configuration. The monotone decline down the list is the accumulation, not a property of each knob.

SECOND, INDEPENDENT REASON, found while restarting it: three copies of `chain_resume.sh` were running at
once, two of them driving this same sweep concurrently. My kill pattern matched the literal string
"bash chain_resume.sh" while the processes were "bash host/chain_resume.sh", so each "restart" added an
instance instead of replacing one. Two campaigns sharing the phone contend for CPU, memory bandwidth and
the app process itself, which is enough to invalidate the rows on its own.

Nothing here is quoted anywhere. The rows are kept because a retraction that deletes its own evidence
cannot be checked.

Fixed three ways before re-running: `setEnv(k, "")` now calls `unsetenv`; `Probe.applyEnv` clears every
knob the sweep can touch before applying the arm's own; and the driver force-stops the package before
each arm so the process is fresh regardless. Two further rows (`htp_default`, `gpu`, `cpu`) had timed
out rather than failed -- the driver's wait was shorter than a cold-page-cache model load -- so the wait
is now 10 minutes and the model is warmed into the page cache before the campaign starts.
