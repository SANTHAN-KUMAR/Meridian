"""
nvme_expert_reads.py — PC scope probe (2026-09-19): how fast does this machine's storage serve the engine's read pattern?
Random O_DIRECT reads of one expert slice (default 884,736 B = one Qwen3-30B-A3B Q4_0 gate/up slice) from a large model file,
with N concurrent lanes for T seconds. It reports GB/s and the p50/p99 per-read latency. Read-only; memory use is N aligned buffers.
  python3 nvme_expert_reads.py FILE [lanes=8] [seconds=10] [bytes=884736]
"""
import mmap, os, random, statistics as S, sys, threading, time

f = sys.argv[1]
lanes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
nb = int(sys.argv[4]) if len(sys.argv) > 4 else 884736
ALIGN = 4096
nb_al = (nb + ALIGN - 1) // ALIGN * ALIGN
size = os.path.getsize(f)
lat, tot = [], [0]
lock = threading.Lock()


def lane(seed):
    fd = os.open(f, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, nb_al)
    rng = random.Random(seed)
    end = time.time() + secs
    my, mylat = 0, []
    while time.time() < end:
        off = rng.randrange(0, (size - nb_al) // ALIGN) * ALIGN
        t0 = time.perf_counter()
        n = os.preadv(fd, [buf], off)
        mylat.append(time.perf_counter() - t0)
        my += n
    os.close(fd)
    with lock:
        tot[0] += my; lat.extend(mylat)


th = [threading.Thread(target=lane, args=(i,)) for i in range(lanes)]
t0 = time.time()
[t.start() for t in th]; [t.join() for t in th]
dt = time.time() - t0
lat.sort()
print(f"RESULT file_GB={size/1e9:.1f} lanes={lanes} read_bytes={nb_al} reads={len(lat)} GBps={tot[0]/dt/1e9:.2f} "
      f"p50_ms={lat[len(lat)//2]*1e3:.2f} p99_ms={lat[int(len(lat)*0.99)]*1e3:.2f}")
