/*
 * memprobe — gate G0: how much memory can ONE unprivileged process keep
 * RESIDENT on this phone before the system pushes back?
 *
 * Why: the memory budget M in moe-phone/ESTIMAND.md section 5 must be a
 * measured operating point, not a spec-sheet 12 GB.
 *
 * WHAT IS MEASURED (the estimand, CLAUDE.md section 2). "Memory one app can
 * hold" is ambiguous between three different numbers, and on a phone with
 * 12 GB of zram they differ by almost an order of magnitude:
 *
 *   allocated       bytes mmap'd                  — says nothing; the kernel
 *                                                   overcommits
 *   allocated+swap  bytes written at least once   — includes pages now sitting
 *                                                   compressed in zram, which
 *                                                   an inference engine must
 *                                                   decompress to touch
 *   RESIDENT        VmRSS: bytes actually in DRAM — this is the one an engine
 *                                                   can read at DRAM speed
 *
 * This program reports the third. The operating point is **max VmRSS observed**,
 * read from /proc/self/status, and the full curve is printed so a reader can
 * see the shape rather than trust one number.
 *
 * Method: allocate 256 MB chunks, fill each with INCOMPRESSIBLE data (zram
 * would otherwise compress zero or repetitive pages and overstate capacity),
 * and after each chunk record VmRSS, VmSwap, fill rate, MemAvailable and
 * SwapFree. Stop — before anything is killed — when any of:
 *   - MemAvailable falls below --floor-mb (default 600),
 *   - VmRSS plateaus: it grew by less than --grow-frac of the chunk just
 *     written, for --plateau-chunks consecutive chunks. That means the system
 *     is evicting our pages as fast as we write them, so the resident ceiling
 *     has been reached and climbing further cannot raise max VmRSS,
 *   - --max-mb is reached,
 *   - an mmap fails.
 *
 * Note on the stop rule and CLAUDE.md section 6.4 (no constant tuned against
 * the metric it influences): the REPORTED quantity is max VmRSS, and the
 * plateau rule only decides when to stop climbing a curve that is, by the
 * rule's own condition, no longer rising. So the reported number does not
 * depend on --grow-frac. The previous version reported the last *allocated*
 * size and stopped on "this chunk took more than 5x the FIRST chunk"; on the
 * OnePlus 15R (2026-09-14) the first chunk was the fastest of the run by
 * chance (0.227 s vs 0.639 s for the second), so the rule fired at 1280 MB
 * while MemAvailable was still 2314 MB — 3.8x above its own floor. A single
 * sample is not a baseline. The timing guard is kept as a safety net only,
 * with a median baseline and the same persistence requirement, and it never
 * sets the reported figure.
 *
 * WARNING: this can make Android close BACKGROUND apps. Close anything you
 * care about first. It never runs past the stop rules above.
 *
 * TWO REGIMES, because an inference engine uses both and they do not behave
 * alike (--file selects the second):
 *
 *   anonymous (default)  private dirty pages, as an engine's own buffers and
 *                        KV cache are. Under pressure Android COMPRESSES these
 *                        into zram, so VmRSS stops rising while VmSwap grows.
 *   file-backed (--file) a read-only MAP_PRIVATE mapping of a large file, as a
 *                        memory-mapped model checkpoint is. These pages are
 *                        clean: under pressure the kernel simply DROPS them and
 *                        re-reads from flash, and they never enter zram. The
 *                        resident ceiling is therefore a different number, and
 *                        it is the one that bounds an mmap-based expert cache.
 *
 * Reporting max VmRSS for the anonymous case and calling it "the memory budget
 * for model weights" would be measuring one quantity and naming another. Run
 * both; report both; say which one a given claim rests on.
 *
 * Build (Termux): clang -O2 memprobe.c -o memprobe
 * Run:            ./memprobe [--floor-mb 600] [--max-mb 11000]
 *                            [--grow-frac 0.25] [--plateau-chunks 2]
 *                            [--slow-x 8] [--oom-adj N]
 *                 ./memprobe --file ~/moe/ufs.bin     # file-backed regime
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Read "<key>: <n> kB" from a /proc file. Returns -1 if absent or unreadable;
 * -1 is propagated to the CSV rather than silently becoming 0, so a denied
 * read never looks like a measurement (CLAUDE.md section 6.3). */
static long proc_kb(const char *path, const char *key) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    size_t n = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, n) && line[n] == ':') { sscanf(line + n + 1, "%ld", &v); break; }
    fclose(f);
    return v;
}

static long meminfo_kb(const char *key) { return proc_kb("/proc/meminfo", key); }
static long status_kb(const char *key) { return proc_kb("/proc/self/status", key); }

static long oom_adj(void) {
    FILE *f = fopen("/proc/self/oom_score_adj", "r");
    long v = 0;
    if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
    return v;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    long floor_mb = 600, max_mb = 11000, set_adj = -2000, plateau_chunks = 2;
    double slow_x = 8.0, grow_frac = 0.25;
    const char *file = NULL;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--floor-mb")) floor_mb = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--max-mb")) max_mb = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--slow-x")) slow_x = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--grow-frac")) grow_frac = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--plateau-chunks")) plateau_chunks = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--oom-adj")) set_adj = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--file")) file = argv[i + 1];
    }
    int fd = -1;
    uint64_t fsize = 0;
    if (file) {
        fd = open(file, O_RDONLY);
        if (fd < 0) { perror("open --file"); return 1; }
        struct stat st;
        if (fstat(fd, &st)) { perror("fstat"); return 1; }
        fsize = (uint64_t)st.st_size;
        /* Every page must be read from flash, not served from a cache warmed by
         * whatever ran before, or the ceiling measured is the page cache's
         * memory of an earlier run rather than this process's residency. */
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        if (fsize < (uint64_t)max_mb << 20) max_mb = (long)(fsize >> 20);
    }
    /* `adb shell` processes run at oom_score_adj -1000 (never killed), which is
     * not the regime of an app. Any process may RAISE its own adj without
     * privilege; --oom-adj 0 makes the probe as killable as a foreground app,
     * so the low-memory killer treats it the way it would treat our engine. */
    if (set_adj > -2000) {
        FILE *f = fopen("/proc/self/oom_score_adj", "w");
        if (!f || fprintf(f, "%ld\n", set_adj) < 0) printf("warning: could not set oom_score_adj\n");
        if (f) fclose(f);
    }
    const size_t chunk = 256u << 20;
    const long chunk_mb = 256;
    uint64_t x = 0x243F6A8885A308D3ULL ^ (uint64_t)time(NULL);

    long nmax = max_mb / chunk_mb + 2;
    double *times = calloc((size_t)nmax, sizeof *times);
    long rss_peak_mb = 0, plateau_run = 0, slow_run = 0, nchunk = 0;

    printf("mode=%s%s%s oom_score_adj=%ld MemTotal_MB=%ld SwapTotal_MB=%ld\n",
           file ? "file_backed" : "anonymous", file ? " file=" : "", file ? file : "",
           oom_adj(), meminfo_kb("MemTotal") / 1024, meminfo_kb("SwapTotal") / 1024);
    printf("held_MB,chunk_s,fill_MBps,VmRSS_MB,VmSwap_MB,MemAvailable_MB,SwapFree_MB,stop_reason\n");

    for (long held = 0; held + chunk_mb <= max_mb;) {
        uint64_t *p = file
            ? mmap(NULL, chunk, PROT_READ, MAP_PRIVATE, fd, (off_t)held << 20)
            : mmap(NULL, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) {
            printf("%ld,0,0,%ld,%ld,%ld,%ld,mmap_failed\n", held,
                   status_kb("VmRSS") / 1024, status_kb("VmSwap") / 1024,
                   meminfo_kb("MemAvailable") / 1024, meminfo_kb("SwapFree") / 1024);
            break;
        }
        double t0 = now_s();
        if (file) {
            /* Touch one word per 4 KB page: enough to fault it in, and the sum
             * is kept so the compiler cannot delete the loop. Timing here is
             * flash read time, not memory bandwidth. */
            volatile uint64_t sink = 0;
            for (size_t i = 0; i < chunk / 8; i += 512) sink ^= p[i];
            (void)sink;
        } else {
            for (size_t i = 0; i < chunk / 8; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; p[i] = x; }
        }
        double dt = now_s() - t0;
        held += chunk_mb;
        times[nchunk++] = dt;

        long rss_mb = status_kb("VmRSS") / 1024, swap_mb = status_kb("VmSwap") / 1024;
        long avail = meminfo_kb("MemAvailable") / 1024, swapf = meminfo_kb("SwapFree") / 1024;
        long rss_gain = rss_mb - rss_peak_mb;
        if (rss_mb > rss_peak_mb) rss_peak_mb = rss_mb;

        /* Plateau: this chunk added less than grow_frac of itself to RSS. */
        plateau_run = (rss_gain < (long)(grow_frac * chunk_mb)) ? plateau_run + 1 : 0;

        /* Timing safety net, median baseline over every chunk so far. A single
         * first sample is not a baseline; see the header note. */
        double med = 0;
        if (nchunk >= 3) {
            double *c = malloc((size_t)nchunk * sizeof *c);
            memcpy(c, times, (size_t)nchunk * sizeof *c);
            qsort(c, (size_t)nchunk, sizeof *c, cmp_double);
            med = c[nchunk / 2];
            free(c);
            slow_run = (dt > slow_x * med) ? slow_run + 1 : 0;
        }

        const char *why = "";
        if (avail >= 0 && avail < floor_mb) why = "memavailable_floor";
        else if (plateau_run >= plateau_chunks) why = "rss_plateau";
        else if (nchunk >= 3 && slow_run >= plateau_chunks) why = "fill_slowdown_swap";
        else if (held + chunk_mb > max_mb) why = "max_reached";

        printf("%ld,%.3f,%.0f,%ld,%ld,%ld,%ld,%s\n", held, dt, chunk_mb / dt,
               rss_mb, swap_mb, avail, swapf, why);
        fflush(stdout);
        if (*why) break;
    }
    /* The reported operating point. Not the last allocated size: allocation
     * that the kernel immediately compresses into zram is not resident memory
     * and an engine cannot read it at DRAM speed. */
    printf("max_VmRSS_MB=%ld chunks=%ld mode=%s\n", rss_peak_mb, nchunk,
           file ? "file_backed" : "anonymous");
    free(times);
    if (fd >= 0) close(fd);
    return 0;
}
