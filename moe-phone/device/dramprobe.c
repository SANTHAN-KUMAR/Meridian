/*
 * dramprobe — stub S1: sustained DRAM READ bandwidth available to one
 * unprivileged process, as a function of thread count.
 *
 * Why this exists (moe-phone/POSITION.md §11, S1): gates/byte_budget.py's
 * --dram-gbps is the last phone property that is still ASSUMED (a HeteroLLM
 * figure). Every resident weight and every expert-cache HIT is read at this
 * rate, so it sets the ceiling a well-cached engine runs into, and it is the
 * term the flash-only formula in ARCHITECTURE.md §1 leaves out.
 *
 * Method. Allocate --mb MiB of anonymous memory, fault every page in with a
 * write, then each of T threads streams its OWN contiguous slice with plain
 * 64-bit loads folded into 8 independent accumulators (the compiler vectorises
 * this at -O3). A decode GEMV reads each weight tensor front to back, so a
 * sequential stream is the access pattern being modelled, not a random one.
 *
 * Correctness rules built in:
 *  - The buffer must dwarf every cache level. The default 1024 MiB is ~25x the
 *    SoC's combined L2 + system-level cache; the artifact records the size.
 *  - Pages compressed into zram are NOT at DRAM speed. After the fill, VmRSS
 *    must cover >= 95% of the buffer or the run refuses to report
 *    (exit 3), rather than timing decompression and calling it DRAM.
 *  - Every accumulator reaches a volatile sink, so no load can be eliminated.
 *  - A thread that was descheduled is not measuring memory. Each thread's CPU
 *    time is compared with its wall time; rows where any thread got < 90% of
 *    the wall clock are marked descheduled=1 and must be excluded downstream
 *    (the same failure ufsbench met with the screen off, README defect table).
 *  - GB/s is DECIMAL (1e9 bytes), the unit byte_budget.py takes.
 *
 * Build with the NDK:  aarch64-linux-android30-clang -O3 -pthread -static dramprobe.c -o dramprobe
 * Build on the phone:  clang -O3 -pthread dramprobe.c -o dramprobe
 * Build on Linux:      gcc -O3 -pthread dramprobe.c -o dramprobe
 *
 * Usage:
 *   dramprobe [--mb 1024] [--threads 1,2,4,6,8] [--seconds 2] [--repeats 5]
 *             [--cpus 0,1,...] [--out dram.csv]
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static double now(clockid_t c) {
    struct timespec ts;
    clock_gettime(c, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int parse_list(const char *s, int *out, int max) {
    int n = 0;
    char *dup = strdup(s), *save = NULL;
    for (char *t = strtok_r(dup, ",", &save); t && n < max; t = strtok_r(NULL, ",", &save))
        out[n++] = atoi(t);
    free(dup);
    return n;
}

static long vmrss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "VmRSS:", 6)) { kb = strtol(line + 6, NULL, 10); break; }
    fclose(f);
    return kb;
}

static volatile uint64_t g_sink;

typedef struct {
    const uint64_t *base;
    size_t words;            /* slice length in uint64 */
    double deadline_s;       /* per-run duration */
    int cpu;                 /* -1: no pinning */
    pthread_barrier_t *bar;
    /* out */
    double bytes, wall, cpu_time;
} job_t;

static void *worker(void *arg) {
    job_t *j = arg;
    if (j->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(j->cpu, &set);
        sched_setaffinity(0, sizeof set, &set);   /* best effort; recorded by caller */
    }
    pthread_barrier_wait(j->bar);
    double t0 = now(CLOCK_MONOTONIC), c0 = now(CLOCK_THREAD_CPUTIME_ID);
    double end = t0 + j->deadline_s;
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    size_t passes = 0;
    const size_t n = j->words & ~(size_t)7;
    do {
        const uint64_t *p = j->base;
        for (size_t i = 0; i < n; i += 8) {
            a0 ^= p[i];     a1 ^= p[i + 1]; a2 ^= p[i + 2]; a3 ^= p[i + 3];
            a4 ^= p[i + 4]; a5 ^= p[i + 5]; a6 ^= p[i + 6]; a7 ^= p[i + 7];
        }
        passes++;
    } while (now(CLOCK_MONOTONIC) < end);
    j->wall = now(CLOCK_MONOTONIC) - t0;
    j->cpu_time = now(CLOCK_THREAD_CPUTIME_ID) - c0;
    j->bytes = (double)passes * n * sizeof(uint64_t);
    g_sink ^= a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    return NULL;
}

int main(int argc, char **argv) {
    size_t mb = 1024;
    int thr[16] = {1, 2, 4, 6, 8}, nthr = 5, cpus[64], ncpus = 0, repeats = 5;
    double seconds = 2.0;
    const char *out = "dramprobe.csv";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--mb")) mb = strtoull(v, NULL, 10), i++;
        else if (!strcmp(a, "--threads")) nthr = parse_list(v, thr, 16), i++;
        else if (!strcmp(a, "--cpus")) ncpus = parse_list(v, cpus, 64), i++;
        else if (!strcmp(a, "--seconds")) seconds = atof(v), i++;
        else if (!strcmp(a, "--repeats")) repeats = atoi(v), i++;
        else if (!strcmp(a, "--out")) out = v, i++;
        else { fprintf(stderr, "usage: dramprobe [--mb N] [--threads 1,2,..] [--seconds S] "
                               "[--repeats R] [--cpus a,b,..] [--out f.csv]\n"); return 2; }
    }
    size_t bytes = mb << 20;
    uint64_t *buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }
    long rss0 = vmrss_kb();
    for (size_t i = 0; i < bytes / sizeof(uint64_t); i++) buf[i] = i * 0x9E3779B97F4A7C15ull;
    long rss1 = vmrss_kb();
    double resident = (rss1 - rss0) * 1024.0 / bytes;
    fprintf(stderr, "buffer %zu MiB, resident fraction after fill %.3f\n", mb, resident);
    if (resident < 0.95) {
        fprintf(stderr, "REFUSING: only %.1f%% of the buffer is resident; the rest is in zram "
                        "and would be timed as decompression, not DRAM. Close apps or lower --mb.\n",
                100 * resident);
        return 3;
    }
    FILE *f = fopen(out, "w");
    if (!f) { perror(out); return 1; }
    const char *hdr = "repeat,threads,buffer_mb,seconds,GBps,min_cpu_over_wall,descheduled,pinned,resident_frac";
    fprintf(f, "%s\n", hdr);
    printf("%s\n", hdr);
    for (int r = 1; r <= repeats; r++) {
        for (int ti = 0; ti < nthr; ti++) {
            int T = thr[ti];
            pthread_t th[64];
            job_t jobs[64];
            pthread_barrier_t bar;
            pthread_barrier_init(&bar, NULL, T);
            size_t words = bytes / sizeof(uint64_t) / T;
            for (int t = 0; t < T; t++) {
                jobs[t] = (job_t){.base = buf + (size_t)t * words, .words = words,
                                  .deadline_s = seconds, .cpu = ncpus ? cpus[t % ncpus] : -1,
                                  .bar = &bar};
                pthread_create(&th[t], NULL, worker, &jobs[t]);
            }
            double tot = 0, wall = 0, minratio = 1e9;
            for (int t = 0; t < T; t++) {
                pthread_join(th[t], NULL);
                tot += jobs[t].bytes;
                if (jobs[t].wall > wall) wall = jobs[t].wall;
                double ratio = jobs[t].cpu_time / jobs[t].wall;
                if (ratio < minratio) minratio = ratio;
            }
            pthread_barrier_destroy(&bar);
            char line[256];
            snprintf(line, sizeof line, "%d,%d,%zu,%.4f,%.3f,%.3f,%d,%d,%.3f", r, T, mb, wall,
                     tot / wall / 1e9, minratio, minratio < 0.90, ncpus > 0, resident);
            fprintf(f, "%s\n", line);
            printf("%s\n", line);
            fflush(f);
        }
    }
    fprintf(f, "# sink=%llu\n", (unsigned long long)g_sink);
    fclose(f);
    munmap(buf, bytes);
    return 0;
}
