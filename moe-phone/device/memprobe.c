/*
 * memprobe — gate G0: how much memory can ONE unprivileged process actually
 * hold resident on this phone before the system pushes back?
 *
 * Why: the memory budget M in moe-phone/ESTIMAND.md §5 must be a measured
 * operating point, not a spec-sheet 12 GB. Android compresses memory (zram)
 * and its low-memory killer reclaims apps, so the usable figure is found by
 * trying.
 *
 * Method: allocate 256 MB chunks, fill each with INCOMPRESSIBLE data (zram
 * would otherwise compress zero or repetitive pages and overstate capacity),
 * and after each chunk print the chunk's fill time and the kernel's
 * MemAvailable / SwapFree. Stop — before anything is killed — when any of:
 *   - MemAvailable falls below --floor-mb (default 600),
 *   - a chunk takes more than --slow-x times the first chunk (swap thrash),
 *   - --max-mb is reached.
 * The last printed line is the measured operating point.
 *
 * WARNING: this can make Android close BACKGROUND apps. Close anything you
 * care about first. It never runs past the stop rules above.
 *
 * Build (Termux): clang -O2 memprobe.c -o memprobe
 * Run:            ./memprobe [--floor-mb 600] [--max-mb 11000] [--slow-x 5]
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static long meminfo_kb(const char *key) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long v = -1;
    size_t n = strlen(key);
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, n) && line[n] == ':') { sscanf(line + n + 1, "%ld", &v); break; }
    fclose(f);
    return v;
}

static long oom_adj(void) {
    FILE *f = fopen("/proc/self/oom_score_adj", "r");
    long v = 0;
    if (f) { if (fscanf(f, "%ld", &v) != 1) v = 0; fclose(f); }
    return v;
}

int main(int argc, char **argv) {
    long floor_mb = 600, max_mb = 11000, set_adj = -2000;
    double slow_x = 5.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--floor-mb")) floor_mb = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--max-mb")) max_mb = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--slow-x")) slow_x = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--oom-adj")) set_adj = atol(argv[i + 1]);
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
    uint64_t x = 0x243F6A8885A308D3ULL ^ (uint64_t)time(NULL);
    double first = 0;
    printf("oom_score_adj=%ld MemTotal_MB=%ld\n", oom_adj(), meminfo_kb("MemTotal") / 1024);
    printf("held_MB,chunk_s,MemAvailable_MB,SwapFree_MB,stop_reason\n");
    for (long held = 0; held + 256 <= max_mb;) {
        uint64_t *p = mmap(NULL, chunk, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { printf("%ld,0,%ld,%ld,mmap_failed\n", held, meminfo_kb("MemAvailable") / 1024, meminfo_kb("SwapFree") / 1024); return 0; }
        double t0 = now_s();
        for (size_t i = 0; i < chunk / 8; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; p[i] = x; }
        double dt = now_s() - t0;
        held += 256;
        if (first == 0) first = dt;
        long avail = meminfo_kb("MemAvailable") / 1024, swapf = meminfo_kb("SwapFree") / 1024;
        const char *why = "";
        if (avail >= 0 && avail < floor_mb) why = "memavailable_floor";
        else if (dt > slow_x * first) why = "fill_slowdown_swap";
        else if (held + 256 > max_mb) why = "max_reached";
        printf("%ld,%.3f,%ld,%ld,%s\n", held, dt, avail, swapf, why);
        fflush(stdout);
        if (*why) return 0;
    }
    return 0;
}
