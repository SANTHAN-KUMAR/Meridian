/* wrbench — sequential WRITE throughput on the model-storage path.
 * Writes --mb MiB of incompressible data in 4 MiB chunks to --file, fsyncs,
 * reports MB/s (decimal). Time includes fsync: page-cache absorption is not
 * storage speed. Repeats R times; each repeat truncates and rewrites.
 * Build: aarch64-linux-android28-clang -O2 wrbench.c -o wrbench
 * Usage: wrbench --file PATH [--mb 1024] [--repeats 3]
 * Output CSV: repeat,mb,seconds,MBps,errors */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    const char *file = NULL; long mb = 1024; int reps = 3;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--file")) file = argv[i + 1];
        else if (!strcmp(argv[i], "--mb")) mb = atol(argv[i + 1]);
        else if (!strcmp(argv[i], "--repeats")) reps = atoi(argv[i + 1]);
    }
    if (!file) { fprintf(stderr, "usage: wrbench --file PATH [--mb N] [--repeats R]\n"); return 2; }
    size_t chunk = 4u << 20;
    uint64_t *b = malloc(chunk), x = 0x9E3779B97F4A7C15ULL ^ (uint64_t)time(NULL);
    for (size_t i = 0; i < chunk / 8; i++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; b[i] = x; }
    printf("repeat,mb,seconds,MBps,errors\n");
    for (int r = 1; r <= reps; r++) {
        int fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) { perror("open"); return 1; }
        long errors = 0; double t0 = now();
        for (long done = 0; done < mb; done += 4) {
            b[0] ^= (uint64_t)done;
            if (write(fd, b, chunk) != (ssize_t)chunk) errors++;
        }
        fsync(fd);
        double dt = now() - t0;
        close(fd);
        printf("%d,%ld,%.4f,%.2f,%ld\n", r, mb, dt, (double)mb * 1048576.0 / dt / 1e6, errors);
        fflush(stdout);
    }
    unlink(file);
    return 0;
}
