/*
 * fadvdrop — evict one file's pages from the page cache, without root.
 *
 * posix_fadvise(POSIX_FADV_DONTNEED) is honoured for any file the caller can
 * open; ufsbench uses the same call before every configuration. A "cold" decode
 * measurement needs the checkpoint OUT of the page cache first, or a miss is a
 * memcpy rather than a storage read (colibri docs/benchmarks.md warns of exactly
 * this). Prints how many of the file's pages were resident before and after,
 * via mincore(), so "cold" is measured rather than assumed.
 *
 * Build: aarch64-linux-android30-clang -O2 -static fadvdrop.c -o fadvdrop
 * Usage: fadvdrop FILE [FILE...]
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static double resident_frac(int fd, size_t len) {
    if (len == 0) return 0.0;
    void *p = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return -1.0;
    long pg = sysconf(_SC_PAGESIZE);
    size_t n = (len + pg - 1) / pg, in = 0;
    unsigned char *v = malloc(n);
    if (v && mincore(p, len, v) == 0)
        for (size_t i = 0; i < n; i++) in += v[i] & 1;
    else
        in = (size_t)-1;
    free(v);
    munmap(p, len);
    return in == (size_t)-1 ? -1.0 : (double)in / n;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: fadvdrop FILE...\n"); return 2; }
    int rc = 0;
    for (int i = 1; i < argc; i++) {
        int fd = open(argv[i], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) { perror(argv[i]); rc = 1; continue; }
        double before = resident_frac(fd, st.st_size);
        fdatasync(fd);
        int e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        double after = resident_frac(fd, st.st_size);
        printf("%s resident_before=%.4f resident_after=%.4f fadvise_rc=%d\n", argv[i], before, after, e);
        if (e != 0 || after > 0.05) rc = 1;   /* not cold: say so via the exit code */
        close(fd);
    }
    return rc;
}
