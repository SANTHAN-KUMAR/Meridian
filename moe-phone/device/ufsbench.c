/*
 * ufsbench — gate G1: phone storage read throughput as a function of request
 * size, concurrency and I/O mode, with optional power and thermal sampling.
 *
 * Why this exists (moe-phone/POSITION.md, assumption A3): flash-streamed
 * inference time depends on the SIZE of each read, not only on bytes read.
 * gates/byte_budget.py needs two measured inputs — bulk bandwidth and
 * small-read bandwidth — and this program measures both on the actual phone,
 * as an ordinary unprivileged process (no root).
 *
 * Correctness rules built in:
 *  - Random reads visit each aligned slot of the file AT MOST ONCE per
 *    configuration (a shuffled permutation), so the page cache cannot serve a
 *    re-read and inflate throughput.
 *  - Before every configuration the file's cached pages are dropped with
 *    posix_fadvise(DONTNEED) — works without root for the caller's own file.
 *  - Buffered random reads set POSIX_FADV_RANDOM so kernel readahead does not
 *    fetch bytes the benchmark never asked for.
 *  - O_DIRECT is tried; if the filesystem refuses it (common with file-based
 *    encryption), direct configurations are reported as unsupported, never
 *    silently replaced by buffered ones.
 *
 * Build on the phone (Termux):  clang -O2 -pthread ufsbench.c -o ufsbench -lm
 * Build with the NDK:           aarch64-linux-android30-clang -O2 -pthread -static ufsbench.c -o ufsbench -lm
 * Build on Linux (self-test):   gcc -O2 -pthread ufsbench.c -o ufsbench -lm
 *
 * Self-test note: under a VM (WSL2) the host's page cache sits below the
 * guest, so even O_DIRECT reads return at memory speed. The analysis script
 * rejects any row above the storage interface ceiling for exactly this reason.
 *
 * Usage:
 *   ufsbench --file PATH --size-mb N [--create] [--seconds S] [--repeats R]
 *            [--sizes-kb 4,16,...] [--threads 1,2,4,8] [--modes direct,buffered]
 *            [--patterns rand,seq] [--cpus 4,5,6,7] [--power] [--out results.csv]
 * CSV goes to --out (default ufsbench.csv) and a copy to stdout.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096
#define LAT_CAP 262144

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;
static uint64_t xorshift(void) {
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return rng_state = x;
}

static int parse_list(const char *s, long *out, int cap) {
    int n = 0;
    char *dup = strdup(s), *tok, *save = NULL;
    for (tok = strtok_r(dup, ",", &save); tok && n < cap; tok = strtok_r(NULL, ",", &save))
        out[n++] = atol(tok);
    free(dup);
    return n;
}

/* ---------- power and thermal sampling (best effort, never fatal) ---------- */

static double read_num(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return NAN;
    double v = NAN;
    if (fscanf(f, "%lf", &v) != 1) v = NAN;
    fclose(f);
    return v;
}

static double max_thermal_c(void) {
    DIR *d = opendir("/sys/class/thermal");
    if (!d) return NAN;
    struct dirent *e;
    double best = NAN;
    char p[512], type[128];
    while ((e = readdir(d))) {
        if (strncmp(e->d_name, "thermal_zone", 12)) continue;
        /* Skip zones that report a configured threshold, not a sensor reading:
         * on the OnePlus 15R `cpu-hw-trip-*` reads a constant 95000 (found in
         * the first G1 run, whose temperature columns are therefore invalid),
         * and `*-bcl-*` zones are battery current-limit levels. */
        snprintf(p, sizeof p, "/sys/class/thermal/%s/type", e->d_name);
        FILE *tf = fopen(p, "r");
        type[0] = 0;
        if (tf) { if (!fgets(type, sizeof type, tf)) type[0] = 0; fclose(tf); }
        if (strstr(type, "trip") || strstr(type, "bcl")) continue;
        snprintf(p, sizeof p, "/sys/class/thermal/%s/temp", e->d_name);
        double v = read_num(p);
        if (isnan(v) || v <= 0) continue;
        if (v > 1000) v /= 1000.0; /* millidegrees */
        if (v > 150) continue;     /* implausible sensor value */
        if (isnan(best) || v > best) best = v;
    }
    closedir(d);
    return best;
}

typedef struct {
    atomic_int stop;
    double sum_w, n;
} power_t;

static void *power_thread(void *arg) {
    power_t *p = arg;
    while (!atomic_load(&p->stop)) {
        double i = read_num("/sys/class/power_supply/battery/current_now");
        double v = read_num("/sys/class/power_supply/battery/voltage_now");
        if (!isnan(i) && !isnan(v)) {
            /* Units are microamps and microvolts on most Android kernels; sign
             * conventions differ by vendor, so the magnitude is recorded. */
            p->sum_w += fabs(i) * 1e-6 * v * 1e-6;
            p->n += 1;
        }
        usleep(50000);
    }
    return NULL;
}

/* ---------- read workers ---------- */

typedef struct {
    int fd, seq, cpu;
    size_t req;
    const uint64_t *perm;
    uint64_t nslots;
    _Atomic uint64_t *next;
    uint64_t seq_lo, seq_hi; /* byte range for sequential pattern */
    double deadline;
    uint64_t bytes, reads, errors;
    int first_errno;
    float *lat;
    uint64_t nlat;
} worker_t;

static void *worker(void *arg) {
    worker_t *w = arg;
    if (w->cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(w->cpu, &s);
        sched_setaffinity((pid_t)syscall(SYS_gettid), sizeof s, &s);
    }
    void *buf = NULL;
    if (posix_memalign(&buf, ALIGN, w->req)) { w->errors++; return NULL; }
    uint64_t off = w->seq_lo;
    for (;;) {
        if (now_s() >= w->deadline) break;
        uint64_t pos;
        if (w->seq) {
            if (off + w->req > w->seq_hi) break;
            pos = off;
            off += w->req;
        } else {
            uint64_t k = atomic_fetch_add(w->next, 1);
            if (k >= w->nslots) break; /* every slot read once: stop, never re-read */
            pos = w->perm[k] * (uint64_t)w->req;
        }
        double t0 = now_s();
        ssize_t r = pread(w->fd, buf, w->req, (off_t)pos);
        double t1 = now_s();
        if (r != (ssize_t)w->req) {
            if (!w->first_errno) w->first_errno = r < 0 ? errno : -1;
            w->errors++;
            if (w->errors > 16) break;
            continue;
        }
        w->bytes += (uint64_t)r;
        w->reads++;
        if (w->nlat < LAT_CAP) w->lat[w->nlat++] = (float)((t1 - t0) * 1e6);
    }
    free(buf);
    return NULL;
}

static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

/* ---------- file creation ---------- */

static int create_file(const char *path, uint64_t size) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { perror("create"); return -1; }
    size_t chunk = 4u << 20;
    uint64_t *b = malloc(chunk);
    for (uint64_t done = 0; done < size; done += chunk) {
        for (size_t i = 0; i < chunk / 8; i++) b[i] = xorshift(); /* incompressible */
        size_t n = (size - done) < chunk ? (size_t)(size - done) : chunk;
        if (write(fd, b, n) != (ssize_t)n) { perror("write"); free(b); close(fd); return -1; }
        if ((done / chunk) % 256 == 0)
            fprintf(stderr, "  created %llu / %llu MB\n", (unsigned long long)(done >> 20),
                    (unsigned long long)(size >> 20));
    }
    fsync(fd);
    free(b);
    close(fd);
    return 0;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    const char *file = NULL, *out = "ufsbench.csv";
    uint64_t size_mb = 0;
    int create = 0, repeats = 1, power = 0;
    double seconds = 3.0;
    long sizes[32], thr[16], cpus[64];
    int nsizes = parse_list("4,8,16,32,64,128,256,512,1024,2048,4096", sizes, 32);
    int nthr = parse_list("1,2,4,8", thr, 16), ncpus = 0;
    int do_direct = 1, do_buffered = 1, do_rand = 1, do_seq = 1;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--file")) file = v, i++;
        else if (!strcmp(a, "--size-mb")) size_mb = strtoull(v, NULL, 10), i++;
        else if (!strcmp(a, "--create")) create = 1;
        else if (!strcmp(a, "--seconds")) seconds = atof(v), i++;
        else if (!strcmp(a, "--repeats")) repeats = atoi(v), i++;
        else if (!strcmp(a, "--sizes-kb")) nsizes = parse_list(v, sizes, 32), i++;
        else if (!strcmp(a, "--threads")) nthr = parse_list(v, thr, 16), i++;
        else if (!strcmp(a, "--cpus")) ncpus = parse_list(v, cpus, 64), i++;
        else if (!strcmp(a, "--power")) power = 1;
        else if (!strcmp(a, "--out")) out = v, i++;
        else if (!strcmp(a, "--modes")) { do_direct = !!strstr(v, "direct"); do_buffered = !!strstr(v, "buffered"); i++; }
        else if (!strcmp(a, "--patterns")) { do_rand = !!strstr(v, "rand"); do_seq = !!strstr(v, "seq"); i++; }
        else { fprintf(stderr, "unknown argument %s\n", a); return 2; }
    }
    if (!file || !size_mb) {
        fprintf(stderr, "usage: ufsbench --file PATH --size-mb N [--create] ... (see source header)\n");
        return 2;
    }
    uint64_t fsize = size_mb << 20;
    rng_state ^= (uint64_t)time(NULL);

    struct stat st;
    if (create || stat(file, &st) || (uint64_t)st.st_size < fsize) {
        fprintf(stderr, "creating %s (%llu MB) ...\n", file, (unsigned long long)size_mb);
        if (create_file(file, fsize)) return 1;
    }

    FILE *csv = fopen(out, "w");
    if (!csv) { perror("out"); return 1; }
    const char *hdr = "repeat,mode,pattern,size_kb,threads,bytes,seconds,MBps,iops,"
                      "lat_p50_us,lat_p99_us,errors,first_errno,power_w,temp_start_c,temp_end_c\n";
    fputs(hdr, csv);
    fputs(hdr, stdout);

    /* Idle power baseline, so per-GB energy can subtract it later. */
    if (power) {
        power_t p = {0};
        pthread_t pt;
        pthread_create(&pt, NULL, power_thread, &p);
        sleep(5);
        atomic_store(&p.stop, 1);
        pthread_join(pt, NULL);
        char line[256];
        snprintf(line, sizeof line, "0,idle,none,0,0,0,5,0,0,0,0,0,0,%.4f,%.1f,%.1f\n",
                 p.n ? p.sum_w / p.n : NAN, max_thermal_c(), max_thermal_c());
        fputs(line, csv); fputs(line, stdout); fflush(csv);
    }

    int direct_ok = 1;
    for (int rep = 1; rep <= repeats; rep++)
    for (int m = 0; m < 2; m++) {
        int direct = (m == 0);
        if ((direct && !do_direct) || (!direct && !do_buffered)) continue;
        if (direct && !direct_ok) continue;
        int fd = open(file, O_RDONLY | (direct ? O_DIRECT : 0));
        if (fd < 0) {
            if (direct) { direct_ok = 0; fprintf(stderr, "O_DIRECT open refused (errno %d) — direct mode unsupported here\n", errno); continue; }
            perror("open"); return 1;
        }
        if (direct) { /* probe: some filesystems accept the flag but fail the read */
            void *pb = NULL;
            posix_memalign(&pb, ALIGN, ALIGN);
            if (pread(fd, pb, ALIGN, 0) != ALIGN) { direct_ok = 0; fprintf(stderr, "O_DIRECT read refused (errno %d) — direct mode unsupported here\n", errno); free(pb); close(fd); continue; }
            free(pb);
        }
        for (int pat = 0; pat < 2; pat++) {
            int seq = (pat == 1);
            if ((seq && !do_seq) || (!seq && !do_rand)) continue;
            for (int si = 0; si < nsizes; si++) {
                size_t req = (size_t)sizes[si] << 10;
                if (req % ALIGN || req > fsize) continue;
                if (seq && sizes[si] < 64) continue; /* sequential small reads are not a regime we use */
                uint64_t nslots = fsize / req;
                uint64_t *perm = NULL;
                if (!seq) {
                    perm = malloc(nslots * sizeof *perm);
                    for (uint64_t k = 0; k < nslots; k++) perm[k] = k;
                    for (uint64_t k = nslots - 1; k > 0; k--) { uint64_t j = xorshift() % (k + 1); uint64_t t = perm[k]; perm[k] = perm[j]; perm[j] = t; }
                }
                for (int ti = 0; ti < nthr; ti++) {
                    int nt = (int)thr[ti];
                    /* Drop this file's cached pages; set access-pattern advice. */
                    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                    posix_fadvise(fd, 0, 0, seq ? POSIX_FADV_SEQUENTIAL : POSIX_FADV_RANDOM);
                    _Atomic uint64_t next = 0;
                    worker_t *w = calloc((size_t)nt, sizeof *w);
                    pthread_t *tid = calloc((size_t)nt, sizeof *tid);
                    power_t p = {0};
                    pthread_t pt;
                    double temp0 = max_thermal_c();
                    if (power) pthread_create(&pt, NULL, power_thread, &p);
                    double t0 = now_s();
                    for (int k = 0; k < nt; k++) {
                        w[k].fd = fd; w[k].seq = seq; w[k].req = req; w[k].perm = perm;
                        w[k].nslots = nslots; w[k].next = &next; w[k].deadline = t0 + seconds;
                        w[k].cpu = ncpus ? (int)cpus[k % ncpus] : -1;
                        uint64_t span = (fsize / (uint64_t)nt) / req * req;
                        w[k].seq_lo = (uint64_t)k * span; w[k].seq_hi = w[k].seq_lo + span;
                        w[k].lat = malloc(LAT_CAP * sizeof(float));
                        pthread_create(&tid[k], NULL, worker, &w[k]);
                    }
                    uint64_t bytes = 0, reads = 0, errs = 0, nl = 0;
                    int ferr = 0;
                    for (int k = 0; k < nt; k++) {
                        pthread_join(tid[k], NULL);
                        bytes += w[k].bytes; reads += w[k].reads; errs += w[k].errors; nl += w[k].nlat;
                        if (!ferr) ferr = w[k].first_errno;
                    }
                    double el = now_s() - t0;
                    if (power) { atomic_store(&p.stop, 1); pthread_join(pt, NULL); }
                    float *all = malloc((nl ? nl : 1) * sizeof(float));
                    uint64_t c = 0;
                    for (int k = 0; k < nt; k++) { memcpy(all + c, w[k].lat, w[k].nlat * sizeof(float)); c += w[k].nlat; free(w[k].lat); }
                    qsort(all, nl, sizeof(float), cmp_float);
                    double p50 = nl ? all[nl / 2] : NAN, p99 = nl ? all[(uint64_t)(nl * 0.99)] : NAN;
                    char line[512];
                    snprintf(line, sizeof line, "%d,%s,%s,%ld,%d,%llu,%.4f,%.2f,%.1f,%.1f,%.1f,%llu,%d,%.4f,%.1f,%.1f\n",
                             rep, direct ? "direct" : "buffered", seq ? "seq" : "rand", sizes[si], nt,
                             (unsigned long long)bytes, el, bytes / el / 1e6, reads / el, p50, p99,
                             (unsigned long long)errs, ferr, power && p.n ? p.sum_w / p.n : NAN,
                             temp0, max_thermal_c());
                    fputs(line, csv); fputs(line, stdout); fflush(csv); fflush(stdout);
                    free(all); free(w); free(tid);
                }
                free(perm);
            }
        }
        close(fd);
    }
    fprintf(csv, "# direct_io_supported=%d\n", direct_ok);
    fclose(csv);
    fprintf(stderr, "done; direct I/O %s. Results in %s\n", direct_ok ? "supported" : "UNSUPPORTED", out);
    return 0;
}
