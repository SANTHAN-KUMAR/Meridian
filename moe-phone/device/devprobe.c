/*
 * devprobe — gate G0: which accelerator device nodes can an ORDINARY Android app
 * open, and with which access mode?
 *
 * Why a program rather than `ls` or `cat`: both conflate three different answers.
 * `ls -e` reports "No such file or directory" for a node that exists but whose
 * directory entry cannot be stat'd, and `cat` reports only the read path. Only
 * open(2) with each mode, reporting errno, distinguishes
 *     ENOENT  absent
 *     EACCES  present, policy denies
 *     success reachable.
 *
 * WHAT THIS PROBE CANNOT TELL YOU (learned the hard way, 2026-09-14):
 * a denied open() here does NOT mean the accelerator is unreachable. On the
 * OnePlus 15R every /dev/fastrpc-* node gives EACCES to Termux
 * (untrusted_app_27) and to `shell` -- and ALSO to an ordinary app that then
 * proceeds to use the NPU successfully. Qualcomm's FastRPC client probes
 * several domains, is refused on the secure ones, and continues on the
 * permitted one; the 213 denials logged during a Geekbench AI QNN run are
 * interleaved with 18 successful loads of libQnnHtpV81Skel.so onto the DSP.
 * Reading EACCES here as "no NPU" was a wrong inference published in this
 * repository and later retracted (POSITION.md stub S7).
 *
 * So: use this program for what it does measure -- which nodes exist and which
 * a given domain may open -- and answer REACHABILITY with a real workload plus
 * a logcat capture, never with this alone.
 *
 * Result on the 15R (ColorOS 16, unrooted): all /dev/fastrpc-* EACCES in both
 * modes from untrusted_app_27 and shell; /dev/kgsl-3d0 opens read-write.
 *
 * Build (Termux): clang -O2 devprobe.c -o devprobe
 * Run:            ./devprobe
 */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
int main(void) {
    const char *paths[] = {"/dev/fastrpc-cdsp", "/dev/fastrpc-cdsp-secure",
                           "/dev/fastrpc-nsp1000", "/dev/fastrpc-adsp-secure",
                           "/dev/kgsl-3d0", "/dev/dma_heap/system",
                           "/dev/dma_heap/qcom,system", NULL};
    int modes[] = {O_RDONLY, O_RDWR};
    const char *mn[] = {"O_RDONLY", "O_RDWR  "};
    for (int i = 0; paths[i]; i++)
        for (int m = 0; m < 2; m++) {
            int fd = open(paths[i], modes[m]);
            printf("%-26s %s  %s\n", paths[i], mn[m],
                   fd >= 0 ? "OPEN_OK" : strerror(errno));
            if (fd >= 0) close(fd);
        }
    return 0;
}
