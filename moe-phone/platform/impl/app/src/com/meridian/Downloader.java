package com.meridian.app;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;

/** Resumable HTTP(S) model download into `<name>.part`, renamed only after the header validates as GGUF and,
 *  if given, the SHA-256 matches (quarantine-until-verified, 02_ARCHITECTURE.md section 3.4). */
public final class Downloader {
    public interface Progress { void update(long done, long total); }
    public volatile boolean cancelled;

    public File download(String url, File dir, String sha256, Progress p) throws Exception {
        dir.mkdirs();
        String name = url.substring(url.lastIndexOf('/') + 1); int q = name.indexOf('?'); if (q >= 0) name = name.substring(0, q);
        if (!name.endsWith(".gguf")) throw new IOException("URL must point at a .gguf file (got '" + name + "')");
        File part = new File(dir, name + ".part"), fin = new File(dir, name);
        long have = part.exists() ? part.length() : 0;
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(20000); c.setReadTimeout(30000); c.setInstanceFollowRedirects(true);
        if (have > 0) c.setRequestProperty("Range", "bytes=" + have + "-");
        int code = c.getResponseCode();
        if (code == 416) { have = 0; part.delete(); c.disconnect(); return download(url, dir, sha256, p); }
        if (code != 200 && code != 206) throw new IOException("HTTP " + code);
        if (code == 200) have = 0;                                   // server ignored Range: restart
        long total = have + c.getContentLengthLong();
        File st = dir; if (st.getUsableSpace() < total - have + (256L << 20)) throw new IOException("not enough free space for " + (total >> 20) + " MiB");
        try (InputStream in = c.getInputStream(); RandomAccessFile out = new RandomAccessFile(part, "rw")) {
            out.seek(have); out.setLength(have);
            byte[] buf = new byte[1 << 17]; int n; long done = have, last = 0;
            while ((n = in.read(buf)) > 0) {
                if (cancelled) throw new IOException("cancelled (partial kept for resume)");
                out.write(buf, 0, n); done += n;
                if (System.currentTimeMillis() - last > 500) { p.update(done, total); last = System.currentTimeMillis(); }
            }
            if (total > 0 && done != total) throw new IOException("short read: " + done + " of " + total);
            p.update(done, total);
        } finally { c.disconnect(); }
        try { Gguf.read(part); } catch (Gguf.GgufError e) { part.renameTo(new File(dir, name + ".quarantine")); throw new IOException("IntegrityFailed: not a valid GGUF (" + e.getMessage() + "); quarantined"); }
        if (sha256 != null && !sha256.isEmpty()) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            try (InputStream in = new BufferedInputStream(new FileInputStream(part), 1 << 20)) { byte[] b = new byte[1 << 20]; int n; while ((n = in.read(b)) > 0) md.update(b, 0, n); }
            StringBuilder h = new StringBuilder(); for (byte x : md.digest()) h.append(String.format("%02x", x));
            if (!h.toString().equalsIgnoreCase(sha256.trim())) { part.renameTo(new File(dir, name + ".quarantine")); throw new IOException("IntegrityFailed: sha256 mismatch; quarantined"); }
        }
        if (!part.renameTo(fin)) throw new IOException("could not finalise download");
        return fin;
    }
}
