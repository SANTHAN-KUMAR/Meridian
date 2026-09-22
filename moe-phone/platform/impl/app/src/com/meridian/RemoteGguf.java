package com.meridian.app;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Reads a GGUF's header over HTTP Range requests, so a model can be evaluated (ModelCard, fit, predicted speed) BEFORE the
 *  user commits gigabytes to downloading it. Only the first few MiB are fetched: the header holds every tensor's name,
 *  shape, type and offset, and the layout cross-check in Gguf.parse needs only those plus the total size (from
 *  Content-Range). Starts at 4 MiB and quadruples while the header is truncated (tokenizer arrays can be large). */
public final class RemoteGguf {
    public static final class Result { public Gguf gguf; public long totalBytes, fetchedBytes; public String finalUrl; }
    static final Pattern RANGE = Pattern.compile("bytes \\d+-\\d+/(\\d+)");
    public static final long MAX_HEADER = 256L << 20;

    public static Result fetch(String url) throws IOException, Gguf.GgufError {
        long n = 4L << 20;
        while (true) {
            HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
            c.setConnectTimeout(20000); c.setReadTimeout(30000); c.setInstanceFollowRedirects(true);
            c.setRequestProperty("Range", "bytes=0-" + (n - 1)); c.setRequestProperty("User-Agent", "meridian/0.2");
            byte[] buf; long total;
            try {
                int code = c.getResponseCode();
                if (code == 206) {
                    Matcher m = RANGE.matcher(String.valueOf(c.getHeaderField("Content-Range")));
                    if (!m.find()) throw new IOException("206 without a parsable Content-Range: " + c.getHeaderField("Content-Range"));
                    total = Long.parseLong(m.group(1));
                } else if (code == 200) total = c.getContentLengthLong();   // server ignored Range: read the prefix, then drop the connection
                else throw new IOException("HTTP " + code + " for " + url);
                if (total <= 0) throw new IOException("server did not report the file size");
                buf = readUpTo(c.getInputStream(), (int) Math.min(n, total));
            } finally { c.disconnect(); }
            try {
                Result r = new Result(); r.gguf = Gguf.parse(new ByteArrayInputStream(buf), total); r.totalBytes = total; r.fetchedBytes = buf.length; r.finalUrl = url;
                return r;
            } catch (Gguf.GgufError e) {
                if (!String.valueOf(e.getMessage()).contains("truncated") || n >= total || n >= MAX_HEADER) throw e;
                n = Math.min(n * 4, MAX_HEADER);
            }
        }
    }

    static byte[] readUpTo(InputStream in, int n) throws IOException {
        ByteArrayOutputStream o = new ByteArrayOutputStream(Math.min(n, 1 << 22)); byte[] b = new byte[1 << 16]; int r, left = n;
        try (InputStream s = in) { while (left > 0 && (r = s.read(b, 0, Math.min(b.length, left))) > 0) { o.write(b, 0, r); left -= r; } }
        return o.toByteArray();
    }
}
