package com.meridian.app;

import org.json.*;
import java.util.*;

/** Calibration cells for one (model, config) and the interpolating performance model of 05_PERFORMANCE_MODEL.md sections 2-5.
 *  Prefill and decode are separate quantities. A prediction is `measured` (a valid cell within 15% of the requested size),
 *  `interpolated` (inside the measured range), `extrapolated` (outside it; interval bracketed by the nearest cell and the
 *  linear-in-log trend, plus the observed repeat spread), or refused as `uncalibrated`. Intervals come from repeats and trend
 *  differences only; no constant is tuned. */
public final class Cells {
    /** One measured point: x = prompt tokens (prefill) or context tokens (decode); value = tokens/s; lo/hi = min/max over repeats. */
    public static final class Pt { public double x, v, lo, hi; public int n; }
    public final List<Pt> prefill = new ArrayList<>(), decode = new ArrayList<>();
    public boolean inRegime; public double loadS = -1;
    public double mapA = 0, mapB = 1;   // measured: actual prompt tokens = mapA + mapB * requested approx tokens (least squares over the calibration runs)
    public double promptTokens(double requested) { return mapA + mapB * requested; }

    public static Cells fromJson(JSONObject j) throws JSONException {
        Cells c = new Cells(); c.inRegime = j.optBoolean("in_regime"); c.loadS = j.optDouble("load_s", -1); c.mapA = j.optDouble("map_a", 0); c.mapB = j.optDouble("map_b", 1);
        for (String k : new String[]{"prefill", "decode"}) { JSONArray a = j.getJSONArray(k); for (int i = 0; i < a.length(); i++) { JSONObject o = a.getJSONObject(i); Pt p = new Pt();
            p.x = o.getDouble("x"); p.v = o.getDouble("v"); p.lo = o.getDouble("lo"); p.hi = o.getDouble("hi"); p.n = o.getInt("n"); (k.equals("prefill") ? c.prefill : c.decode).add(p); } }
        Comparator<Pt> cmp = (a, b) -> Double.compare(a.x, b.x); Collections.sort(c.prefill, cmp); Collections.sort(c.decode, cmp); return c;
    }
    public JSONObject toJson() throws JSONException {
        JSONObject j = new JSONObject().put("in_regime", inRegime).put("load_s", loadS).put("map_a", mapA).put("map_b", mapB);
        for (String k : new String[]{"prefill", "decode"}) { JSONArray a = new JSONArray(); for (Pt p : k.equals("prefill") ? prefill : decode)
            a.put(new JSONObject().put("x", p.x).put("v", p.v).put("lo", p.lo).put("hi", p.hi).put("n", p.n)); j.put(k, a); }
        return j;
    }
    /** Group raw (x, v) observations into points: observations whose x is within 15% of a group's first x are one point. */
    public static List<Pt> group(List<double[]> obs) {
        List<double[]> s = new ArrayList<>(obs); Collections.sort(s, (a, b) -> Double.compare(a[0], b[0]));
        List<Pt> out = new ArrayList<>(); List<double[]> cur = new ArrayList<>();
        for (double[] o : s) { if (!cur.isEmpty() && o[0] > cur.get(0)[0] * 1.15) { out.add(mk(cur)); cur = new ArrayList<>(); } cur.add(o); }
        if (!cur.isEmpty()) out.add(mk(cur)); return out;
    }
    private static Pt mk(List<double[]> g) {
        Pt p = new Pt(); double sx = 0; List<Double> v = new ArrayList<>(); for (double[] o : g) { sx += o[0]; v.add(o[1]); } Collections.sort(v);
        p.x = sx / g.size(); p.v = v.get(v.size() / 2); p.lo = v.get(0); p.hi = v.get(v.size() - 1); p.n = g.size(); return p;
    }

    /** Prediction for x on `pts`. Returns {value, lo, hi, state, basis}; throws NotCalibrated-style IllegalStateException if no cells. */
    public static JSONObject predict(List<Pt> pts, double x, boolean inRegime) throws JSONException {
        if (pts.isEmpty()) throw new IllegalStateException("uncalibrated");
        JSONObject o = new JSONObject(); String state; double v, lo, hi; JSONArray basis = new JSONArray();
        Pt nearest = pts.get(0); for (Pt p : pts) if (Math.abs(Math.log(p.x / x)) < Math.abs(Math.log(nearest.x / x))) nearest = p;
        if (Math.abs(nearest.x / x - 1) <= 0.15) { state = "measured"; v = nearest.v; lo = nearest.lo; hi = nearest.hi; basis.put("cell x=" + Math.round(nearest.x) + " n=" + nearest.n); }
        else if (x > pts.get(0).x && x < pts.get(pts.size() - 1).x) {
            Pt a = pts.get(0), b = pts.get(1); for (int i = 0; i + 1 < pts.size(); i++) if (x >= pts.get(i).x && x <= pts.get(i + 1).x) { a = pts.get(i); b = pts.get(i + 1); }
            double t = Math.log(x / a.x) / Math.log(b.x / a.x);
            v = a.v + t * (b.v - a.v); lo = a.lo + t * (b.lo - a.lo); hi = a.hi + t * (b.hi - a.hi); state = "interpolated"; basis.put("cells x=" + Math.round(a.x) + "," + Math.round(b.x));
            lo = Math.min(lo, Math.min(a.lo, b.lo)); hi = Math.max(hi, Math.max(a.hi, b.hi));   // a monotone trend is not assumed between cells
        } else {
            state = "extrapolated"; Pt e = x < pts.get(0).x ? pts.get(0) : pts.get(pts.size() - 1); Pt nx = pts.size() > 1 ? (x < pts.get(0).x ? pts.get(1) : pts.get(pts.size() - 2)) : e;
            double slope = e == nx ? 0 : (e.v - nx.v) / Math.log(e.x / nx.x); v = e.v + slope * Math.log(x / e.x);
            lo = Math.min(Math.min(e.lo, v), e.v); hi = Math.max(Math.max(e.hi, v), e.v);
            if (pts.size() == 1) { lo = Math.min(lo, e.lo); hi = Math.max(hi, e.hi); }
            basis.put("nearest cell x=" + Math.round(e.x) + " and trend with x=" + Math.round(nx.x));
        }
        if (v < 0) v = 0; return o.put("value", v).put("lo", Math.max(0, lo)).put("hi", hi).put("state", state).put("basis", basis).put("provenance", inRegime && !state.equals("extrapolated") ? "calibrated" : "prior");
    }

    /** TTFT for n_new prompt tokens: prefill time + first decode step, as bounds from the end-to-end prefill and decode predictions (same wall clock). */
    public JSONObject ttftMs(double nNew, double ctx) throws JSONException {
        JSONObject p = predict(prefill, nNew, inRegime), d = predict(decode, ctx, inRegime);
        double lo = nNew / Math.max(1e-9, p.getDouble("hi")) + 1.0 / Math.max(1e-9, d.getDouble("hi")), hi = nNew / Math.max(1e-9, p.getDouble("lo")) + 1.0 / Math.max(1e-9, d.getDouble("lo"));
        double mid = nNew / Math.max(1e-9, p.getDouble("value")) + 1.0 / Math.max(1e-9, d.getDouble("value"));
        String st = p.getString("state").equals("extrapolated") || d.getString("state").equals("extrapolated") ? "extrapolated" : (p.getString("state").equals("measured") && d.getString("state").equals("measured") ? "measured" : "interpolated");
        return new JSONObject().put("value_ms", mid * 1000).put("lo_ms", lo * 1000).put("hi_ms", hi * 1000).put("state", st).put("prefill", p).put("decode", d);
    }
}
