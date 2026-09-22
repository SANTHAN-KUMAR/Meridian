package com.meridian.app;

import android.animation.ValueAnimator;
import android.app.Dialog;
import android.content.Context;
import android.content.res.ColorStateList;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;

/** Meridian components (brand kit "Components" board), built from plain framework views so the app stays dependency-free.
 *  Rules: 48dp minimum touch targets, flat fills with 1dp lines, one animated element at most (the sun pulse). */
final class UiKit {
    final Context c; final UiTheme t;
    UiKit(Context c, UiTheme t) { this.c = c; this.t = t; }
    static final int MATCH = ViewGroup.LayoutParams.MATCH_PARENT, WRAP = ViewGroup.LayoutParams.WRAP_CONTENT;

    // ---------- layout ----------
    LinearLayout col() { LinearLayout l = new LinearLayout(c); l.setOrientation(LinearLayout.VERTICAL); return l; }
    LinearLayout row() { LinearLayout l = new LinearLayout(c); l.setOrientation(LinearLayout.HORIZONTAL); l.setGravity(Gravity.CENTER_VERTICAL); return l; }
    LinearLayout.LayoutParams lp(int w, int h) { return new LinearLayout.LayoutParams(w, h); }
    LinearLayout.LayoutParams lpGap(int topDp) { LinearLayout.LayoutParams p = lp(MATCH, WRAP); p.topMargin = t.dp(topDp); return p; }
    LinearLayout.LayoutParams weight() { return new LinearLayout.LayoutParams(0, WRAP, 1f); }
    void add(LinearLayout parent, View v, int gapDp) { parent.addView(v, lpGap(gapDp)); }
    View space(int dp) { View v = new View(c); v.setLayoutParams(new LinearLayout.LayoutParams(1, t.dp(dp))); return v; }
    View fill() { View v = new View(c); v.setLayoutParams(new LinearLayout.LayoutParams(0, 0, 1f)); return v; }
    ScrollView scroll(View child) { ScrollView s = new ScrollView(c); s.setFillViewport(true); s.setClipToPadding(false); s.addView(child); return s; }
    View divider() { View v = new View(c); v.setBackgroundColor(t.line); v.setLayoutParams(new LinearLayout.LayoutParams(MATCH, Math.max(1, t.dp(1)))); return v; }

    /** Caps content width (readable line length on tablets and unfolded foldables) and centres it. */
    static final class MaxWidth extends FrameLayout {
        final int max;
        MaxWidth(Context c, int maxPx) { super(c); max = maxPx; }
        @Override protected void onMeasure(int w, int h) {
            // always exact: the full screen width on phones, capped at `max` on tablets and unfolded foldables
            w = MeasureSpec.makeMeasureSpec(Math.min(MeasureSpec.getSize(w), max), MeasureSpec.EXACTLY);
            super.onMeasure(w, h);
        }
    }
    FrameLayout centered(View content) {
        FrameLayout outer = new FrameLayout(c); MaxWidth mw = new MaxWidth(c, t.dp(680));
        mw.addView(content, new FrameLayout.LayoutParams(MATCH, MATCH));
        outer.addView(mw, new FrameLayout.LayoutParams(WRAP, MATCH, Gravity.CENTER_HORIZONTAL));
        return outer;
    }

    // ---------- text ----------
    TextView text(CharSequence s, UiTheme.Text style, int color) {
        TextView v = new TextView(c); t.style(v, style); v.setTextColor(color); v.setText(s); return v;
    }
    TextView heading(CharSequence s, UiTheme.Text style) { TextView v = text(s, style, t.ink); v.setAccessibilityHeading(true); return v; }
    TextView section(String title) { TextView v = text(title, UiTheme.Text.OVERLINE, t.ink2); v.setAccessibilityHeading(true); return v; }

    // ---------- buttons ----------
    enum Kind { PRIMARY, SECONDARY, QUIET, DESTRUCTIVE, SUN }
    Button button(String label, Kind k, View.OnClickListener l) {
        Button b = new Button(c); b.setText(label); b.setAllCaps(false); b.setStateListAnimator(null);
        t.style(b, UiTheme.Text.LABEL); b.setMinHeight(t.dp(48)); b.setMinimumHeight(t.dp(48)); b.setMinWidth(t.dp(64)); b.setMinimumWidth(t.dp(64));
        b.setPadding(t.dp(20), 0, t.dp(20), 0); b.setGravity(Gravity.CENTER);
        int fill, fg; float stroke = 0; int strokeC = 0;
        switch (k) {
            case PRIMARY: fill = t.primary; fg = t.onPrimary; break;
            case SUN: fill = t.sun; fg = 0xFF141A2E; break;
            case SECONDARY: fill = Color.TRANSPARENT; fg = t.ink; stroke = 1.5f; strokeC = t.lineStrong; break;
            case DESTRUCTIVE: fill = Color.TRANSPARENT; fg = t.danger; stroke = 1.5f; strokeC = t.danger; break;
            default: fill = Color.TRANSPARENT; fg = t.ink; break;
        }
        b.setTextColor(fg); b.setBackground(t.pressable(t.rounded(fill, 999, stroke, strokeC), 999));
        if (l != null) b.setOnClickListener(l);
        return b;
    }
    Button button(String label, String icon, Kind k, View.OnClickListener l) {
        Button b = button(label, k, l);
        Drawable d = t.icon(c, icon, b.getCurrentTextColor());
        if (d != null) { d.setBounds(0, 0, t.dp(18), t.dp(18)); b.setCompoundDrawablesRelative(d, null, null, null); b.setCompoundDrawablePadding(t.dp(8)); }
        return b;
    }
    ImageButton iconButton(String icon, String description, int color, View.OnClickListener l) {
        ImageButton b = new ImageButton(c); b.setImageDrawable(t.icon(c, icon, color)); b.setContentDescription(description);
        b.setBackground(t.pressable(new ColorDrawable(Color.TRANSPARENT), 999)); b.setScaleType(ImageView.ScaleType.CENTER);
        b.setLayoutParams(new LinearLayout.LayoutParams(t.dp(48), t.dp(48))); b.setOnClickListener(l);
        return b;
    }
    /** A round filled icon button (composer send, etc.). */
    ImageButton roundButton(String icon, String description, int fill, int fg, int sizeDp, View.OnClickListener l) {
        ImageButton b = new ImageButton(c); b.setImageDrawable(t.icon(c, icon, fg)); b.setContentDescription(description);
        b.setBackground(t.pressable(t.rounded(fill, 999), 999)); b.setScaleType(ImageView.ScaleType.CENTER);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(t.dp(sizeDp), t.dp(sizeDp)); b.setLayoutParams(p); b.setOnClickListener(l);
        return b;
    }
    ImageView iconView(String icon, int color, int sizeDp) {
        ImageView v = new ImageView(c); v.setImageDrawable(t.icon(c, icon, color)); v.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        v.setLayoutParams(new LinearLayout.LayoutParams(t.dp(sizeDp), t.dp(sizeDp))); return v;
    }
    /** An icon on a tinted circle (feature rows, problem cards). */
    FrameLayout badge(String icon, int tint, int fg, int sizeDp) {
        FrameLayout f = new FrameLayout(c); f.setBackground(t.rounded(tint, 999));
        ImageView iv = new ImageView(c); iv.setImageDrawable(t.icon(c, icon, fg));
        int s = t.dp(sizeDp * 0.5f); f.addView(iv, new FrameLayout.LayoutParams(s, s, Gravity.CENTER));
        f.setLayoutParams(new LinearLayout.LayoutParams(t.dp(sizeDp), t.dp(sizeDp))); f.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        return f;
    }

    // ---------- containers ----------
    LinearLayout card() { LinearLayout l = col(); l.setBackground(t.rounded(t.surface, 18, 1, t.line)); l.setPadding(t.dp(16), t.dp(14), t.dp(16), t.dp(14)); return l; }
    LinearLayout tintCard(int tint) { LinearLayout l = col(); l.setBackground(t.rounded(tint, 18)); l.setPadding(t.dp(16), t.dp(14), t.dp(16), t.dp(14)); return l; }
    LinearLayout pressCard(View.OnClickListener l) {
        LinearLayout v = card(); v.setBackground(t.pressable(t.rounded(t.surface, 18, 1, t.line), 18)); v.setClickable(true); v.setFocusable(true); v.setOnClickListener(l); return v;
    }

    /** One compact line of state: an icon and a sentence on a tinted pill (offline badge, warm phone). */
    LinearLayout pill(String icon, String label, int tint, int fg) {
        LinearLayout r = row(); r.setBackground(t.rounded(tint, 999)); r.setPadding(t.dp(12), t.dp(6), t.dp(14), t.dp(6));
        if (icon != null) { r.addView(iconView(icon, fg, 16)); r.addView(space(0), new LinearLayout.LayoutParams(t.dp(8), 1)); }
        r.addView(text(label, UiTheme.Text.CAPTION, fg));
        return r;
    }

    // ---------- confidence chip: shape + colour + word (never colour alone) ----------
    enum Basis { MEASURED, CALIBRATED, ESTIMATED }
    static Basis basisOf(String provenance) {
        if (provenance == null) return Basis.ESTIMATED;
        if (provenance.startsWith("measured")) return Basis.MEASURED;
        if (provenance.startsWith("calibrated")) return Basis.CALIBRATED;
        return Basis.ESTIMATED;
    }
    LinearLayout chip(Basis b) {
        int tint, fg; String word;
        switch (b) { case MEASURED: tint = t.goodTint; fg = t.good; word = "Measured"; break;
            case CALIBRATED: tint = t.calibTint; fg = t.calib; word = "Calibrated"; break;
            default: tint = t.estTint; fg = t.est; word = "Estimated"; }
        LinearLayout r = row(); r.setBackground(t.rounded(tint, 999)); r.setPadding(t.dp(8), t.dp(4), t.dp(10), t.dp(4));
        View dot = new BasisDot(c, b, fg, tint); r.addView(dot, new LinearLayout.LayoutParams(t.dp(14), t.dp(14)));
        TextView w = text(word, UiTheme.Text.CAPTION, fg); LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(WRAP, WRAP); p.leftMargin = t.dp(6); r.addView(w, p);
        r.setContentDescription(word + (b == Basis.MEASURED ? ": measured on this phone" : b == Basis.CALIBRATED ? ": predicted from this phone's own tests" : ": a guess from similar phones"));
        return r;
    }
    static final class BasisDot extends View {
        final Basis b; final int fg, bg; final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        BasisDot(Context c, Basis b, int fg, int bg) { super(c); this.b = b; this.fg = fg; this.bg = bg; }
        static final int SIZE_DP = 14;
        @Override protected void onMeasure(int ws, int hs) { int d = Math.round(SIZE_DP * getResources().getDisplayMetrics().density);   // never grows with WRAP/MATCH params
            setMeasuredDimension(MeasureSpec.getMode(ws) == MeasureSpec.EXACTLY ? Math.min(MeasureSpec.getSize(ws), d * 2) : d, MeasureSpec.getMode(hs) == MeasureSpec.EXACTLY ? Math.min(MeasureSpec.getSize(hs), d * 2) : d); }

        @Override protected void onDraw(Canvas cv) {
            float w = getWidth(), h = getHeight(), r = Math.min(w, h) / 2f - 1.5f * getResources().getDisplayMetrics().density / 2f;
            float cx = w / 2f, cy = h / 2f, sw = 1.5f * getResources().getDisplayMetrics().density; p.setColor(fg);
            if (b == Basis.MEASURED) { p.setStyle(Paint.Style.FILL); cv.drawCircle(cx, cy, r, p);
                p.setColor(bg); p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(sw); p.setStrokeCap(Paint.Cap.ROUND);
                cv.drawLine(cx - r * 0.45f, cy + r * 0.02f, cx - r * 0.1f, cy + r * 0.38f, p); cv.drawLine(cx - r * 0.1f, cy + r * 0.38f, cx + r * 0.5f, cy - r * 0.3f, p); }
            else if (b == Basis.CALIBRATED) { p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(sw); cv.drawCircle(cx, cy, r - sw / 2, p);
                p.setStyle(Paint.Style.FILL); cv.drawArc(new RectF(cx - r, cy - r, cx + r, cy + r), -90, 180, true, p); }
            else { p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(sw); p.setPathEffect(new DashPathEffect(new float[]{sw * 1.6f, sw * 1.3f}, 0)); cv.drawCircle(cx, cy, r - sw / 2, p); p.setPathEffect(null); }
        }
    }

    // ---------- the sun pulse: shown only while Meridian works ----------
    /** A marigold dot that breathes, with a ring rippling out of it: unmistakably "working". Drawn on the animation clock only
     *  while attached; when the system's animations are off (Remove animations) it is a steady dot. Cheap: one small view. */
    static final class Pulse extends View {
        final int core; final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG); float f = 0; ValueAnimator a;
        Pulse(Context c, int core) { super(c); this.core = core; setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO); }
        static final int SIZE_DP = 22;
        @Override protected void onMeasure(int ws, int hs) { int d = Math.round(SIZE_DP * getResources().getDisplayMetrics().density);   // never grows with WRAP/MATCH params
            setMeasuredDimension(MeasureSpec.getMode(ws) == MeasureSpec.EXACTLY ? Math.min(MeasureSpec.getSize(ws), d * 2) : d, MeasureSpec.getMode(hs) == MeasureSpec.EXACTLY ? Math.min(MeasureSpec.getSize(hs), d * 2) : d); }

        @Override protected void onAttachedToWindow() { super.onAttachedToWindow();
            if (!ValueAnimator.areAnimatorsEnabled()) return;
            a = ValueAnimator.ofFloat(0f, 1f); a.setDuration(1400); a.setRepeatCount(ValueAnimator.INFINITE); a.setInterpolator(new android.view.animation.LinearInterpolator());
            a.addUpdateListener(v -> { f = (float) v.getAnimatedValue(); invalidate(); }); a.start(); }
        @Override protected void onDetachedFromWindow() { if (a != null) a.cancel(); a = null; super.onDetachedFromWindow(); }
        @Override protected void onDraw(Canvas cv) {
            float cx = getWidth() / 2f, cy = getHeight() / 2f, r = Math.min(cx, cy);
            if (a != null) {   // the ripple: grows from the dot's edge to the view's edge while fading out
                float e = 1f - (1f - f) * (1f - f);
                p.setStyle(Paint.Style.STROKE); p.setStrokeWidth(r * 0.14f); p.setColor(core); p.setAlpha((int) (200 * (1f - f)));
                cv.drawCircle(cx, cy, r * (0.42f + 0.52f * e), p);
            }
            float breathe = a == null ? 1f : 0.72f + 0.28f * (float) Math.cos(2 * Math.PI * f);   // the dot itself breathes
            p.setStyle(Paint.Style.FILL); p.setColor(core); p.setAlpha((int) (255 * (0.55f + 0.45f * breathe)));
            cv.drawCircle(cx, cy, r * (0.34f + 0.06f * breathe), p);
        }
    }
    Pulse pulse() { Pulse v = new Pulse(c, t.sun); v.setLayoutParams(new LinearLayout.LayoutParams(t.dp(22), t.dp(22))); return v; }
    Pulse pulse(int sizeDp) { Pulse v = new Pulse(c, t.sun); v.setLayoutParams(new LinearLayout.LayoutParams(t.dp(sizeDp), t.dp(sizeDp))); return v; }

    // ---------- motion helpers (all skipped when the system's animations are off) ----------
    static boolean motion() { return ValueAnimator.areAnimatorsEnabled(); }
    /** A new item arriving: fade in and rise 8dp. Used once per item, never on a plain refresh. */
    void appear(View v, long delayMs) {
        if (!motion()) return;
        v.setAlpha(0f); v.setTranslationY(t.dp(8));
        v.animate().alpha(1f).translationY(0).setStartDelay(delayMs).setDuration(240).setInterpolator(new android.view.animation.DecelerateInterpolator()).start();
    }
    /** A step or badge that just completed: a small pop. */
    void pop(View v) {
        if (!motion()) return;
        v.setScaleX(0.4f); v.setScaleY(0.4f); v.setAlpha(0f);
        v.animate().scaleX(1f).scaleY(1f).alpha(1f).setDuration(260).setInterpolator(new android.view.animation.OvershootInterpolator(2f)).start();
    }

    /** Thin determinate bar (downloads, steps done). No indeterminate shimmer: waiting is shown with the pulse and words. */
    static final class Bar extends View {
        final int track, fill; float value; final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG); final RectF r = new RectF();
        Bar(Context c, int track, int fill) { super(c); this.track = track; this.fill = fill; }
        /** Always a thin bar: a WRAP_CONTENT height would otherwise take all the space offered (15R, 2026-09-22: the plan bar
         *  filled its card and drew as a giant circle). Only an EXACT height from the parent is honoured. */
        @Override protected void onMeasure(int ws, int hs) {
            int thin = Math.round(6 * getResources().getDisplayMetrics().density);
            int h = MeasureSpec.getMode(hs) == MeasureSpec.EXACTLY ? Math.min(MeasureSpec.getSize(hs), thin * 2) : thin;
            setMeasuredDimension(getDefaultSize(getSuggestedMinimumWidth(), ws), h);
        }
        static final java.util.Map<String, Float> LAST = new java.util.HashMap<>();
        void set(float v) { value = Math.max(0, Math.min(1, v)); invalidate(); setContentDescription(Math.round(value * 100) + " percent"); }
        /** Glide from where the bar with this key last was to `v` (views are rebuilt on refresh; the motion should not restart). */
        void glide(String key, float v) {
            final float to = Math.max(0, Math.min(1, v)); Float from = LAST.get(key); LAST.put(key, to);
            if (from == null || !motion() || Math.abs(from - to) < 0.001f) { set(to); return; }
            set(from); ValueAnimator an = ValueAnimator.ofFloat(from, to); an.setDuration(450); an.setInterpolator(new android.view.animation.DecelerateInterpolator());
            an.addUpdateListener(x -> set((float) x.getAnimatedValue())); an.start();
        }
        @Override protected void onDraw(Canvas cv) { float h = getHeight(), w = getWidth(); r.set(0, 0, w, h); p.setColor(track); cv.drawRoundRect(r, h / 2, h / 2, p);
            if (value > 0) { r.set(0, 0, Math.max(h, w * value), h); p.setColor(fill); cv.drawRoundRect(r, h / 2, h / 2, p); } }
    }
    Bar bar(int fill) { Bar b = new Bar(c, t.surface2, fill); b.setLayoutParams(new LinearLayout.LayoutParams(MATCH, t.dp(6))); return b; }

    // ---------- steps ----------
    enum StepState { DONE, NOW, NEXT, FAILED, SKIPPED }
    LinearLayout step(StepState s, String title, String sub) {
        LinearLayout r = row(); r.setGravity(Gravity.TOP);
        View mark;
        switch (s) {
            case DONE: mark = badge("check", t.good, t.dark ? 0xFF0D1120 : 0xFFFFFFFF, 22); break;
            case FAILED: mark = badge("close", t.warmTint, t.warm, 22); break;
            case SKIPPED: mark = badge("close", t.surface2, t.ink3, 22); break;
            case NOW: mark = pulse(22); break;
            default: { View v = new View(c); v.setBackground(t.dashed(Color.TRANSPARENT, 999, t.lineStrong)); v.setLayoutParams(new LinearLayout.LayoutParams(t.dp(22), t.dp(22))); mark = v; }
        }
        r.addView(mark);
        LinearLayout txt = col(); LinearLayout.LayoutParams p = weight(); p.leftMargin = t.dp(12);
        TextView tt = text(title, UiTheme.Text.BODY, s == StepState.NEXT || s == StepState.SKIPPED ? t.ink3 : t.ink);
        if (s == StepState.NOW) tt.setTypeface(t.bodySemi);
        txt.addView(tt);
        if (sub != null && !sub.isEmpty()) txt.addView(text(sub, UiTheme.Text.CAPTION, t.ink2), lpGap(2));
        r.addView(txt, p);
        String st = s == StepState.DONE ? "Done: " : s == StepState.NOW ? "In progress: " : s == StepState.FAILED ? "Didn't work: " : s == StepState.SKIPPED ? "Skipped: " : "Next: ";
        r.setContentDescription(st + title + (sub == null ? "" : ". " + sub)); r.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_YES);
        return r;
    }

    // ---------- inputs ----------
    EditText field(String hint, boolean singleLine) {
        EditText e = new EditText(c); t.style(e, UiTheme.Text.BODY_L); e.setTextColor(t.ink); e.setHintTextColor(t.ink3); e.setHint(hint);
        e.setBackground(t.rounded(t.surface, 14, 1, t.line)); e.setPadding(t.dp(16), t.dp(12), t.dp(16), t.dp(12)); e.setMinHeight(t.dp(52));
        if (singleLine) { e.setSingleLine(true); e.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI); }
        return e;
    }
    /** The composer: always pinned to the bottom, within thumb reach. */
    static final class Composer { LinearLayout view; EditText input; ImageButton send, mic; }
    Composer composer(String hint, boolean withMic, View.OnClickListener onSend, View.OnClickListener onMic) {
        Composer k = new Composer();
        LinearLayout r = row(); r.setBackground(t.rounded(t.surface, 28, 1, t.line)); r.setPadding(t.dp(18), t.dp(6), t.dp(6), t.dp(6));
        EditText e = new EditText(c); t.style(e, UiTheme.Text.BODY_L); e.setTextColor(t.ink); e.setHintTextColor(t.ink3); e.setHint(hint); e.setBackground(null);
        e.setPadding(0, t.dp(8), 0, t.dp(8)); e.setMaxLines(5); e.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES | InputType.TYPE_TEXT_FLAG_MULTI_LINE);
        e.setImeOptions(EditorInfo.IME_ACTION_SEND); e.setContentDescription(hint);
        r.addView(e, weight());
        if (withMic) { k.mic = roundButton("mic", "Speak your task", t.surface2, t.ink, 44, onMic); LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(t.dp(44), t.dp(44)); p.leftMargin = t.dp(6); r.addView(k.mic, p); }
        k.send = roundButton("send", "Send", t.primary, t.onPrimary, 44, onSend); LinearLayout.LayoutParams p2 = new LinearLayout.LayoutParams(t.dp(44), t.dp(44)); p2.leftMargin = t.dp(6); r.addView(k.send, p2);
        final ImageButton send = k.send;
        send.setAlpha(0.35f); send.setEnabled(false);
        e.addTextChangedListener(new android.text.TextWatcher() { public void beforeTextChanged(CharSequence x, int a, int b, int d) { } public void onTextChanged(CharSequence x, int a, int b, int d) { }
            public void afterTextChanged(android.text.Editable x) { boolean has = x.toString().trim().length() > 0; if (has != send.isEnabled()) { send.setEnabled(has); send.animate().alpha(has ? 1f : 0.35f).setDuration(motion() ? 150 : 0).start(); } } });
        k.view = r; k.input = e; return k;
    }
    LinearLayout switchRow(String title, String sub, boolean on, View.OnClickListener l) {
        LinearLayout r = row(); r.setPadding(t.dp(16), t.dp(12), t.dp(12), t.dp(12)); r.setMinimumHeight(t.dp(56));
        r.setBackground(t.pressable(new ColorDrawable(Color.TRANSPARENT), 0));
        LinearLayout tx = col(); tx.addView(text(title, UiTheme.Text.BODY_L, t.ink)); if (sub != null) tx.addView(text(sub, UiTheme.Text.CAPTION, t.ink2), lpGap(2));
        r.addView(tx, weight());
        Switch s = new Switch(c); s.setChecked(on); s.setClickable(false); s.setFocusable(false); s.setImportantForAccessibility(View.IMPORTANT_FOR_ACCESSIBILITY_NO);
        int[][] st = {{android.R.attr.state_checked}, {}};
        s.setThumbTintList(new ColorStateList(st, new int[]{t.dark ? 0xFF0D1120 : 0xFFFFFFFF, t.dark ? t.ink2 : 0xFFFFFFFF}));
        s.setTrackTintList(new ColorStateList(st, new int[]{t.primary, t.lineStrong}));
        r.addView(s);
        r.setClickable(true); r.setFocusable(true); r.setOnClickListener(l);
        r.setContentDescription(title + (sub == null ? "" : ". " + sub) + (on ? ". On" : ". Off")); r.setAccessibilityDelegate(null);
        return r;
    }
    LinearLayout navRow(String icon, String title, String trailing, View.OnClickListener l) {
        LinearLayout r = row(); r.setPadding(t.dp(12), 0, t.dp(12), 0); r.setMinimumHeight(t.dp(52));
        r.setBackground(t.pressable(new ColorDrawable(Color.TRANSPARENT), 12)); r.setClickable(true); r.setFocusable(true); r.setOnClickListener(l);
        if (icon != null) { r.addView(iconView(icon, t.ink, 22)); r.addView(space(0), new LinearLayout.LayoutParams(t.dp(14), 1)); }
        TextView tt = text(title, UiTheme.Text.BODY_L, t.ink); tt.setMaxLines(1); tt.setEllipsize(android.text.TextUtils.TruncateAt.END); r.addView(tt, weight());
        if (trailing != null) r.addView(text(trailing, UiTheme.Text.CAPTION, t.ink3));
        return r;
    }

    // ---------- bottom sheet ----------
    /** A bottom sheet: rounded top, scrim, content scrolls when tall (big font sizes). On tablets it sits centred at 560dp. */
    Dialog sheet(View content, boolean cancelable) {
        Dialog d = new Dialog(c); d.requestWindowFeature(Window.FEATURE_NO_TITLE); d.setCancelable(cancelable); d.setCanceledOnTouchOutside(cancelable);
        LinearLayout box = col(); box.setBackground(t.rounded(t.surface, 0)); box.setPadding(t.dp(20), t.dp(12), t.dp(20), t.dp(24));
        android.graphics.drawable.GradientDrawable g = new android.graphics.drawable.GradientDrawable(); g.setColor(t.surface);
        float r = t.dp(28); g.setCornerRadii(new float[]{r, r, r, r, 0, 0, 0, 0}); box.setBackground(g);
        View handle = new View(c); handle.setBackground(t.rounded(t.lineStrong, 999)); LinearLayout.LayoutParams hp = new LinearLayout.LayoutParams(t.dp(40), t.dp(4)); hp.gravity = Gravity.CENTER_HORIZONTAL; hp.bottomMargin = t.dp(12);
        box.addView(handle, hp); box.addView(content, lp(MATCH, WRAP));
        ScrollView sv = new ScrollView(c); sv.addView(box);
        d.setContentView(sv);
        Window w = d.getWindow();
        if (w != null) { w.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT)); w.setGravity(Gravity.BOTTOM);
            int width = Math.min(c.getResources().getDisplayMetrics().widthPixels, t.dp(560));
            w.setLayout(width, WindowManager.LayoutParams.WRAP_CONTENT); w.setDimAmount(t.dark ? 0.6f : 0.4f);
            w.addFlags(WindowManager.LayoutParams.FLAG_DIM_BEHIND); w.setWindowAnimations(android.R.style.Animation_InputMethod); }
        return d;
    }
}
