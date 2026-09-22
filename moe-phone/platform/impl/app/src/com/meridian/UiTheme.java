package com.meridian.app;

import android.content.Context;
import android.content.res.ColorStateList;
import android.content.res.Configuration;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.util.TypedValue;
import android.widget.TextView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/** Meridian design tokens (brand kit v0.1): porcelain + midnight ink, one Marigold accent for the sun, "working" and "needs you".
 *  No gradients, glows or tinted "AI" purples: every surface is a flat fill with a 1dp line. Light and dark follow the system. */
final class UiTheme {
    final boolean dark; final float density;
    // neutrals
    final int bg, surface, surface2, line, lineStrong, ink, ink2, ink3;
    // action: ink-filled in light, marigold-filled in dark (text on it = onPrimary)
    final int primary, onPrimary;
    // the sun: logo, working pulse, things waiting for the user
    final int sun, sunTint, sunInk;
    // meaning (always paired with an icon or shape, never colour alone)
    final int good, goodTint, calib, calibTint, est, estTint, warm, warmTint, danger, dangerTint, scrim;
    Typeface body, bodyMedium, bodySemi, bodyBold, display, mono;

    UiTheme(Context c) {
        dark = (c.getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES;
        density = c.getResources().getDisplayMetrics().density;
        if (!dark) {
            bg = 0xFFF5F6F9; surface = 0xFFFFFFFF; surface2 = 0xFFECEEF4; line = 0xFFDDE1EA; lineStrong = 0xFFC3C9D8;
            ink = 0xFF141A2E; ink2 = 0xFF4A5270; ink3 = 0xFF646C88;
            primary = 0xFF141A2E; onPrimary = 0xFFFFFFFF;
            sun = 0xFFF5A524; sunTint = 0xFFFDEBC8; sunInk = 0xFF6B4200;
            good = 0xFF1F7A55; goodTint = 0xFFDCF1E7; calib = 0xFF0F6E80; calibTint = 0xFFDDF0F3; est = 0xFF646C88; estTint = 0xFFE8EAF1;
            warm = 0xFF9A5B00; warmTint = 0xFFFCEBD0; danger = 0xFFB42318; dangerTint = 0xFFFBE0DD; scrim = 0x66141A2E;
        } else {
            bg = 0xFF0D1120; surface = 0xFF161B2E; surface2 = 0xFF1F2540; line = 0xFF2C3352; lineStrong = 0xFF3B4368;
            ink = 0xFFEEF0F7; ink2 = 0xFFA9B0C8; ink3 = 0xFF8088A6;
            primary = 0xFFF5A524; onPrimary = 0xFF0D1120;
            sun = 0xFFF5A524; sunTint = 0xFF3A2C12; sunInk = 0xFFFFC76B;
            good = 0xFF3DBE8B; goodTint = 0xFF123326; calib = 0xFF4FC3D6; calibTint = 0xFF0F2D36; est = 0xFFA9B0C8; estTint = 0xFF1F2540;
            warm = 0xFFFFB84D; warmTint = 0xFF3A2C12; danger = 0xFFFF8A80; dangerTint = 0xFF3B1512; scrim = 0x99000000;
        }
        loadFonts(c);
    }

    int dp(float v) { return Math.round(v * density); }

    // ---------- type ----------
    /** Variable fonts shipped under res/font (OFL, licences in res/raw). Instances are cut with font-variation settings;
     *  any failure falls back to the platform sans so the UI never blocks on a font. */
    private void loadFonts(Context c) {
        File fig = extract(c, "figtree"), bri = extract(c, "bricolage");
        body = var(fig, "'wght' 400", Typeface.SANS_SERIF);
        bodyMedium = var(fig, "'wght' 500", Typeface.create("sans-serif-medium", Typeface.NORMAL));
        bodySemi = var(fig, "'wght' 600", Typeface.create("sans-serif-medium", Typeface.NORMAL));
        bodyBold = var(fig, "'wght' 700", Typeface.DEFAULT_BOLD);
        display = var(bri, "'wght' 620, 'opsz' 40, 'wdth' 100", bodySemi);
        mono = Typeface.MONOSPACE;
    }
    private static File extract(Context c, String name) {
        try {
            File f = new File(c.getCacheDir(), "font_" + name + ".ttf");
            int id = c.getResources().getIdentifier(name, "font", c.getPackageName());
            if (id == 0) return null;
            if (f.length() > 0) return f;
            try (InputStream in = c.getResources().openRawResource(id); FileOutputStream o = new FileOutputStream(f)) {
                byte[] b = new byte[1 << 16]; int n; while ((n = in.read(b)) > 0) o.write(b, 0, n);
            }
            return f;
        } catch (Exception e) { return null; }
    }
    private static Typeface var(File f, String settings, Typeface fallback) {
        if (f == null) return fallback;
        try { Typeface t = new Typeface.Builder(f).setFontVariationSettings(settings).build(); return t != null ? t : fallback; }
        catch (Exception e) { return fallback; }
    }

    /** Named text styles (sp, so they follow the user's font size up to 200%). */
    enum Text { DISPLAY, HEADLINE, TITLE_L, TITLE, BODY_L, BODY, LABEL, CAPTION, OVERLINE, MONO }
    void style(TextView t, Text s) {
        float size; Typeface tf; float lh;
        switch (s) {
            case DISPLAY: size = 34; tf = display; lh = 1.12f; t.setLetterSpacing(-0.02f); break;
            case HEADLINE: size = 26; tf = display; lh = 1.2f; t.setLetterSpacing(-0.01f); break;
            case TITLE_L: size = 20; tf = bodySemi; lh = 1.3f; break;
            case TITLE: size = 17; tf = bodySemi; lh = 1.4f; break;
            case BODY_L: size = 17; tf = body; lh = 1.5f; break;
            case BODY: size = 15; tf = body; lh = 1.45f; break;
            case LABEL: size = 15; tf = bodySemi; lh = 1.3f; break;
            case CAPTION: size = 13; tf = bodyMedium; lh = 1.35f; break;
            case OVERLINE: size = 13; tf = bodyBold; lh = 1.3f; break;
            default: size = 13; tf = mono; lh = 1.4f; break;
        }
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, size); t.setTypeface(tf); t.setLineSpacing(0, lh);
        t.setIncludeFontPadding(false);
        if (s == Text.CAPTION) t.setFontFeatureSettings("tnum");
    }

    // ---------- surfaces ----------
    GradientDrawable rounded(int fill, float radiusDp) { return rounded(fill, radiusDp, 0, 0); }
    GradientDrawable rounded(int fill, float radiusDp, float strokeDp, int strokeColor) {
        GradientDrawable g = new GradientDrawable(); g.setColor(fill); g.setCornerRadius(dp(radiusDp));
        if (strokeDp > 0) g.setStroke(Math.max(1, dp(strokeDp)), strokeColor);
        return g;
    }
    GradientDrawable dashed(int fill, float radiusDp, int strokeColor) {
        GradientDrawable g = rounded(fill, radiusDp); g.setStroke(dp(1), strokeColor, dp(5), dp(4)); return g;
    }
    /** A bounded press ripple over `content` (flat: the ripple is the only feedback, no elevation change). */
    Drawable pressable(Drawable content, float radiusDp) {
        GradientDrawable mask = rounded(0xFFFFFFFF, radiusDp);
        return new RippleDrawable(ColorStateList.valueOf(dark ? 0x33FFFFFF : 0x1F141A2E), content, mask);
    }
    Drawable icon(Context c, String name, int color) {
        int id = c.getResources().getIdentifier("ic_" + name, "drawable", c.getPackageName());
        if (id == 0) return null;
        Drawable d = c.getDrawable(id).mutate(); d.setTint(color); return d;
    }
    Drawable mark(Context c) {
        int id = c.getResources().getIdentifier(dark ? "meridian_mark_dark" : "meridian_mark", "drawable", c.getPackageName());
        return id == 0 ? null : c.getDrawable(id);
    }
}
