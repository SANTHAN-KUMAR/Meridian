package com.meridian.app;

import java.util.*;
import java.util.regex.*;

/** Pure parsing helpers for the daily-task, communication and screen tools: no Android types, so every rule here is
 *  checked on the host JVM (test/ToolGatesMain.java). Each parser refuses input it cannot read instead of guessing, and
 *  any default it applies (a time of day for "tomorrow") is returned as a stated assumption, never applied silently. */
public final class Parse {
    private Parse() { }

    /** A resolved point in time plus the assumptions made to get there (empty when none). */
    public static final class When { public final long at; public final String assumed; When(long at, String assumed) { this.at = at; this.assumed = assumed; } }

    static final String[] DAYS = {"sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"};
    static final String[] MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};
    static final Pattern REL = Pattern.compile("\\bin\\s+(an?|half an|\\d+(?:\\.\\d+)?)\\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s|days?|weeks?)\\b(?:\\s*(?:and\\s*)?(\\d+)\\s*(minutes?|mins?|m)\\b)?");
    static final Pattern CLOCK = Pattern.compile("\\b(\\d{1,2})(?:[:.](\\d{2}))?\\s*(am|pm|a\\.m\\.|p\\.m\\.)?(?![\\d/-])");
    static final Pattern ISO = Pattern.compile("\\b(\\d{4})-(\\d{2})-(\\d{2})\\b");
    static final Pattern DAY_MONTH = Pattern.compile("\\b(\\d{1,2})(?:st|nd|rd|th)?\\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\b");
    static final Pattern MONTH_DAY = Pattern.compile("\\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\s+(\\d{1,2})(?:st|nd|rd|th)?\\b");

    /** "in 20 minutes", "in an hour", "at 6pm", "tomorrow 9am", "friday 5:30 pm", "next monday", "2026-09-25 14:00",
     *  "25 sep 3pm", "tonight", "noon". A clock time with no date that is already past today rolls to tomorrow (stated).
     *  Refuses slash dates (dd/mm vs mm/dd is ambiguous) and anything without a recognisable time or date. */
    public static When when(String s, Calendar now) {
        String q = s.toLowerCase(Locale.ROOT).trim().replaceAll("\\s+", " ");
        if (q.isEmpty()) throw new IllegalArgumentException("no time given");
        if (q.matches(".*\\b\\d{1,2}/\\d{1,2}\\b.*")) throw new IllegalArgumentException("ambiguous date '" + s + "': write it as '25 sep' or 2026-09-25");
        Matcher m = REL.matcher(q);
        if (m.find()) {
            String n = m.group(1), u = m.group(2); double v = n.equals("a") || n.equals("an") ? 1 : n.equals("half an") ? 0.5 : Double.parseDouble(n);
            double sec = u.startsWith("w") ? v * 604800 : u.startsWith("d") ? v * 86400 : u.startsWith("h") ? v * 3600 : u.startsWith("s") ? v : v * 60;
            if (m.group(3) != null) sec += Integer.parseInt(m.group(3)) * 60;
            if (sec <= 0) throw new IllegalArgumentException("not a future time: '" + s + "'");
            return new When(now.getTimeInMillis() + Math.round(sec * 1000), "");
        }
        Calendar c = (Calendar) now.clone(); c.set(Calendar.SECOND, 0); c.set(Calendar.MILLISECOND, 0);
        List<String> assumed = new ArrayList<>(); boolean dated = false;
        String rest = q;
        Matcher iso = ISO.matcher(rest), dm = DAY_MONTH.matcher(rest), md = MONTH_DAY.matcher(rest);
        if (iso.find()) { c.set(Integer.parseInt(iso.group(1)), Integer.parseInt(iso.group(2)) - 1, Integer.parseInt(iso.group(3))); dated = true; rest = rest.replace(iso.group(), " "); }
        else if (dm.find()) { setDayMonth(c, now, Integer.parseInt(dm.group(1)), month(dm.group(2))); dated = true; rest = rest.replace(dm.group(), " "); }
        else if (md.find()) { setDayMonth(c, now, Integer.parseInt(md.group(2)), month(md.group(1))); dated = true; rest = rest.replace(md.group(), " "); }
        else if (rest.matches(".*\\b(day after tomorrow)\\b.*")) { c.add(Calendar.DAY_OF_MONTH, 2); dated = true; }
        else if (rest.matches(".*\\b(tomorrow|tmrw|tmr)\\b.*")) { c.add(Calendar.DAY_OF_MONTH, 1); dated = true; }
        else if (rest.matches(".*\\b(today|tonight|this (morning|afternoon|evening))\\b.*")) dated = true;
        else for (int d = 0; d < 7; d++) if (rest.matches(".*\\b" + DAYS[d] + "\\b.*")) {
            int diff = (d + 1 - now.get(Calendar.DAY_OF_WEEK) + 7) % 7; if (diff == 0 || rest.contains("next " + DAYS[d])) diff = diff == 0 ? 7 : diff;
            c.add(Calendar.DAY_OF_MONTH, diff); dated = true; break; }
        rest = rest.replaceAll("\\b(day after tomorrow|tomorrow|tmrw|tmr|today|next|this|on|at|by|the|" + String.join("|", DAYS) + ")\\b", " ");
        int[] hm = null;
        if (rest.contains("noon") || rest.contains("midday")) hm = new int[]{12, 0};
        else if (rest.contains("midnight")) { hm = new int[]{0, 0}; if (!dated) c.add(Calendar.DAY_OF_MONTH, 1); }
        else { Matcher t = CLOCK.matcher(rest);
            while (t.find()) { int h = Integer.parseInt(t.group(1)), mi = t.group(2) == null ? 0 : Integer.parseInt(t.group(2)); String ap = t.group(3) == null ? "" : t.group(3);
                if (h > 23 || mi > 59) continue;
                if (ap.startsWith("p") && h < 12) h += 12; else if (ap.startsWith("a") && h == 12) h = 0;
                else if (ap.isEmpty() && h < 12 && q.matches(".*\\b(tonight|evening|pm)\\b.*")) h += 12;
                else if (ap.isEmpty() && h >= 1 && h <= 6 && t.group(2) == null) { h += 12; assumed.add(t.group(1) + " read as " + h + ":00 (no am/pm given)"); }
                else if (ap.isEmpty() && h >= 1 && h < 12) assumed.add(t.group() .trim() + " read as " + String.format(Locale.ROOT, "%02d:%02d", h, mi) + " (no am/pm given)");
                hm = new int[]{h, mi}; break; } }
        if (hm == null) {
            if (q.contains("tonight") || q.contains("evening")) { hm = new int[]{q.contains("tonight") ? 20 : 18, 0}; assumed.add("time " + hm[0] + ":00 for '" + (q.contains("tonight") ? "tonight" : "evening") + "'"); }
            else if (q.contains("afternoon")) { hm = new int[]{15, 0}; assumed.add("time 15:00 for 'afternoon'"); }
            else if (q.contains("morning")) { hm = new int[]{9, 0}; assumed.add("time 09:00 for 'morning'"); }
            else if (dated) { hm = new int[]{9, 0}; assumed.add("no time given; 09:00 used"); }
            else throw new IllegalArgumentException("not a time or date: '" + s + "'");
        }
        c.set(Calendar.HOUR_OF_DAY, hm[0]); c.set(Calendar.MINUTE, hm[1]);
        if (!dated && c.getTimeInMillis() <= now.getTimeInMillis()) { c.add(Calendar.DAY_OF_MONTH, 1); assumed.add("that time has passed today; tomorrow used"); }
        if (c.getTimeInMillis() <= now.getTimeInMillis()) throw new IllegalArgumentException("'" + s + "' is in the past");
        return new When(c.getTimeInMillis(), String.join("; ", assumed));
    }
    static int month(String m) { for (int i = 0; i < 12; i++) if (m.startsWith(MONTHS[i])) return i; throw new IllegalArgumentException("not a month: " + m); }
    /** A day-month with no year is the next such date on or after today. */
    static void setDayMonth(Calendar c, Calendar now, int day, int month) {
        if (day < 1 || day > 31) throw new IllegalArgumentException("not a day of the month: " + day);
        c.set(Calendar.YEAR, now.get(Calendar.YEAR)); c.set(Calendar.MONTH, month); c.set(Calendar.DAY_OF_MONTH, day);
        Calendar today = (Calendar) now.clone(); today.set(Calendar.HOUR_OF_DAY, 0); today.set(Calendar.MINUTE, 0); today.set(Calendar.SECOND, 0); today.set(Calendar.MILLISECOND, 0);
        if (c.before(today)) c.add(Calendar.YEAR, 1);
    }

    /** A day range [start, end) for calendar reads: "today", "tomorrow", "this week" (today + 6 days), "next week",
     *  a weekday, or any date `when` accepts (that whole day). Empty means today. */
    public static long[] range(String s, Calendar now) {
        String q = s.toLowerCase(Locale.ROOT).trim(); Calendar c = (Calendar) now.clone(); c.set(Calendar.HOUR_OF_DAY, 0); c.set(Calendar.MINUTE, 0); c.set(Calendar.SECOND, 0); c.set(Calendar.MILLISECOND, 0);
        int days = 1;
        if (q.isEmpty() || q.equals("today")) { }
        else if (q.contains("next week")) { c.add(Calendar.DAY_OF_MONTH, 7); days = 7; }
        else if (q.contains("week")) days = 7;
        else if (q.contains("month")) days = 30;
        else if (q.equals("tomorrow")) c.add(Calendar.DAY_OF_MONTH, 1);
        else { Calendar d = Calendar.getInstance(now.getTimeZone()); d.setTimeInMillis(when(q.matches(".*\\d.*|.*(noon|night|morning|evening).*") ? q : q + " 23:59", now).at);
            c.set(d.get(Calendar.YEAR), d.get(Calendar.MONTH), d.get(Calendar.DAY_OF_MONTH)); }
        long st = c.getTimeInMillis(); c.add(Calendar.DAY_OF_MONTH, days); return new long[]{st, c.getTimeInMillis()};
    }

    /** Great-circle distance in metres (haversine, mean Earth radius 6,371,008.8 m, IUGG). */
    public static double metres(double la1, double lo1, double la2, double lo2) {
        double r = 6371008.8, p1 = Math.toRadians(la1), p2 = Math.toRadians(la2), dp = p2 - p1, dl = Math.toRadians(lo2 - lo1);
        double h = Math.sin(dp / 2) * Math.sin(dp / 2) + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
        return 2 * r * Math.asin(Math.min(1, Math.sqrt(h)));
    }

    /** WMO weather interpretation codes (WMO 4677 as used by Open-Meteo's `weather_code`). Unknown codes are named as such. */
    public static String wmo(int code) {
        switch (code) {
            case 0: return "clear sky"; case 1: return "mainly clear"; case 2: return "partly cloudy"; case 3: return "overcast";
            case 45: case 48: return "fog"; case 51: return "light drizzle"; case 53: return "drizzle"; case 55: return "dense drizzle";
            case 56: case 57: return "freezing drizzle"; case 61: return "light rain"; case 63: return "rain"; case 65: return "heavy rain";
            case 66: case 67: return "freezing rain"; case 71: return "light snow"; case 73: return "snow"; case 75: return "heavy snow"; case 77: return "snow grains";
            case 80: return "light rain showers"; case 81: return "rain showers"; case 82: return "violent rain showers"; case 85: case 86: return "snow showers";
            case 95: return "thunderstorm"; case 96: case 99: return "thunderstorm with hail";
            default: return "weather code " + code;
        }
    }

    static final String[][] CURRENCY = {{"rupee", "INR"}, {"rs", "INR"}, {"₹", "INR"}, {"dollar", "USD"}, {"usd", "USD"}, {"$", "USD"}, {"euro", "EUR"}, {"€", "EUR"},
        {"pound", "GBP"}, {"£", "GBP"}, {"yen", "JPY"}, {"¥", "JPY"}, {"dirham", "AED"}, {"riyal", "SAR"}, {"yuan", "CNY"}, {"renminbi", "CNY"}, {"ringgit", "MYR"},
        {"baht", "THB"}, {"won", "KRW"}, {"franc", "CHF"}, {"taka", "BDT"}, {"dinar", "KWD"}};
    /** A currency as people say it ("rupees", "$", "euro") or an ISO 4217 code -> the code; refuses anything else. */
    public static String currency(String s) {
        String q = s.trim().toLowerCase(Locale.ROOT);
        if (q.matches("[a-z]{3}") && !q.equals("yen") && !q.equals("won")) return q.toUpperCase(Locale.ROOT);
        for (String[] c : CURRENCY) if (q.equals(c[0]) || q.startsWith(c[0]) && c[0].length() > 2) return c[1];
        if (q.contains("singapore")) return "SGD"; if (q.contains("australian")) return "AUD"; if (q.contains("canadian")) return "CAD";
        throw new IllegalArgumentException("not a currency: '" + s + "' (use a 3-letter code such as INR or USD)");
    }

    /** Button and link labels whose tap sends, pays, deletes or submits something: such a tap is gated every time, with the
     *  label shown, even inside an app the user already allowed for this task (07_AGENT_RUNTIME.md section 8, transact class). */
    static final Pattern SENSITIVE = Pattern.compile("(?i)\\b(send|pay|payment|buy|purchase|order|checkout|check out|place order|delete|remove|discard|confirm|submit|transfer|post|publish|tweet|share|book|reserve|subscribe|uninstall|reset|erase|sign out|log out|logout|call|accept|decline|block|report|install)\\b");
    public static boolean sensitive(String label) { return label != null && SENSITIVE.matcher(label).find(); }

    /** Phone numbers are compared on their last 8 digits (country codes and trunk prefixes vary; 8 digits is below every
     *  national significant number length in common use, so a match on them is a match of the subscriber number). */
    public static String digitsKey(String n) { String d = n == null ? "" : n.replaceAll("[^0-9]", ""); return d.length() > 8 ? d.substring(d.length() - 8) : d; }
    public static boolean looksLikeNumber(String s) { return s != null && s.replaceAll("[\\s()+.-]", "").matches("\\d{3,15}"); }

    /** RSS 2.0 <item>s -> {title, source, published, link}; entities decoded, CDATA unwrapped. Up to `max` items. */
    public static List<String[]> rssItems(String xml, int max) {
        List<String[]> out = new ArrayList<>(); Matcher it = Pattern.compile("(?s)<item>(.*?)</item>").matcher(xml);
        while (it.find() && out.size() < max) { String b = it.group(1);
            out.add(new String[]{tag(b, "title"), tag(b, "source"), tag(b, "pubDate"), tag(b, "link")}); }
        return out;
    }
    static String tag(String b, String t) {
        Matcher m = Pattern.compile("(?s)<" + t + "(?:\\s[^>]*)?>(.*?)</" + t + ">").matcher(b); if (!m.find()) return "";
        String v = m.group(1).trim(); if (v.startsWith("<![CDATA[")) v = v.substring(9, v.length() - 3);
        return v.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'").replace("&apos;", "'").trim();
    }

    /** OSM tag filter(s) for a kind of place people ask for; null when the words name no known kind (then a name search is used). */
    public static String[] osmTags(String what) {
        String q = what.toLowerCase(Locale.ROOT);
        String[][] map = {{"restaurant|food|dinner|lunch|eat|biryani|pizza|dosa|meal", "amenity=restaurant", "amenity=fast_food"}, {"cafe|coffee|tea", "amenity=cafe"},
            {"bar|pub|beer", "amenity=bar", "amenity=pub"}, {"pharmac|chemist|medical store|medicine", "amenity=pharmacy"}, {"hospital|emergency", "amenity=hospital"},
            {"clinic|doctor", "amenity=clinic", "amenity=doctors"}, {"atm|cash", "amenity=atm"}, {"bank", "amenity=bank"}, {"petrol|fuel|gas station|diesel", "amenity=fuel"},
            {"charging|ev charger", "amenity=charging_station"}, {"supermarket|grocery|groceries", "shop=supermarket", "shop=convenience"}, {"hotel|stay|lodge", "tourism=hotel", "tourism=guest_house"},
            {"parking", "amenity=parking"}, {"police", "amenity=police"}, {"gym|fitness", "leisure=fitness_centre"}, {"park", "leisure=park"}, {"school", "amenity=school"},
            {"temple|church|mosque|worship", "amenity=place_of_worship"}, {"cinema|movie|theatre|theater", "amenity=cinema"}, {"bakery|cake", "shop=bakery"},
            {"toilet|restroom|washroom", "amenity=toilets"}, {"bus stop|bus station", "highway=bus_stop"}, {"metro|train station|railway", "railway=station"},
            {"mall|shopping", "shop=mall"}, {"salon|haircut|barber", "shop=hairdresser"}, {"post office", "amenity=post_office"}};
        for (String[] r : map) if (Pattern.compile("\\b(" + r[0] + ")").matcher(q).find()) return Arrays.copyOfRange(r, 1, r.length);
        return null;
    }
}
