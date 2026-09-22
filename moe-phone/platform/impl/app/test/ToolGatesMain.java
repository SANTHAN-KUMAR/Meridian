package com.meridian.app;

import java.util.*;

/** Host-JVM gates for the agent tools' pure layer (Parse.java). Expected values are fixed by the calendar, by published constants
 *  (a degree of latitude, WMO code tables, ISO 4217) or by the refusal rule, never by the parser's own output.
 *  Would each still pass if the parser returned a constant? Answered next to each one. Run: test/run_gates.sh. */
public class ToolGatesMain {
    static int fails = 0, passes = 0;
    static void check(String name, boolean ok, String detail) { System.out.println((ok ? "PASS " : "FAIL ") + name + (ok ? "" : "  " + detail)); if (ok) passes++; else fails++; }
    static Calendar at(int y, int mo, int d, int h, int mi) { Calendar c = Calendar.getInstance(TimeZone.getTimeZone("Asia/Kolkata")); c.clear(); c.set(y, mo - 1, d, h, mi, 0); return c; }
    static String show(long ms) { Calendar c = Calendar.getInstance(TimeZone.getTimeZone("Asia/Kolkata")); c.setTimeInMillis(ms); return String.format(Locale.ROOT, "%1$tF %1$tR", c); }
    static boolean refuses(String s, Calendar now) { try { Parse.when(s, now); return false; } catch (IllegalArgumentException e) { return true; } }

    public static void main(String[] a) {
        TimeZone.setDefault(TimeZone.getTimeZone("Asia/Kolkata"));
        Calendar now = at(2026, 9, 22, 18, 30);   // a Tuesday (2026-09-22 is a Tuesday in the Gregorian calendar)
        // 1. Recovery: relative times are exact offsets. A constant could not match three different offsets.
        check("in 20 minutes = now + 1200 s", Parse.when("in 20 minutes", now).at - now.getTimeInMillis() == 1_200_000L, show(Parse.when("in 20 minutes", now).at));
        check("in an hour = now + 3600 s", Parse.when("remind me in an hour", now).at - now.getTimeInMillis() == 3_600_000L, "");
        check("in 1 hour 30 minutes = now + 5400 s", Parse.when("in 1 hour 30 minutes", now).at - now.getTimeInMillis() == 5_400_000L, show(Parse.when("in 1 hour 30 minutes", now).at));
        // 2. Recovery against the calendar: named days resolve to the dates the calendar fixes.
        check("tomorrow 9am = 2026-09-23 09:00", show(Parse.when("tomorrow 9am", now).at).equals("2026-09-23 09:00"), show(Parse.when("tomorrow 9am", now).at));
        check("friday 5:30 pm = 2026-09-25 17:30", show(Parse.when("friday 5:30 pm", now).at).equals("2026-09-25 17:30"), show(Parse.when("friday 5:30 pm", now).at));
        check("tuesday (today) means next tuesday 2026-09-29", show(Parse.when("tuesday 10am", now).at).equals("2026-09-29 10:00"), show(Parse.when("tuesday 10am", now).at));
        check("25 sep 3pm = 2026-09-25 15:00", show(Parse.when("25 sep 3pm", now).at).equals("2026-09-25 15:00"), show(Parse.when("25 sep 3pm", now).at));
        check("ISO 2026-10-01 14:00", show(Parse.when("2026-10-01 14:00", now).at).equals("2026-10-01 14:00"), show(Parse.when("2026-10-01 14:00", now).at));
        check("10 sep (past this year) rolls to 2027-09-10", show(Parse.when("10 sep 9am", now).at).equals("2027-09-10 09:00"), show(Parse.when("10 sep 9am", now).at));
        // 3. Assumptions are stated, not silent: a passed clock time rolls to tomorrow AND says so; a bare date states its default time.
        Parse.When w = Parse.when("at 7am", now);
        check("7am (passed) -> tomorrow, stated", show(w.at).equals("2026-09-23 07:00") && w.assumed.contains("tomorrow"), show(w.at) + " / " + w.assumed);
        Parse.When b = Parse.when("tomorrow", now);
        check("bare 'tomorrow' -> 09:00, stated", show(b.at).equals("2026-09-23 09:00") && b.assumed.contains("09:00"), show(b.at) + " / " + b.assumed);
        check("'tonight' -> 20:00 today, stated", show(Parse.when("tonight", now).at).equals("2026-09-22 20:00") && !Parse.when("tonight", now).assumed.isEmpty(), show(Parse.when("tonight", now).at));
        check("'8 tonight' -> 20:00", show(Parse.when("8 tonight", now).at).equals("2026-09-22 20:00"), show(Parse.when("8 tonight", now).at));
        // 4. Degeneracy: ambiguous or empty input is refused, never guessed. A parser returning a constant would accept all of these.
        check("refuses slash date 03/04", refuses("03/04 5pm", now), "");
        check("refuses empty", refuses("", now), "");
        check("refuses words without a time", refuses("whenever", now), "");
        check("refuses a past ISO date", refuses("2026-09-01 10:00", now), "");
        // 5. Ranges: a day range is exactly 24 h and starts at local midnight; a week is 7 days.
        long[] r = Parse.range("tomorrow", now);
        check("range tomorrow = [2026-09-23 00:00, +24h)", show(r[0]).equals("2026-09-23 00:00") && r[1] - r[0] == 86_400_000L, show(r[0]) + " " + (r[1] - r[0]));
        long[] wk = Parse.range("this week", now);
        check("range this week = 7 days from today", show(wk[0]).equals("2026-09-22 00:00") && wk[1] - wk[0] == 7 * 86_400_000L, show(wk[0]));
        check("range friday = 2026-09-25", show(Parse.range("friday", now)[0]).equals("2026-09-25 00:00"), show(Parse.range("friday", now)[0]));
        // 6. Physical: one degree of latitude is 111.19 km on a sphere of the IUGG mean radius (2*pi*6371.0088/360); symmetric; zero at identity.
        double deg = Parse.metres(0, 0, 1, 0);
        check("1 deg latitude = 111,195 m (+-1 m)", Math.abs(deg - 111_195.08) < 1, String.valueOf(deg));
        check("distance symmetric and zero at identity", Math.abs(Parse.metres(13.08, 80.27, 12.97, 77.59) - Parse.metres(12.97, 77.59, 13.08, 80.27)) < 1e-6 && Parse.metres(13.08, 80.27, 13.08, 80.27) == 0, "");
        //    Chennai -> Bengaluru great-circle is ~290 km (published straight-line distance); a constant distance fails one of these three.
        double cb = Parse.metres(13.0827, 80.2707, 12.9716, 77.5946);
        check("Chennai-Bengaluru within 280-300 km", cb > 280_000 && cb < 300_000, String.valueOf(cb));
        // 7. External tables: WMO 4677 codes as published by Open-Meteo; ISO 4217 codes.
        check("WMO 0 clear, 3 overcast, 95 thunderstorm, 61 light rain", Parse.wmo(0).equals("clear sky") && Parse.wmo(3).equals("overcast") && Parse.wmo(95).equals("thunderstorm") && Parse.wmo(61).equals("light rain"), "");
        check("WMO unknown code is named, not invented", Parse.wmo(42).equals("weather code 42"), Parse.wmo(42));
        check("currency: rupees INR, $ USD, euro EUR, gbp GBP, yen JPY", Parse.currency("rupees").equals("INR") && Parse.currency("$").equals("USD") && Parse.currency("Euro").equals("EUR") && Parse.currency("gbp").equals("GBP") && Parse.currency("yen").equals("JPY"), "");
        boolean refusedCur; try { Parse.currency("shells"); refusedCur = false; } catch (IllegalArgumentException e) { refusedCur = true; }
        check("currency refuses unknown words", refusedCur, "");
        // 8. Safety rule: labels that send/pay/delete are gated every time; ordinary navigation is not. Both directions are checked,
        //    so neither a constant true nor a constant false passes.
        check("sensitive: Send, Pay now, Delete chat, Place order", Parse.sensitive("Send") && Parse.sensitive("Pay now") && Parse.sensitive("Delete chat") && Parse.sensitive("Place order"), "");
        check("not sensitive: Chats, Search, Settings, Mom", !Parse.sensitive("Chats") && !Parse.sensitive("Search") && !Parse.sensitive("Settings") && !Parse.sensitive("Mom"), "");
        // 9. Phone matching on the subscriber number: +91 and trunk-0 forms of one number match; different numbers do not.
        check("digitsKey: +91 98400 12345 == 098400-12345", Parse.digitsKey("+91 98400 12345").equals(Parse.digitsKey("098400-12345")), Parse.digitsKey("+91 98400 12345"));
        check("digitsKey distinguishes different numbers", !Parse.digitsKey("+91 98400 12345").equals(Parse.digitsKey("+91 98400 12346")), "");
        check("looksLikeNumber: digits yes, names no", Parse.looksLikeNumber("+91 98400 12345") && !Parse.looksLikeNumber("Mom") && !Parse.looksLikeNumber("12"), "");
        // 10. RSS parsing on a fixed document: two items, entity and CDATA handled.
        String rss = "<rss><channel><item><title>A &amp; B</title><link>https://x.test/1</link><pubDate>Tue, 22 Sep 2026</pubDate><source url=\"u\">Src</source></item>"
            + "<item><title><![CDATA[Second <b>one</b>]]></title><link>https://x.test/2</link></item></channel></rss>";
        List<String[]> it = Parse.rssItems(rss, 10);
        check("rss: 2 items, entities and CDATA decoded", it.size() == 2 && it.get(0)[0].equals("A & B") && it.get(0)[1].equals("Src") && it.get(1)[0].equals("Second <b>one</b>") && it.get(1)[3].equals("https://x.test/2"), it.size() + "");
        check("osm tags: restaurant, pharmacy; unknown -> null (name search)", Arrays.asList(Parse.osmTags("top rated restaurants")).contains("amenity=restaurant") && Parse.osmTags("pharmacy")[0].equals("amenity=pharmacy") && Parse.osmTags("Saravana Bhavan") == null, "");

        System.out.println(passes + " passed, " + fails + " failed");
        System.exit(fails == 0 ? 0 : 1);
    }
}
