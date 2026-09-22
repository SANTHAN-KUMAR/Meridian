package com.meridian.app;

import java.util.Locale;

/** Brand kit "Voice: errors become next steps". Every raw failure the platform produces (typed refusals from 03_INTERFACES.md
 *  section 8, agent prefixes, engine and I/O exceptions) becomes a plain sentence plus one action. The raw text is never
 *  discarded: it is kept as `detail` and shown only behind "Details". Unknown failures get an honest generic message. */
final class UiHumanize {
    /** What the problem card offers. The activity maps these to real handlers; NONE shows no button. */
    enum Action { NONE, RETRY, SETUP, OPEN_MODELS, MEASURE, CHOOSE_SMALLER, DOWNLOAD_AGAIN, FREE_SPACE, OPEN_SETTINGS, CONTINUE }
    static final class Problem {
        final String title, body, actionLabel, detail; final Action action; final boolean warm;
        Problem(String title, String body, Action action, String actionLabel, String detail, boolean warm) {
            this.title = title; this.body = body; this.action = action; this.actionLabel = actionLabel; this.detail = detail; this.warm = warm; }
    }
    static Problem p(String t, String b, Action a, String l, String d) { return new Problem(t, b, a, l, d, false); }

    static Problem of(String raw) {
        String s = raw == null ? "" : raw.trim(), l = s.toLowerCase(Locale.ROOT);
        if (l.contains("memoryunavailable") || l.contains("outofmemory") || l.contains("lease") && l.contains("refus"))
            return p("Your phone needs a bit more room.", "Other apps are using memory right now. Close a few and try again, or pick a smaller model.", Action.RETRY, "Try again", s);
        if (l.contains("no model") && !l.contains("infeasible") || l.contains("select a model") || l.contains("not downloaded") || l.contains("no .gguf"))
            return p("No assistant model is on this phone yet.", "Download one that fits this phone to get started. It works offline after that.", Action.OPEN_MODELS, "Choose a model", s);
        if (l.contains("thermallimited") || l.contains("too warm") || l.contains("thermal"))
            return new Problem("Your phone is warm, so I've slowed down.", "This protects your phone and battery. Speed comes back once it cools.", Action.CONTINUE, "Keep going", s, true);
        if (l.contains("notcalibrated") || l.contains("not calibrated"))
            return p("I haven't measured this yet.", "A short test on this phone tells me how fast it will run.", Action.MEASURE, "Measure now", s);
        if (l.contains("infeasible") || l.contains("outofrange") || l.contains("does not fit"))
            return p("That model is too big for this phone.", "Pick one from the list that is marked as fitting this phone.", Action.OPEN_MODELS, "See models that fit", s);
        if (l.contains("storagetooslow"))
            return p("The model is stored somewhere too slow.", "Move it to the phone's internal storage and it will run properly.", Action.OPEN_MODELS, "Manage models", s);
        if (l.contains("nospace") || l.contains("no space") || l.contains("enospc"))
            return p("There isn't enough free space.", "Free up some storage on the phone, then try again.", Action.FREE_SPACE, "Open storage settings", s);
        if (l.contains("integrityfailed") || l.contains("sha256") || l.contains("not a valid gguf") || l.contains("checksum"))
            return p("That download looks damaged.", "I've set it aside so it can't cause problems.", Action.DOWNLOAD_AGAIN, "Download again", s);
        if (l.contains("architectureunsupported"))
            return p("I can't run this kind of model yet.", "Choose another model from the list. The ones shown there all work here.", Action.OPEN_MODELS, "See models", s);
        if (l.contains("engineunsupported"))
            return p("This phone's processor isn't supported.", "The assistant needs a newer processor than this phone has.", Action.NONE, null, s);
        if (l.contains("profile this phone") || l.contains("profile the device"))
            return p("Let's get to know your phone first.", "A quick check measures what this phone can run. Keep the app open while it runs.", Action.SETUP, "Check my phone", s);
        if (l.startsWith("noprogress") || l.contains("no progress") || l.contains("step budget"))
            return p("I got stuck on this one.", "Here's how far I got. You can try again with more detail, or do the last part yourself.", Action.RETRY, "Try again", s);
        if (l.startsWith("consentrequired") || l.contains("declined"))
            return p("Stopped, as you asked.", "Nothing was sent or changed without your OK.", Action.NONE, null, s);
        if (l.contains("rate") && l.contains("limit") || l.contains("messages/calls per"))
            return p("I've hit the safety limit for messages.", "To protect you, I send only a few messages or calls per task. Start a new task to continue.", Action.NONE, null, s);
        if (l.contains("timed out") || l.contains("did not become ready"))
            return p("That took too long, so I stopped.", "The phone may be busy. Close other apps and try again.", Action.RETRY, "Try again", s);
        if (l.contains("exceeds the session n_ctx") || l.contains("context full"))
            return p("That was too much to hold at once.", "Try splitting it into smaller tasks.", Action.RETRY, "Try again", s);
        if (l.contains("unknownhost") || l.contains("unable to resolve") || l.contains("network") || l.contains("connect"))
            return p("I couldn't reach the internet.", "This part needs a connection. Everything else still works offline.", Action.RETRY, "Try again", s);
        if (l.contains("engine") && (l.contains("fail") || l.contains("exit") || l.contains("crash")))
            return p("I had to restart.", "Nothing you did caused this. Your earlier tasks are safe.", Action.RETRY, "Try again", s);
        return p("Something went wrong on my side.", "Try again. If it keeps happening, the details below help with support.", Action.RETRY, "Try again", s);
    }

    /** A model id or filename as a person would say it: "Qwen3-30B-A3B-Q4_0.gguf" -> "Qwen3 30B A3B". */
    static String modelName(String file) {
        if (file == null) return "";
        String n = file.replaceAll("(?i)\\.gguf$", "").replaceAll("(?i)[-_.](q\\d[\\w]*|iq\\d[\\w]*|f16|bf16|f32|q8_0|ud)(?=$|[-_.])", "");
        n = n.replace('_', ' ').replace('-', ' ').replaceAll("\\s+", " ").trim();
        StringBuilder b = new StringBuilder();
        for (String w : n.split(" ")) {
            if (w.isEmpty() || w.matches("\\d{4}")) continue;   // release dates such as 0924 mean nothing to a person
            String lw = w.toLowerCase(Locale.ROOT), k = KNOWN.get(lw);
            String out = k != null ? k : lw.matches("\\d+(\\.\\d+)?[bm]") ? lw.toUpperCase(Locale.ROOT) : lw.matches("a\\d+(\\.\\d+)?[bm]") ? lw.toUpperCase(Locale.ROOT)
                : w.equals(lw) ? Character.toUpperCase(w.charAt(0)) + w.substring(1) : w;
            if (b.length() > 0) b.append(' '); b.append(out);
        }
        return b.length() == 0 ? file : b.toString();
    }
    static final java.util.Map<String, String> KNOWN = new java.util.HashMap<>();
    static { String[] k = {"olmoe", "OLMoE", "qwen", "Qwen", "qwen2", "Qwen2", "qwen3", "Qwen3", "gpt", "GPT", "oss", "OSS", "phi", "Phi", "llama", "Llama", "gemma", "Gemma", "mistral", "Mistral", "granite", "Granite", "deepseek", "DeepSeek", "smollm", "SmolLM", "instruct", "Instruct", "it", "IT", "moe", "MoE"};
        for (int i = 0; i < k.length; i += 2) KNOWN.put(k[i], k[i + 1]); }
    /** A speed as words first; the number stays secondary (brand kit "Speed, in words first"). */
    static String speedWords(double tokS) {
        if (Double.isNaN(tokS) || tokS <= 0) return "Speed not known yet";
        if (tokS >= 20) return "Instant replies";
        if (tokS >= 8) return "Quick";
        if (tokS >= 3) return "Takes its time";
        return "Slow on this phone";
    }
    static int speedLevel(double tokS) { return tokS >= 20 ? 3 : tokS >= 8 ? 2 : tokS > 0 ? 1 : 0; }
    static String gb(long bytes) { return bytes >= 1000L * 1000 * 1000 ? String.format(Locale.ROOT, "%.1f GB", bytes / 1e9) : String.format(Locale.ROOT, "%d MB", Math.max(1, bytes / 1000000)); }
    static String greeting() {
        int h = java.util.Calendar.getInstance().get(java.util.Calendar.HOUR_OF_DAY);
        return h < 5 ? "Good evening" : h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";
    }
}
