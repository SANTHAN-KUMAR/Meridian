package com.meridian.app;

import android.app.*;
import android.content.*;
import android.os.Build;
import org.json.*;
import java.io.*;
import java.util.*;

/** Reminders that fire even when Meridian is closed: stored in files/reminders.json, scheduled with AlarmManager, delivered as a
 *  high-priority notification by this receiver, and re-scheduled after a reboot (alarms do not survive one). */
public final class ReminderReceiver extends BroadcastReceiver {
    static final String CHANNEL = "meridian_reminders", ACTION = "com.meridian.app.REMINDER";

    static File store(Context c) { return new File(c.getFilesDir(), "reminders.json"); }
    static synchronized JSONArray load(Context c) { try { File f = store(c); return f.exists() ? new JSONArray(Native.readFile(f.getAbsolutePath())) : new JSONArray(); } catch (Exception e) { return new JSONArray(); } }
    static synchronized void save(Context c, JSONArray a) throws IOException { try (FileWriter w = new FileWriter(store(c), false)) { w.write(a.toString()); } }
    static PendingIntent pi(Context c, int id, int flags) { return PendingIntent.getBroadcast(c, id, new Intent(c, ReminderReceiver.class).setAction(ACTION).putExtra("id", id), flags | PendingIntent.FLAG_IMMUTABLE); }

    /** Schedules one reminder; returns whether the alarm is exact (inexact alarms can fire some minutes late, and the result says so). */
    static boolean schedule(Context c, int id, long at) {
        AlarmManager am = (AlarmManager) c.getSystemService(Context.ALARM_SERVICE); PendingIntent p = pi(c, id, PendingIntent.FLAG_UPDATE_CURRENT);
        if (Build.VERSION.SDK_INT < 31 || am.canScheduleExactAlarms()) { am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, at, p); return true; }
        am.setWindow(AlarmManager.RTC_WAKEUP, at, 10 * 60 * 1000L, p); return false;
    }
    static boolean scheduled(Context c, int id) { return pi(c, id, PendingIntent.FLAG_NO_CREATE) != null; }
    static void cancel(Context c, int id) { PendingIntent p = pi(c, id, PendingIntent.FLAG_NO_CREATE); if (p != null) { ((AlarmManager) c.getSystemService(Context.ALARM_SERVICE)).cancel(p); p.cancel(); } }

    @Override public void onReceive(Context c, Intent i) {
        if (Intent.ACTION_BOOT_COMPLETED.equals(i.getAction())) {   // re-arm every future reminder
            JSONArray a = load(c); long now = System.currentTimeMillis();
            for (int k = 0; k < a.length(); k++) { JSONObject r = a.optJSONObject(k); if (r != null && r.optLong("at") > now) schedule(c, r.optInt("id"), r.optLong("at")); }
            return;
        }
        int id = i.getIntExtra("id", -1); JSONArray a = load(c), keep = new JSONArray(); String text = "Reminder";
        for (int k = 0; k < a.length(); k++) { JSONObject r = a.optJSONObject(k); if (r == null) continue; if (r.optInt("id") == id) text = r.optString("text", text); else keep.put(r); }
        try { save(c, keep); } catch (IOException ignored) { }
        NotificationManager nm = (NotificationManager) c.getSystemService(Context.NOTIFICATION_SERVICE);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Reminders", NotificationManager.IMPORTANCE_HIGH));
        PendingIntent open = PendingIntent.getActivity(c, 0, new Intent(c, MainActivity.class), PendingIntent.FLAG_IMMUTABLE);
        nm.notify(10000 + id, new Notification.Builder(c, CHANNEL).setSmallIcon(android.R.drawable.ic_popup_reminder).setContentTitle("Reminder").setContentText(text)
            .setStyle(new Notification.BigTextStyle().bigText(text)).setCategory(Notification.CATEGORY_REMINDER).setContentIntent(open).setAutoCancel(true).build());
    }
}
