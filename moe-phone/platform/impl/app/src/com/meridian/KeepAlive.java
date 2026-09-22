package com.meridian.app;

import android.app.*;
import android.content.Intent;
import android.os.IBinder;

/** Foreground service that holds the engine process's priority while a model is loaded (02_ARCHITECTURE.md section 8:
 *  "one foreground service hosts the inference session"). It does no work; its job is to keep the app process, and
 *  therefore the child engine process, from being reclaimed when the UI goes to the background. */
public final class KeepAlive extends Service {
    static final String CHANNEL = "meridian_engine";
    @Override public int onStartCommand(Intent i, int flags, int id) {
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Meridian engine", NotificationManager.IMPORTANCE_LOW));
        PendingIntent open = PendingIntent.getActivity(this, 0, new Intent(this, MainActivity.class), PendingIntent.FLAG_IMMUTABLE);
        Notification n = new Notification.Builder(this, CHANNEL).setSmallIcon(android.R.drawable.stat_notify_sync).setContentTitle("Meridian")
            .setContentText(i != null && i.getStringExtra("text") != null ? i.getStringExtra("text") : i != null && i.getStringExtra("model") != null ? "Model loaded: " + i.getStringExtra("model") : "Model loaded").setContentIntent(open).setOngoing(true).build();
        startForeground(7, n);
        return START_NOT_STICKY;
    }
    @Override public IBinder onBind(Intent i) { return null; }
}
