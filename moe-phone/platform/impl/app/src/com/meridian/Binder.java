package com.meridian.app;

import org.json.*;
import java.util.*;

/** bind(CapabilityRequest) -> ModelBinding | Refusal (03_INTERFACES.md section 7). Expectations are quoted only from this
 *  device's recorded in-regime turns of this model; with fewer than 3 such observations the answer is NotCalibrated. */
public final class Binder {
    public static JSONObject bind(JSONObject profile, Planner.Card card, long context, List<Double> observedTokS) throws JSONException {
        JSONObject out = new JSONObject().put("model_id", card.modelId).put("context", context);
        try { out.put("feasibility", Planner.plan(profile, card, context)); }
        catch (Planner.Refusal r) { return out.put("refusal", new JSONObject().put("reason", r.reason).put("detail", r.getMessage()).put("missing", new JSONArray(r.missing))); }
        if (observedTokS.size() >= 3) {
            List<Double> v = new ArrayList<>(observedTokS); Collections.sort(v);
            out.put("expectation", new JSONObject().put("decode_tok_s", v.get(v.size() / 2)).put("interval", new JSONArray().put(v.get(0)).put(v.get(v.size() - 1)))
                .put("provenance", "measured").put("basis", v.size() + " recorded in-regime turns of this model on this device"));
        } else out.put("expectation", new JSONObject().put("provenance", "unknown").put("reason", "NotCalibrated: only " + observedTokS.size() + " recorded in-regime turns; run a few chat turns with the screen on, unplugged"));
        return out;
    }
}
