package com.meridian.app;

import java.io.File;

/** Host-JVM parity check: prints ModelCard numbers for a GGUF as `key=value` lines, compared against the Python planner by test_parity.py. */
public class ParityMain {
    public static void main(String[] a) throws Exception {
        try {
            Planner.Card c = Planner.derive(new File(a[0]));
            System.out.println("total_bytes=" + c.totalBytes); System.out.println("resident_bytes=" + c.residentBytes); System.out.println("expert_bytes=" + c.expertBytes);
            System.out.println("active_bytes_per_token=" + c.activeBytesPerToken); System.out.println("kv_f16=" + c.kvF16PerToken); System.out.println("slice_sum_max=" + c.sliceSumMax);
        } catch (Planner.Refusal r) { System.out.println("refusal=" + r.reason); }
    }
}
