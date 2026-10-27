package supplychain;

import supplychain.MultilayerGraph.Layer;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * CONCEPT 2 - Local anomaly detection (pipeline step 1).
 * EWMA z-score on live metrics -> disruption seeds (node index -> injected stress severity).
 */
public final class AnomalyDetector {

    private AnomalyDetector() { }

    /** EWMA mean/variance tracker producing z-scores; anomalous points do not contaminate the baseline. */
    public static final class EwmaDetector {
        private final double alpha; private final int warmup; private final double zCut;
        private double mean, var; private int n;

        public EwmaDetector(double alpha, int warmup, double zCut) {
            this.alpha = alpha; this.warmup = warmup; this.zCut = zCut;
        }

        public double update(double x) {
            if (n == 0) { mean = x; var = 0.01; n++; return 0; }
            double z = (n < warmup) ? 0 : (x - mean) / Math.sqrt(var + 1e-6);
            if (Math.abs(z) < zCut) {
                double d = x - mean;
                mean += alpha * d;
                var = (1 - alpha) * (var + alpha * d * d);
            }
            n++;
            return z;
        }
    }

    /** Disruption seed = (node index, injected stress severity). */
    public static Map<Integer, Double> detectDisruptions(MultilayerGraph g, Map<String, double[]> delayRatio, double zCut) {
        Map<Integer, Double> seeds = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> en : delayRatio.entrySet()) {
            EwmaDetector det = new EwmaDetector(0.2, 8, zCut);
            double[] s = en.getValue();
            double maxRecentZ = 0;
            for (int t = 0; t < s.length; t++) {
                double z = det.update(s[t]);
                if (t >= s.length - 3) maxRecentZ = Math.max(maxRecentZ, z);
            }
            if (maxRecentZ >= zCut) {
                double severity = Math.min(2.0, 0.3 * maxRecentZ);
                seeds.put(g.node(en.getKey(), Layer.PHYSICAL).idx, severity);
                System.out.printf("  anomaly: %-10s z=%.1f -> injected stress %.2f%n", en.getKey(), maxRecentZ, severity);
            }
        }
        return seeds;
    }
}
