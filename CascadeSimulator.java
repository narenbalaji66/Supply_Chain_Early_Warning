package supplychain;

import supplychain.MultilayerGraph.Edge;
import supplychain.MultilayerGraph.Node;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * CONCEPT 3 - Temporal cascade simulation and early warning (pipeline step 2).
 * Monte-Carlo propagation of stress through the MTG -> alert level, lead time, nodes at risk.
 */
public final class CascadeSimulator {

    public static final class Result {
        public double[] failProb;          // P(node fails within horizon)
        public double[] meanFirstFail;     // mean step of first failure (NaN if never)
        public double expectedLoss;        // importance-weighted fraction of time failed
        public double pNetworkFailure;     // P(failed importance share >= threshold at some step)
        public double medianLeadTime = Double.NaN; // steps until network-wide failure (given it happens)
    }

    final int horizon, trials; final long seed;
    final double decay = 0.7;              // load memory
    final double networkFailShare = 0.25;  // "network-wide failure" = >=25% of importance failed

    public CascadeSimulator(int horizon, int trials, long seed) {
        this.horizon = horizon; this.trials = trials; this.seed = seed;
    }

    public int horizon() { return horizon; }

    /** Common random numbers: noise depends only on (trial, edge id / node idx) so scenarios are comparable. */
    private static double hashUnit(long a, long b, long c) {
        long z = a * 0x9E3779B97F4A7C15L + b * 0xBF58476D1CE4E5B9L + c * 0x94D049BB133111EBL;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z ^= (z >>> 31);
        return (z >>> 11) * 0x1.0p-53;
    }

    public Result run(MultilayerGraph g, Map<Integer, Double> seeds, int now) {
        int N = g.nodes.size(), H = horizon;
        double total = g.totalImportance();
        Result r = new Result();
        r.failProb = new double[N];
        r.meanFirstFail = new double[N];
        int[] firstFailCount = new int[N];
        double[] firstFailSum = new double[N];
        double lossSum = 0;
        int netFailures = 0;
        List<Integer> leads = new ArrayList<>();

        for (int trial = 0; trial < trials; trial++) {
            double[] load = new double[N];
            boolean[] failed = new boolean[N];
            int[] firstFail = new int[N];
            Arrays.fill(firstFail, -1);
            int[] failSteps = new int[N];
            double[][] out = new double[N][H + 1];
            int netStep = -1;

            for (int k = 0; k <= H; k++) {
                int t = now + k;
                double[] incoming = new double[N];
                for (Edge e : g.edges) {
                    if (!e.activeAt(t)) continue;
                    int ks = k - e.lag;
                    if (ks < 0) continue;
                    double noise = 0.75 + 0.5 * hashUnit(seed, trial, e.id);
                    incoming[e.dst] += e.weight * noise * out[e.src][ks];
                }
                double failedImportance = 0;
                for (Node n : g.nodes) {
                    int i = n.idx;
                    double inj = 0;
                    if (k == 0 && seeds.containsKey(i)) inj = seeds.get(i) * (0.8 + 0.4 * hashUnit(seed, trial, 100000L + i));
                    load[i] = decay * load[i] + incoming[i] + inj;
                    if (!failed[i] && load[i] >= n.threshold) { failed[i] = true; if (firstFail[i] < 0) firstFail[i] = k; }
                    else if (failed[i] && load[i] < 0.5 * n.threshold) failed[i] = false;   // hysteresis recovery
                    double ratio = load[i] / n.threshold;
                    out[i][k] = failed[i] ? 1.0 : Math.min(1.0, 0.5 * ratio * ratio);      // graded stress output
                    if (failed[i]) { failSteps[i]++; failedImportance += n.importance; }
                }
                if (netStep < 0 && failedImportance / total >= networkFailShare) netStep = k;
            }

            double loss = 0;
            for (Node n : g.nodes) {
                loss += n.importance * failSteps[n.idx] / (double) (H + 1);
                if (firstFail[n.idx] >= 0) { r.failProb[n.idx]++; firstFailSum[n.idx] += firstFail[n.idx]; firstFailCount[n.idx]++; }
            }
            lossSum += loss / total;
            if (netStep >= 0) { netFailures++; leads.add(netStep); }
        }
        for (int i = 0; i < N; i++) {
            r.meanFirstFail[i] = firstFailCount[i] == 0 ? Double.NaN : firstFailSum[i] / firstFailCount[i];
            r.failProb[i] /= trials;
        }
        r.expectedLoss = lossSum / trials;
        r.pNetworkFailure = netFailures / (double) trials;
        if (!leads.isEmpty()) { Collections.sort(leads); r.medianLeadTime = leads.get(leads.size() / 2); }
        return r;
    }

    // --------------------------------------------------------- EARLY WARNING

    public static String warningLevel(Result r, MultilayerGraph g, Map<Integer, Double> seeds) {
        int spread = 0;
        for (Node n : g.nodes) if (!seeds.containsKey(n.idx) && r.failProb[n.idx] >= 0.3) spread++;
        if (r.pNetworkFailure >= 0.5) return "RED";
        if (r.pNetworkFailure >= 0.2) return "AMBER";
        if (r.pNetworkFailure >= 0.05 || spread >= 2) return "WATCH";
        return "GREEN";
    }

    public static void printWarning(String title, MultilayerGraph g, Result r, Map<Integer, Double> seeds, int now) {
        System.out.println("\n=== " + title + " ===");
        System.out.printf("Alert level          : %s%n", warningLevel(r, g, seeds));
        System.out.printf("P(network-wide fail) : %.0f%%%n", 100 * r.pNetworkFailure);
        if (!Double.isNaN(r.medianLeadTime))
            System.out.printf("Lead time            : ~%d steps (t=%d) before cascade crosses network threshold%n",
                    (int) r.medianLeadTime, now + (int) r.medianLeadTime);
        System.out.printf("Expected loss index  : %.3f%n", r.expectedLoss);
        System.out.println("Nodes at risk (P(fail) >= 20%):");
        g.nodes.stream()
                .filter(n -> r.failProb[n.idx] >= 0.2)
                .sorted((a, b) -> Double.compare(r.failProb[b.idx], r.failProb[a.idx]))
                .forEach(n -> System.out.printf("   %-22s P=%3.0f%%  first failure ~ step %.1f %s%n",
                        n.key(), 100 * r.failProb[n.idx], r.meanFirstFail[n.idx],
                        seeds.containsKey(n.idx) ? "(origin)" : ""));
    }
}
