package supplychain;

import supplychain.CascadeSimulator.Result;
import supplychain.MitigationPlanner.Action;
import supplychain.MitigationPlanner.Ranked;
import supplychain.MultilayerGraph.Layer;
import supplychain.MultilayerGraph.Node;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

/**
 * Supply-chain early-warning & mitigation decision-support system - entry point.
 *
 * Pipeline:
 *  1. AnomalyDetector    : EWMA z-score on live metrics               -> disruption seeds
 *  2. CascadeSimulator   : Monte-Carlo cascade over the MTG            -> early warning (level, lead time, risky nodes)
 *  3. MitigationPlanner  : generate + evaluate actions by re-simulation -> ranked list (loss reduction per cost)
 *  4. MitigationPlanner  : greedy budgeted plan (marginal gains)        -> recommended action bundle
 *
 * Requires Java 17+ (no external libraries). See README.md for build/run instructions.
 */
public final class SupplyChainEarlyWarning {

    private SupplyChainEarlyWarning() { }

    // --------------------------------------------------------------- DEMO DATA

    static MultilayerGraph buildDemoNetwork() {
        MultilayerGraph g = new MultilayerGraph();
        Object[][] entities = {   // id, importance, physical buffer threshold
                {"SUP_A", 1.0, 1.0}, {"SUP_B", 0.8, 1.0}, {"PORT", 1.5, 1.2},
                {"PLANT_1", 2.0, 1.0}, {"PLANT_2", 1.5, 1.3},
                {"DC_N", 1.5, 1.5}, {"DC_S", 1.5, 1.5}, {"RETAIL_1", 1.0, 2.0}, {"RETAIL_2", 1.0, 2.0}};
        for (Object[] e : entities) {
            String id = (String) e[0]; double imp = (Double) e[1], th = (Double) e[2];
            g.addNode(id, Layer.PHYSICAL, imp, th);
            g.addNode(id, Layer.INFORMATION, imp * 0.3, th * 1.5);
            g.addNode(id, Layer.FINANCIAL, imp * 0.3, th * 2.0);
        }
        // PHYSICAL layer: goods flow (downstream)
        g.link(Layer.PHYSICAL, "SUP_A", "PORT", 0.8, 1);
        g.link(Layer.PHYSICAL, "SUP_B", "PLANT_1", 0.6, 2);
        g.link(Layer.PHYSICAL, "PORT", "PLANT_1", 0.7, 2);
        g.link(Layer.PHYSICAL, "PORT", "PLANT_2", 0.6, 2);
        g.link(Layer.PHYSICAL, "PLANT_1", "DC_N", 0.8, 1);
        g.link(Layer.PHYSICAL, "PLANT_2", "DC_S", 0.8, 1);
        g.link(Layer.PHYSICAL, "PLANT_1", "DC_S", 0.3, 2, 0, 36);      // temporal: contract lane expires at t=36
        g.link(Layer.PHYSICAL, "DC_N", "RETAIL_1", 0.8, 1);
        g.link(Layer.PHYSICAL, "DC_S", "RETAIL_2", 0.8, 1);
        // INFORMATION layer: backlog / order-uncertainty signals (upstream)
        g.link(Layer.INFORMATION, "RETAIL_1", "DC_N", 0.5, 1);
        g.link(Layer.INFORMATION, "RETAIL_2", "DC_S", 0.5, 1);
        g.link(Layer.INFORMATION, "DC_N", "PLANT_1", 0.5, 1);
        g.link(Layer.INFORMATION, "DC_S", "PLANT_2", 0.5, 1);
        g.link(Layer.INFORMATION, "PLANT_1", "PORT", 0.4, 1);
        g.link(Layer.INFORMATION, "PLANT_2", "PORT", 0.4, 1);
        g.link(Layer.INFORMATION, "PLANT_1", "SUP_B", 0.5, 1);
        g.link(Layer.INFORMATION, "PORT", "SUP_A", 0.4, 1);
        // FINANCIAL layer: payment delays / credit stress (upstream)
        g.link(Layer.FINANCIAL, "RETAIL_1", "DC_N", 0.4, 2);
        g.link(Layer.FINANCIAL, "RETAIL_2", "DC_S", 0.4, 2);
        g.link(Layer.FINANCIAL, "DC_N", "PLANT_1", 0.4, 2);
        g.link(Layer.FINANCIAL, "DC_S", "PLANT_2", 0.4, 2);
        g.link(Layer.FINANCIAL, "PLANT_1", "SUP_B", 0.4, 2);
        g.link(Layer.FINANCIAL, "PLANT_1", "PORT", 0.3, 2);
        g.link(Layer.FINANCIAL, "PORT", "SUP_A", 0.3, 2);
        // INTER-LAYER couplings (same entity)
        for (Object[] e : entities) {
            String id = (String) e[0];
            g.addEdge(g.node(id, Layer.PHYSICAL), g.node(id, Layer.INFORMATION), 0.6, 1, 0, Integer.MAX_VALUE);
            g.addEdge(g.node(id, Layer.INFORMATION), g.node(id, Layer.PHYSICAL), 0.3, 1, 0, Integer.MAX_VALUE);
            g.addEdge(g.node(id, Layer.PHYSICAL), g.node(id, Layer.FINANCIAL), 0.4, 2, 0, Integer.MAX_VALUE);
            g.addEdge(g.node(id, Layer.FINANCIAL), g.node(id, Layer.PHYSICAL), 0.3, 2, 0, Integer.MAX_VALUE);
        }
        return g;
    }

    /** Synthetic live telemetry: delay ratio (actual lead time / planned) per physical node. */
    static Map<String, double[]> syntheticTelemetry(MultilayerGraph g, int now) {
        Random rnd = new Random(7);
        Map<String, double[]> m = new LinkedHashMap<>();
        for (Node n : g.nodes) {
            if (n.layer != Layer.PHYSICAL) continue;
            double[] s = new double[now + 1];
            for (int t = 0; t <= now; t++) s[t] = 1.0 + 0.1 * rnd.nextGaussian();
            m.put(n.entity, s);
        }
        double[] a = m.get("SUP_A");           // local disruption at supplier A over the last 3 steps
        a[now - 2] = 1.4; a[now - 1] = 1.9; a[now] = 2.4;
        m.get("PORT")[now] = 1.2;              // mild noise elsewhere (should NOT trigger)
        return m;
    }

    // ----------------------------------------------------------- CONSOLE INPUT

    /**
     * Reads live telemetry from the console.
     * For each physical node the user types the most recent delay ratios (actual lead time / planned lead time,
     * 1.0 = on time), oldest first. Older history that was not typed is assumed normal (1.0 +/- 0.1 noise),
     * so the detector still has a baseline to compare against. An empty line means "running normally".
     */
    static Map<String, double[]> readTelemetry(MultilayerGraph g, int now, Scanner in) {
        Random rnd = new Random(7);
        Map<String, double[]> m = new LinkedHashMap<>();
        System.out.println("\nEnter the most recent delay ratios (actual lead time / planned lead time; 1.0 = on time)");
        System.out.println("for each entity, oldest first, separated by commas or spaces, e.g.  1.4, 1.9, 2.4");
        System.out.println("Press Enter on an empty line if the entity is running normally.\n");
        for (Node n : g.nodes) {
            if (n.layer != Layer.PHYSICAL) continue;
            double[] s = new double[now + 1];
            for (int t = 0; t <= now; t++) s[t] = 1.0 + 0.1 * rnd.nextGaussian();   // assumed normal history
            double[] recent = null;
            while (recent == null) {
                System.out.printf("  %-9s recent readings: ", n.entity);
                String line = in.hasNextLine() ? in.nextLine().trim() : "";
                recent = parseReadings(line);
                if (recent == null) System.out.println("    Please enter positive numbers only, e.g. 1.2, 1.8, 2.3");
            }
            int len = Math.min(recent.length, now + 1);
            for (int i = 0; i < len; i++) s[now - len + 1 + i] = recent[recent.length - len + i];
            m.put(n.entity, s);
        }
        return m;
    }

    /** Returns the parsed numbers, an empty array for a blank line, or null if the input is invalid. */
    static double[] parseReadings(String line) {
        if (line.isEmpty()) return new double[0];
        String[] parts = line.split("[,\\s]+");
        double[] out = new double[parts.length];
        try {
            for (int i = 0; i < parts.length; i++) {
                out[i] = Double.parseDouble(parts[i]);
                if (!(out[i] > 0) || Double.isInfinite(out[i])) return null;
            }
        } catch (NumberFormatException ex) {
            return null;
        }
        return out;
    }

    static double readBudget(Scanner in, double defaultBudget) {
        System.out.printf("Mitigation budget (press Enter for default %.0f): ", defaultBudget);
        String line = in.hasNextLine() ? in.nextLine().trim() : "";
        if (line.isEmpty()) return defaultBudget;
        try {
            double b = Double.parseDouble(line);
            if (b > 0 && !Double.isInfinite(b)) return b;
        } catch (NumberFormatException ignored) { }
        System.out.println("Invalid budget, using default " + defaultBudget);
        return defaultBudget;
    }

    // -------------------------------------------------------------------- MAIN

    public static void main(String[] args) {
        int now = 30;
        MultilayerGraph graph = buildDemoNetwork();
        CascadeSimulator sim = new CascadeSimulator(14, 400, 42L);
        Scanner in = new Scanner(System.in);

        System.out.println("Layers: " + Arrays.toString(Layer.values()) + " | nodes=" + graph.nodes.size()
                + " edges=" + graph.edges.size() + " | now=t" + now + ", horizon=" + sim.horizon());

        System.out.println("\n[1] Local anomaly detection");
        System.out.print("Use [d]emo data or [e]nter your own readings? (d/e, default d): ");
        String choice = in.hasNextLine() ? in.nextLine().trim().toLowerCase() : "d";
        Map<String, double[]> telemetry = choice.startsWith("e")
                ? readTelemetry(graph, now, in)
                : syntheticTelemetry(graph, now);
        double budget = readBudget(in, 40);

        System.out.println();
        Map<Integer, Double> seeds = AnomalyDetector.detectDisruptions(graph, telemetry, 3.0);
        if (seeds.isEmpty()) { System.out.println("No disruption detected."); return; }

        System.out.println("\n[2] Early warning (temporal multilayer cascade simulation)");
        Result base = sim.run(graph, seeds, now);
        CascadeSimulator.printWarning("EARLY WARNING", graph, base, seeds, now);

        System.out.println("\n[3] Ranked mitigation actions (each evaluated alone, ranked by loss reduction per cost)");
        List<Action> cands = MitigationPlanner.generateCandidates(graph, base, seeds);
        List<Ranked> ranked = MitigationPlanner.rankActions(graph, seeds, now, sim, base, cands);
        int rank = 1;
        for (Ranked r : ranked)
            System.out.printf("%2d. %-18s cost=%4.1f  lossReduction=%.3f  netFailDrop=%.0f pts  ROI=%.4f%n    %s%n",
                    rank++, r.action.id, r.action.cost, r.lossReduction, 100 * r.pNetDrop, r.roi, r.action.description);

        System.out.println("\n[4] Recommended plan within budget = " + budget);
        List<Ranked> plan = new ArrayList<>();
        Result after = MitigationPlanner.greedyPlan(graph, seeds, now, sim, cands, budget, plan);
        double spent = 0;
        for (Ranked r : plan) {
            spent += r.action.cost;
            System.out.printf(" -> %-18s cost=%4.1f  marginal gain=%.3f   %s%n", r.action.id, r.action.cost, r.lossReduction, r.action.description);
        }
        System.out.printf("Spent %.1f of %.1f | P(network fail): %.0f%% -> %.0f%% | expected loss: %.3f -> %.3f%n",
                spent, budget, 100 * base.pNetworkFailure, 100 * after.pNetworkFailure, base.expectedLoss, after.expectedLoss);
    }
}
