package supplychain;

import supplychain.CascadeSimulator.Result;
import supplychain.MultilayerGraph.Edge;
import supplychain.MultilayerGraph.Layer;
import supplychain.MultilayerGraph.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * CONCEPT 4 - Mitigation actions and budgeted planning (pipeline steps 3 and 4).
 * Generate candidate actions, evaluate each by re-simulation, then build a greedy plan within budget.
 */
public final class MitigationPlanner {

    private MitigationPlanner() { }

    public static final class Action {
        public final String id, description; public final double cost; public final Consumer<MultilayerGraph> effect;

        public Action(String id, String description, double cost, Consumer<MultilayerGraph> effect) {
            this.id = id; this.description = description; this.cost = cost; this.effect = effect;
        }
    }

    public static final class Ranked {
        public final Action action; public final double lossReduction, pNetDrop, roi;

        public Ranked(Action a, double lossReduction, double pNetDrop) {
            this.action = a; this.lossReduction = lossReduction; this.pNetDrop = pNetDrop; this.roi = lossReduction / a.cost;
        }
    }

    private static void scaleOutgoing(MultilayerGraph g, String entity, Layer layer, double factor) {
        int src = g.node(entity, layer).idx;
        for (Edge e : g.edges) if (e.src == src && g.nodes.get(e.dst).layer == layer) e.weight *= factor;
    }

    public static List<Action> generateCandidates(MultilayerGraph g, Result base, Map<Integer, Double> seeds) {
        List<Action> list = new ArrayList<>();
        for (Node n : g.nodes) {
            if (base.failProb[n.idx] < 0.15) continue;
            String e = n.entity;
            boolean origin = seeds.containsKey(n.idx);
            switch (n.layer) {
                case PHYSICAL -> {
                    if (!origin)
                        list.add(new Action("BUFFER_" + e, "Raise safety stock / expedite replenishment at " + e,
                                6 + 4 * n.importance, gr -> gr.node(e, Layer.PHYSICAL).threshold *= 1.5));
                    list.add(new Action("REROUTE_" + e,
                            origin ? "Qualify alternate supplier / lane to bypass " + e : "Re-route flows around " + e,
                            20, gr -> scaleOutgoing(gr, e, Layer.PHYSICAL, 0.4)));
                }
                case INFORMATION -> list.add(new Action("SHAPE_" + e,
                        "Dampen order / backlog signals from " + e + " (allocation, order smoothing)", 5,
                        gr -> { scaleOutgoing(gr, e, Layer.INFORMATION, 0.5); gr.node(e, Layer.INFORMATION).threshold *= 1.3; }));
                case FINANCIAL -> list.add(new Action("LIQUIDITY_" + e,
                        "Liquidity bridge / early payment for " + e, 8,
                        gr -> gr.node(e, Layer.FINANCIAL).threshold *= 1.6));
            }
        }
        return list;
    }

    /** Each action evaluated independently against the same random scenarios (common random numbers). */
    public static List<Ranked> rankActions(MultilayerGraph g, Map<Integer, Double> seeds, int now,
                                           CascadeSimulator sim, Result base, List<Action> cands) {
        List<Ranked> ranked = new ArrayList<>();
        for (Action a : cands) {
            MultilayerGraph copy = g.copy();
            a.effect.accept(copy);
            Result r = sim.run(copy, seeds, now);
            double gain = base.expectedLoss - r.expectedLoss;
            if (gain > 1e-4) ranked.add(new Ranked(a, gain, base.pNetworkFailure - r.pNetworkFailure));
        }
        ranked.sort((x, y) -> Double.compare(y.roi, x.roi));
        return ranked;
    }

    /** Budget-constrained greedy plan using marginal gain per cost (captures overlap between actions). */
    public static Result greedyPlan(MultilayerGraph g, Map<Integer, Double> seeds, int now, CascadeSimulator sim,
                                    List<Action> cands, double budget, List<Ranked> planOut) {
        MultilayerGraph cur = g.copy();
        Result curRes = sim.run(cur, seeds, now);
        List<Action> remaining = new ArrayList<>(cands);
        double spent = 0;
        while (true) {
            Ranked best = null; Result bestRes = null; MultilayerGraph bestG = null; Action bestA = null;
            for (Action a : remaining) {
                if (spent + a.cost > budget) continue;
                MultilayerGraph tmp = cur.copy();
                a.effect.accept(tmp);
                Result r = sim.run(tmp, seeds, now);
                double gain = curRes.expectedLoss - r.expectedLoss;
                if (gain < 1e-3) continue;
                Ranked cand = new Ranked(a, gain, curRes.pNetworkFailure - r.pNetworkFailure);
                if (best == null || cand.roi > best.roi) { best = cand; bestRes = r; bestG = tmp; bestA = a; }
            }
            if (best == null) break;
            planOut.add(best);
            remaining.remove(bestA);
            spent += bestA.cost;
            cur = bestG;
            curRes = bestRes;
        }
        return curRes;
    }
}
