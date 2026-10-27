package supplychain;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * CONCEPT 1 - Multilayer Temporal Graph (MTG) model.
 *
 *  Layers  : PHYSICAL (goods flow), INFORMATION (orders / backlog signals), FINANCIAL (payments / credit)
 *  Nodes   : one node per (entity, layer)   -> e.g. PORT@PHYSICAL, PORT@FINANCIAL
 *  Edges   : intra-layer dependencies + inter-layer couplings (same entity, different layer)
 *  Time    : every edge has a propagation lag and a validity window [activeFrom, activeTo]
 */
public final class MultilayerGraph {

    public enum Layer { PHYSICAL, INFORMATION, FINANCIAL }

    public static final class Node {
        public final int idx;
        public final String entity;
        public final Layer layer;
        public final double importance;   // business value of this node (revenue share, criticality)
        public double threshold;          // absorption capacity (buffer): failure when load >= threshold

        Node(int idx, String entity, Layer layer, double importance, double threshold) {
            this.idx = idx; this.entity = entity; this.layer = layer;
            this.importance = importance; this.threshold = threshold;
        }

        public String key() { return entity + "@" + layer; }
    }

    public static final class Edge {
        public final int id, src, dst, lag, activeFrom, activeTo;
        public double weight;             // transmission strength 0..1

        Edge(int id, int src, int dst, double weight, int lag, int activeFrom, int activeTo) {
            this.id = id; this.src = src; this.dst = dst; this.weight = weight;
            this.lag = lag; this.activeFrom = activeFrom; this.activeTo = activeTo;
        }

        public boolean activeAt(int t) { return t >= activeFrom && t <= activeTo; }
    }

    public final List<Node> nodes = new ArrayList<>();
    public final List<Edge> edges = new ArrayList<>();
    private final Map<String, Integer> index = new HashMap<>();

    public Node addNode(String entity, Layer layer, double importance, double threshold) {
        Node n = new Node(nodes.size(), entity, layer, importance, threshold);
        nodes.add(n);
        index.put(n.key(), n.idx);
        return n;
    }

    public Node node(String entity, Layer layer) {
        Integer i = index.get(entity + "@" + layer);
        if (i == null) throw new IllegalArgumentException("Unknown node " + entity + "@" + layer);
        return nodes.get(i);
    }

    /** lag is forced to be >= 1 so that propagation is strictly causal in discrete time. */
    public Edge addEdge(Node from, Node to, double weight, int lag, int activeFrom, int activeTo) {
        Edge e = new Edge(edges.size(), from.idx, to.idx, weight, Math.max(1, lag), activeFrom, activeTo);
        edges.add(e);
        return e;
    }

    public void link(Layer layer, String from, String to, double w, int lag) {
        link(layer, from, to, w, lag, 0, Integer.MAX_VALUE);
    }

    public void link(Layer layer, String from, String to, double w, int lag, int activeFrom, int activeTo) {
        addEdge(node(from, layer), node(to, layer), w, lag, activeFrom, activeTo);
    }

    /** Deep copy, so mitigation actions can be tried on a scenario without touching the original. */
    public MultilayerGraph copy() {
        MultilayerGraph g = new MultilayerGraph();
        for (Node n : nodes) g.addNode(n.entity, n.layer, n.importance, n.threshold);
        for (Edge e : edges) g.edges.add(new Edge(e.id, e.src, e.dst, e.weight, e.lag, e.activeFrom, e.activeTo));
        return g;
    }

    public double totalImportance() {
        double s = 0;
        for (Node n : nodes) s += n.importance;
        return s;
    }
}
