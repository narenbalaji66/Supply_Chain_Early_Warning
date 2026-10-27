# Supply-Chain Early Warning (Multilayer Temporal Graph)

Early-warning and mitigation decision support for supply chains. Requires Java 17+, no external libraries.

## Layout (one concept per file, one owner per file)

| File | Concept | Suggested owner |
|------|---------|-----------------|
| `src/supplychain/MultilayerGraph.java` | Graph model: layers, nodes, edges, time lags and validity windows | Person 1 |
| `src/supplychain/AnomalyDetector.java` | EWMA z-score anomaly detection -> disruption seeds | Person 2 |
| `src/supplychain/CascadeSimulator.java` | Monte-Carlo cascade simulation + early-warning output | Person 3 |
| `src/supplychain/MitigationPlanner.java` | Candidate actions, ranking, greedy budgeted plan | Person 4 |
| `src/supplychain/SupplyChainEarlyWarning.java` | Main: demo data, console input, pipeline wiring | Person 5 |

Dependencies flow one way: `MultilayerGraph` <- `AnomalyDetector`, `CascadeSimulator` <- `MitigationPlanner` <- `Main`.

## Build and run

```bash
mkdir -p out
javac -d out src/supplychain/*.java
java -cp out supplychain.SupplyChainEarlyWarning
```

At the prompt choose `d` for demo data or `e` to type your own delay-ratio readings, then enter a budget (default 40).
