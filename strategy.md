# ANN3 Project Strategy

## 1. Objective

We will build a reproducible benchmark of approximate nearest-neighbor (ANN) indexes, with particular emphasis on quantization, memory use, search quality, throughput, tail latency, and reranking.

The project should answer one central systems question:

> Under fixed hardware and comparable recall, which vector representation and index design provides the best latency, throughput, and memory trade-off, and how much quality can exact reranking recover?

The required project is **ANN3** from [Class_Projects_2026-27.pdf](./Class_Projects_2026-27.pdf):

- FAISS Flat, scalar quantization (SQ), product quantization (PQ), and HNSW
- FAISS RaBitQ
- Binary quantization with reranking
- Three datasets with different dimensionalities and similarity metrics
- Index construction measurements
- Recall-QPS curves and tail latency
- Single-threaded and multithreaded performance
- Quantization quality versus memory, with and without reranking

Our priorities, in order, are:

1. Satisfy every grading requirement.
2. Make every comparison methodologically fair.
3. Produce a strong systems-engineering artifact.
4. Keep the experiment matrix controlled enough to finish.
5. Add GPU results only where they are genuinely comparable and useful.

## 2. Final project outputs

By submission time, the repository should contain:

- A technical report in PDF form
- A clean Git repository with a release tag matching the submitted report
- A pinned CPU environment and an optional pinned GPU environment
- Dataset download and verification scripts
- A common benchmark harness for all indexes
- Index-build scripts
- Search and reranking scripts
- Hardware and software inventory
- Machine-readable raw results
- Processed summary tables
- Publication-quality figures
- A one-command or short documented reproduction path
- A concise README explaining the conclusions

The report should not merely identify the fastest method. It should explain *why* each method behaves as it does and under which resource and recall requirements it should be selected.

## 3. Hardware strategy

### 3.1 Official benchmark machine

Use the Linux desktop as the sole source of official performance numbers:

- Intel Core i9-12900K
- NVIDIA RTX 5080 with 16 GB VRAM
- 32 GB system RAM
- Linux
- Local SSD/NVMe storage

Record the exact Linux distribution, kernel, compiler, CUDA, driver, FAISS, BLAS, Python, NumPy, and library versions.

### 3.2 MacBook role

Use the M4 Pro MacBook for:

- Developing the harness
- Unit and correctness tests on small datasets
- Inspecting raw results
- Statistical analysis
- Plotting
- Writing the report
- Remotely controlling the Linux desktop

Do not mix MacBook and Linux timings in the same performance comparison. If we run a portability experiment on the MacBook, label it as a separate appendix and do not use it to rank the main methods.

### 3.3 CPU versus GPU separation

The primary comparison will be CPU-only because every required ANN3 method must be compared on equivalent hardware.

The GPU study will be a separate extension containing only indexes with legitimate GPU implementations, such as Flat, IVF-Flat, and IVF-PQ. RaBitQ and other CPU-only methods must not be presented as slower merely because they lack equivalent GPU implementations.

Main report structure:

1. Fair CPU comparison of all required methods
2. Quantization and reranking study
3. Separate CPU-versus-GPU acceleration study for supported indexes

## 4. Research questions

The experiments should be organized around explicit questions.

### RQ1: Quality and speed

At equal recall targets, which index provides the highest single-threaded QPS and lowest p95/p99 latency?

### RQ2: Memory efficiency

How much index memory does SQ, PQ, RaBitQ, and binary quantization save relative to full-precision Flat and HNSW?

### RQ3: Reranking

How much recall can be recovered by reranking compressed-index candidates using the original float vectors, and what latency and memory costs does this impose?

### RQ4: Dataset dependence

Do the conclusions change with dimensionality, similarity metric, or embedding distribution?

### RQ5: Parallel scaling

How does throughput scale across 1, 2, 4, and 8 CPU workers, and where does scaling efficiency begin to decline?

### RQ6: GPU acceleration

For indexes with equivalent GPU implementations, how do batch size, transfer overhead, and VRAM use affect end-to-end performance?

## 5. Dataset plan

Use three datasets of up to approximately one million base vectors each.

### Dataset A: SIFT1M

- Approximately 1 million vectors
- 128 dimensions
- Euclidean/L2 distance
- Established ground-truth neighbors
- Represents relatively low-dimensional dense vectors

### Dataset B: GloVe-100

- Approximately 1 million vectors
- 100 dimensions
- Cosine/angular similarity
- Normalize base and query vectors once, then use inner product consistently
- Represents word embeddings

### Dataset C: modern text embeddings

- 384-1024 dimensions
- Approximately 1 million vectors or a reproducible one-million-vector subset
- Cosine or inner-product similarity
- Prefer a dataset from VIBE or another source with documented ground truth
- Freeze the exact dataset choice during the initial technical spike

Selection criteria for Dataset C:

- Publicly downloadable without paid services
- Stable URL or versioned source
- License permits academic use
- Ground truth is supplied or can be computed exactly
- Fits the project scale on 32 GB RAM
- Has enough queries to estimate p95 and p99 latency reliably

For every dataset, store a manifest containing:

- Source URL
- Download date
- Checksums
- Vector count and dimensionality
- Query count
- Data type
- Similarity metric
- Normalization performed
- Subsampling seed and procedure, if applicable

## 6. Indexes to evaluate

The final API choices must be confirmed in a short technical spike before the full harness is written. The target method set is:

### 6.1 Exact baseline

- FAISS `IndexFlatL2` or `IndexFlatIP`
- Provides exact recall and an uncompressed speed/memory reference
- Also validates supplied ground truth on a sample

### 6.2 Scalar quantization

- FAISS scalar-quantized index
- Primary configuration: 8-bit scalar quantization
- Optional second configuration: FP16 or 4-bit if supported cleanly and fairly
- Report code size and total index size

### 6.3 Product quantization

- FAISS PQ and/or IVF-PQ
- Use dimension-compatible values of `m`
- Keep code size comparable across datasets where possible
- Use IVF-PQ for recall-QPS curves through `nprobe`

Candidate PQ subquantizer values:

| Dataset | Candidate `m` values |
|---|---:|
| SIFT, 128d | 16, 32 |
| GloVe, 100d | 10, 20 |
| Text, 768d example | 48, 96 |

These values are provisional and must be validated for divisibility and practical build time.

### 6.4 HNSW

- FAISS HNSW with full-precision vectors
- Construction grid: `M` in approximately {16, 32}; `efConstruction` in approximately {100, 200}
- Search grid: `efSearch` in approximately {16, 32, 64, 128, 256}
- Reduce the construction grid after the pilot if configurations are clearly dominated

### 6.5 RaBitQ

- Use the current pinned FAISS RaBitQ implementation
- Evaluate one-bit and selected multi-bit modes supported by the pinned release
- If using IVF-RaBitQ, vary `nprobe` to produce recall-QPS curves
- Record random-rotation settings and seeds
- Confirm serialization, code size, and reranking compatibility in the technical spike

### 6.6 Binary quantization with reranking

- Define and document the float-to-binary transformation
- Begin with centered sign quantization
- Add a fixed seeded random rotation if required for a fairer/stronger binary baseline
- Search binary codes using Hamming distance
- Retrieve more than `k=10` candidates
- Rerank candidates using exact distance on the original float vectors

Candidate reranking depths:

- 10: no additional candidate margin
- 20
- 50
- 100
- 200

For every reranking result, report both:

- Compressed-index memory alone
- Total memory including the retained full-precision vectors required for exact reranking

This prevents us from claiming unrealistic memory savings.

## 7. Controlled experiment matrix

The project can become unmanageably large. Use a staged matrix rather than taking the Cartesian product of every possible setting.

### Stage A: correctness matrix

Run all methods on 10,000-50,000 vectors to verify:

- Index construction
- Serialization and reloading
- Distance metric correctness
- Recall calculation
- Deterministic seeds
- Reranking correctness
- Result schema

No performance conclusions should be drawn from this stage.

### Stage B: pilot matrix

On one representative full-scale dataset:

- Run broad but coarse parameter sweeps
- Identify broken or redundant settings
- Estimate total machine time
- Choose final parameter ranges that span low to high recall
- Ensure curves reach recall 0.90 and 0.95 when technically possible

### Stage C: final CPU matrix

Run the selected parameter grid on all three datasets.

Required outputs:

- Construction time
- Peak construction memory
- Serialized index size
- Recall@10
- Single-threaded QPS
- p50, p95, and p99 latency
- Multithreaded QPS at one fixed recall level

### Stage D: reranking matrix

For PQ, RaBitQ, and binary quantization where applicable:

- Select representative compressed configurations
- Evaluate reranking depths {10, 20, 50, 100, 200}
- Measure recall recovery
- Measure incremental latency
- Measure full memory cost

### Stage E: optional GPU matrix

For supported equivalent GPU indexes:

- Flat
- IVF-Flat
- IVF-PQ

Candidate batch sizes:

- 1
- 32
- 256
- 1024

Report two timing modes:

1. Index and queries already resident on the GPU
2. End-to-end timing including host-device transfers

Stop the GPU extension if it threatens completion of the required CPU study.

## 8. Benchmark protocol

### 8.1 Fairness rules

- Use the same base vectors and query vectors for every method.
- Use the correct metric consistently.
- Use the same `k=10`.
- Compare QPS and latency at equal recall, not at arbitrary defaults.
- Give comparable tuning effort to each method.
- Record every non-default parameter.
- Never combine timings from different machines in one comparison.
- Do not include index-loading time in search latency unless explicitly studying startup.
- Do not include ground-truth computation in index-build time.

### 8.2 Warm and cold behavior

The main recall-QPS comparison should use warm indexes because it is the most reproducible default for in-memory ANN.

For every run:

1. Load or build the index.
2. Execute a documented warm-up set that is not included in measurements.
3. Execute the fixed measured query set.
4. Save per-query latency and run-level throughput.

If we study cold behavior, present it separately and define exactly how caches were cleared or why a process restart is used as an approximation.

### 8.3 Repetitions and statistics

- Repeat every timing measurement at least five times.
- Use the same query order or a fixed set of seeded permutations.
- Report the median run-level QPS with a 95% bootstrap confidence interval.
- Report p50, p95, and p99 from per-query latency.
- Also report variability between runs.
- Recall is deterministic for a fixed index, query set, and seed; verify it for every measured configuration.

### 8.4 Single-threaded search

For the primary recall-QPS curves:

- Pin execution to one performance core.
- Set FAISS/OpenMP threads to one.
- Use a closed-loop client.
- Issue queries one at a time for latency measurements.
- Do not silently use batched BLAS for a result labeled single-query latency.

### 8.5 Multithreaded scaling

At a representative recall target, ideally recall 0.95:

- Test 1, 2, 4, and 8 workers pinned to the 12900K performance cores.
- Optionally add an all-logical-CPU result, clearly labeled.
- Keep client behavior constant.
- Report QPS, speedup, parallel efficiency, and tail latency.

Use:

```text
speedup(p) = throughput(p) / throughput(1)
parallel_efficiency(p) = speedup(p) / p
```

Because the 12900K has performance and efficiency cores, do not allow uncontrolled scheduling to distort the main scaling curve.

### 8.6 Machine-state controls

Before official runs:

- Use a fixed Linux CPU governor or document the chosen power policy.
- Stop unnecessary background workloads.
- Record CPU temperatures and watch for thermal throttling.
- Pin CPU affinity.
- Fix OpenMP environment variables.
- Record GPU power/temperature state for GPU runs.
- Run from local storage.
- Maintain adequate free disk space.
- Reboot or restart the environment before a major final run if necessary, and document it.

## 9. Measurements and result schema

Every raw measurement row should contain enough information to reproduce and group it.

Suggested fields:

```text
timestamp
git_commit
machine_id
dataset
dataset_checksum
metric
index_family
index_parameters
search_parameters
rerank_depth
seed
run_number
thread_count
cpu_affinity
batch_size
execution_device
warm_or_cold
build_time_seconds
peak_rss_bytes
serialized_index_bytes
search_seconds
query_count
qps
recall_at_10
p50_ms
p95_ms
p99_ms
gpu_memory_bytes
notes
```

Store:

- Per-run summaries as CSV or Parquet
- Per-query latencies in compressed Parquet/NumPy form
- Configuration files in YAML or JSON
- Logs separately from clean result tables

Never manually copy numbers into the report. Generate every table and figure from saved raw results.

## 10. Systems instrumentation

Required instrumentation:

- Wall-clock time using a monotonic clock
- Peak RSS
- Serialized index size
- CPU utilization
- GPU memory for GPU runs

Useful optional instrumentation:

- `perf stat` for cycles, instructions, cache misses, and branches
- Disk and page-fault measurements if unexpected paging occurs
- CPU and GPU temperature/power logs

Hardware-counter analysis should be limited to a few representative configurations. Do not run detailed profiling across the entire matrix.

## 11. Figures and tables to produce

At minimum:

1. Dataset characteristics table
2. Hardware/software configuration table
3. Method and parameter table
4. Index build time by dataset and method
5. Peak build memory by dataset and method
6. Serialized index size or bytes per vector
7. Recall@10 versus single-threaded QPS
8. Recall@10 versus p99 latency
9. QPS and latency at recall 0.90
10. QPS and latency at recall 0.95
11. Memory saving versus recall loss
12. Reranking depth versus recall
13. Reranking depth versus p99 latency
14. Full memory cost with and without retained reranking vectors
15. Multithreaded speedup and parallel efficiency
16. Optional CPU-versus-GPU performance by batch size

Every plot should include:

- Units
- Dataset name
- Distance metric
- Thread or device configuration
- Clear legend
- Error bars where applicable
- A caption that states the main observation

## 12. Analysis approach

The discussion should explain observed behavior using systems concepts:

- Bytes read per candidate
- Cache locality
- Memory bandwidth
- SIMD-friendly layouts
- Random versus sequential memory access
- Quantization decode cost
- Candidate-list size
- Reranking computation
- Index-construction overhead
- Multithreaded contention or bandwidth saturation
- CPU-GPU transfer overhead
- Query batching

Avoid conclusions such as "PQ is fastest" without conditions. Prefer:

> On Dataset X at recall 0.95, configuration A achieved the highest QPS because its smaller codes reduced memory traffic; however, retaining full vectors for reranking reduced the apparent memory saving from X to Y.

We should explicitly identify cases where conclusions change by dataset.

## 13. Repository design

Proposed structure:

```text
.
├── README.md
├── strategy.md
├── pyproject.toml
├── requirements-cpu.lock
├── requirements-gpu.lock
├── Dockerfile.cpu
├── Dockerfile.gpu
├── Makefile
├── configs/
│   ├── datasets/
│   ├── indexes/
│   └── experiments/
├── src/
│   ├── datasets/
│   ├── indexes/
│   ├── benchmark/
│   ├── metrics/
│   └── monitoring/
├── scripts/
│   ├── download_datasets.sh
│   ├── verify_datasets.sh
│   ├── run_correctness.sh
│   ├── run_pilot.sh
│   ├── run_cpu_matrix.sh
│   ├── run_reranking.sh
│   ├── run_gpu_matrix.sh
│   └── generate_report_assets.sh
├── tests/
├── data/
│   └── manifests/
├── results/
│   ├── raw/
│   ├── processed/
│   └── logs/
├── figures/
├── report/
└── docs/
    ├── methodology.md
    └── reproduction.md
```

Large datasets and generated indexes should not be committed to Git. Commit manifests, checksums, small fixtures, configurations, and scripts.

## 14. Implementation phases

### Phase 0: freeze scope

Deliverables:

- Confirm the three datasets
- Confirm the pinned FAISS version
- Confirm available RaBitQ APIs
- Confirm the binary-quantization transformation
- Decide which GPU indexes are genuinely equivalent
- Write the final method/parameter matrix

Exit criterion: every planned experiment maps to a specific research question and required rubric item.

### Phase 1: environment and repository

Deliverables:

- Initialize repository structure
- CPU environment works on Linux
- Optional GPU environment works on Linux
- Hardware inventory script
- Reproducible smoke test
- CI or local test command for small synthetic data

Exit criterion: a fresh environment can build a small Flat index and calculate recall.

### Phase 2: dataset pipeline

Deliverables:

- Download scripts
- Checksums and manifests
- Normalization code
- Query/base/ground-truth loaders
- Small deterministic fixtures for tests

Exit criterion: all three datasets load through the same interface and supplied ground truth is validated against exact search on a sample.

### Phase 3: common benchmark harness

Deliverables:

- Common index adapter interface
- Build-time measurement
- Peak-memory measurement
- Search timing
- Recall calculation
- Per-query latency capture
- Structured result writing
- Seed and configuration capture

Exit criterion: Flat and one approximate index produce correct, reproducible raw result files.

### Phase 4: integrate all methods

Recommended order:

1. Flat
2. HNSW
3. SQ
4. PQ/IVF-PQ
5. RaBitQ
6. Binary quantization
7. Exact reranking

Exit criterion: every required method passes correctness tests on the small fixture and one real-data subset.

### Phase 5: pilot

Deliverables:

- One full-scale dataset tested
- Preliminary recall-QPS curves
- Estimated time and memory for every method
- Final reduced parameter grid
- Identified failure modes

Exit criterion: we know the complete final matrix will fit the available calendar and hardware.

### Phase 6: final CPU experiments

Run:

- All datasets
- All selected construction settings
- All selected search settings
- Five timing repetitions
- Single-threaded search
- Fixed-recall multithreaded scaling

Do not change code or dependencies halfway through this phase without invalidating and rerunning affected results.

### Phase 7: quantization and reranking analysis

Run:

- Selected compressed indexes
- Fixed candidate depths
- Exact reranking
- Memory accounting with full vectors included
- Quality-cost frontier generation

### Phase 8: optional GPU experiments

Only begin after the required CPU results are complete and backed up.

### Phase 9: analysis and report

Deliverables:

- Clean result tables
- Final figures generated from scripts
- Written answers to every research question
- Threats-to-validity section
- Reproduction instructions

### Phase 10: independent reproduction and release

Before submission:

- Create a fresh environment
- Run the documented smoke reproduction
- Recreate at least one representative result end to end
- Verify every figure can be regenerated
- Check that every reported number exists in processed results
- Tag the exact submitted commit
- Archive raw results and generated SQL-free logs/configurations

## 15. Suggested timeline

The timeline is expressed in relative weeks so it can be mapped onto the actual deadline.

| Week | Goal |
|---:|---|
| 1 | Scope freeze, literature reading, environments, technical spike |
| 2 | Dataset pipeline and ground-truth verification |
| 3 | Common benchmark harness and Flat/HNSW |
| 4 | SQ, PQ, RaBitQ, binary quantization, reranking correctness |
| 5 | Full-scale pilot and parameter-grid reduction |
| 6 | Final CPU construction and single-thread experiments |
| 7 | Multithreaded scaling and reranking experiments |
| 8 | GPU extension and targeted profiling, if time remains |
| 9 | Statistical analysis, figures, and first complete report |
| 10 | Reproduction audit, revisions, release tag, submission package |

Start writing the methodology and dataset sections during Weeks 2-4. Do not wait for the results before beginning the report.

## 16. Team division

If there are two people, use primary ownership without creating isolated codebases.

### Person A: harness and systems measurements

- Environment and builds
- Benchmark runner
- CPU affinity and thread control
- Timing and resource monitoring
- HNSW and GPU integrations
- Reproducibility packaging

### Person B: datasets and quantization analysis

- Dataset pipeline
- Ground-truth validation
- SQ, PQ, RaBitQ, and binary encoding
- Reranking implementation
- Statistical processing and plotting
- Result interpretation

Shared responsibilities:

- Experiment design
- Code review
- Pilot decisions
- Final runs
- Report writing
- Reproduction audit

Each person should be able to explain every method and figure during evaluation.

## 17. Risk register

### Risk: RaBitQ API/version instability

Mitigation:

- Pin FAISS immediately after the technical spike
- Create a dedicated RaBitQ correctness test
- Save a small serialized test index
- Avoid upgrading dependencies after pilot results exist

### Risk: experiment matrix becomes too large

Mitigation:

- Use the pilot to eliminate dominated settings
- Preserve at least 4-6 search points per curve
- Prefer representative construction settings over exhaustive combinations
- Treat GPU work as optional

### Risk: out-of-memory failures

Mitigation:

- Process one dataset and index at a time
- Use memory mapping where appropriate
- Avoid unnecessary NumPy copies
- Persist and free intermediates explicitly
- Monitor peak RSS during the pilot

### Risk: unfair results

Mitigation:

- Centralize all timing and recall logic
- Keep one configuration schema
- Compare at equal recall
- Separate CPU and GPU studies
- Report total reranking memory honestly

### Risk: noisy CPU results

Mitigation:

- Pin work to 12900K performance cores
- Fix thread counts and power policy
- Warm indexes consistently
- Run at least five repetitions
- Record temperature and background-load anomalies

### Risk: a long run fails late

Mitigation:

- Write results incrementally
- Make runs resumable by configuration ID
- Save logs for each configuration
- Checkpoint built indexes when economical
- Validate free disk and memory before dispatch

### Risk: good numbers but weak report

Mitigation:

- Organize experiments around the six research questions
- Write expected analysis prompts before running experiments
- Produce figures continuously
- Explain mechanisms, not only rankings

## 18. Report outline

1. Abstract
2. Introduction and motivation
3. Background
   - Exact and approximate nearest-neighbor search
   - HNSW, IVF, SQ, PQ, RaBitQ, and binary quantization
   - Reranking
4. Research questions
5. Experimental methodology
   - Hardware and software
   - Datasets
   - Index configurations
   - Metrics
   - Statistical protocol
6. Index-construction results
7. Recall, throughput, and tail-latency results
8. Memory and quantization trade-offs
9. Reranking results
10. Parallel-scaling results
11. Optional GPU results
12. Discussion and selection guide
13. Threats to validity
14. Reproducibility
15. Conclusion

## 19. Definition of done

The project is complete only when:

- Every required method runs on all three datasets or any exception is explicitly justified.
- Every timing configuration has at least five repetitions.
- Recall-QPS curves contain enough points to show the trade-off.
- Comparisons at recall 0.90 and 0.95 are reported where reachable.
- p95 and p99 latency are reported.
- Build time, index size, and peak memory are reported.
- Multithreaded QPS is measured at a fixed recall level.
- Quantization memory savings are measured.
- Reranking is evaluated for quality and cost.
- CPU and GPU conclusions are kept separate.
- Raw results and configuration files are preserved.
- Figures and tables are generated automatically.
- A fresh environment can reproduce a representative experiment.
- The submitted report points to the exact tagged Git commit.

## 20. Immediate next actions

1. Confirm the project deadline and team size.
2. Inventory the Linux machine and SSD capacity.
3. Initialize the repository structure.
4. Pin a current FAISS version after testing RaBitQ, CPU, CUDA, and serialization.
5. Select and verify the modern text-embedding dataset.
6. Implement the exact Flat baseline and recall evaluator.
7. Run a 10,000-vector end-to-end smoke test.
8. Integrate one approximate method and validate the result schema.
9. Freeze the detailed parameter matrix after the full-scale pilot.

The key strategic rule is simple: **finish the complete, fair CPU experiment before spending time on impressive optional GPU work.**
