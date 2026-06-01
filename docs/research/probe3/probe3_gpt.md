Feasibility Research Brief for Kind Probe 3 Dream-State System
Executive summary
The uploaded project brief identifies the topic as Kind Probe 3, a research probe about whether Io, a single-agent RSSM-based system, can undergo dream-like offline change while the environment is off but Io is still running. The brief fixes several non-negotiables: self-opacity by default, no explicit self-model or self-optimization machinery, no installed drive for self-continuation, a one-way mirror that reads but does not write into Io’s data plane, and a four-state model in which dreaming is distinct from both dormancy and pause. The brief also already names two constraints that are technically decisive: pure prior rollout is unlikely to satisfy the mirror’s dream-side evidential criterion, and Dreamer-style model rollouts already occur during waking-time planning, so “dreaming” cannot just mean “more of the same rollout code.” 

Given those constraints, the strongest first-build scope is not pure prior rollout alone. PlaNet and Dreamer established that latent stochastic-deterministic world models and imagined trajectories are powerful for planning and policy improvement, and DreamerV3 shows that the same family scales across many domains; but in this project that exact strength becomes a confound, because ordinary planning already uses imagined trajectories. The most defensible MVP is therefore replay-anchored generative recombination: start dreams from replayed latent states or compact episode fragments, branch them with a policy-decoupled or weakly policy-conditioned rollout regime, apply a distinct perturbation/temperature policy, and log enough provenance that the mirror can separate replay, planning, and dream sessions. 
 

The recommended deployment topology is hybrid local-first rather than full cloud. The brief explicitly commits dream-to-wake variability to conditions outside the simulation, including the builder’s desktop on/off patterns, which argues for a runtime close to the workstation. At the same time, append-only telemetry, metadata storage, mirror analytics, and experiment tracking benefit from standard system components: a Python/PyTorch runtime, FastAPI/OpenAPI control plane, PostgreSQL metadata store, Parquet trace storage queried via DuckDB, and vendor-neutral observability via OpenTelemetry. Kubernetes is optional for later scale, not necessary for an MVP. 
 

The main feasibility finding is that the hard problem is epistemic, not infrastructural. Replay, world-model imagination, generative replay, and biologically inspired offline consolidation all have strong precedent. What is novel here is the combination of self-opacity, one-way interpretation, and the requirement that dream evidence be legible to the mirror without becoming covert planning or covert self-optimization. That makes the load-bearing decisions: dream taxonomy, gradient-flow policy, telemetry schema, bounded mirror-to-runtime process control, and success metrics that distinguish dream sessions from ordinary planning and from pure-GRU baseline dynamics. 
 

A credible MVP is feasible in roughly 5.5 to 8.5 person-months, with a more robust research platform requiring closer to 8.5 to 12 person-months. Monthly infrastructure can remain modest for single-GPU burst research: for example, RunPod lists L4 at $0.44/hr, A100 PCIe at $1.19/hr, and standard network storage at $0.05–$0.07/GB-month; Datadog Infrastructure Pro starts at $15/host-month billed annually; Weights & Biases Pro starts at $60/month, while W&B’s academic Pro program is free for qualifying academic users and includes 200 GB of cloud storage. Staffing, not compute, will dominate total cost. 

Project framing, scope, and users
This report interprets the otherwise “unspecified project topic” as the topic in the uploaded brief: Kind Probe 3, the dream-state probe. The brief already decides the substrate lineage, the four-state model, the mirror’s role, and several stance commitments, while leaving open the dream sub-modes, telemetry shape, trigger/exit policy, idling mechanism, dream content, learning policy, success definition, and minimal first build. Those open items are where the technical brief must stay explicit about uncertainty rather than silently inventing a spec. 

Area	Current status	Working interpretation for this brief
Project topic	Specified by upload	Kind Probe 3 dream-state subsystem
Substrate	Specified by upload	Custom minimal RSSM with PlaNet/Dreamer lineage
Mirror architecture	Specified by upload	One-way reader; bounded process-control exception is still open
Deployment scale	Unspecified	Start as single-user research system, scale only if the probe succeeds
External product users	Unspecified	Treat primary users as internal research personas, not consumers
Regulatory regime	Unspecified	Minimal by default, but privacy obligations appear if builder-context signals are stored
Budget	Unspecified	Present low, medium, and cloud-heavy cost envelopes rather than one committed budget

The most important scope choice is what counts as a dream mode. Replay and generative simulation are the most mature precedents in both ML and neuroscience; generative replay is a direct continual-learning analogue; hippocampal replay and ripple literature provide a strong biological consolidation precedent; and predictive-processing accounts of dreaming provide a principled story for offline model maintenance. By contrast, lucid control is a poor first-build fit because lucid-dreaming literature is bound up with metacognition and reflective access, which pulls directly against the brief’s self-opacity commitment. Associative/noise-annealed activity is scientifically attractive, but it is much easier to confuse with drift if introduced before a stronger baseline exists. 
 

Scope option	What it actually builds	Fit to Kind	Main tradeoff
Replay-only consolidation	Offline reprocessing of stored episodes or latent fragments, possibly prioritized by novelty/uncertainty	High interpretability, weak “dreamness”	Risks looking like maintenance rather than a distinct dream state
Replay plus counterfactual recombination	Start from replay anchors, then branch into policy-decoupled or weakly policy-coupled rollout under a distinct perturbation regime	Best MVP fit	Requires richer telemetry and control experiments
Pure prior-only generative simulation	Free-run the RSSM prior without replay anchoring	Weak as sole mode	Too close to GRU baseline and waking planning, matching Concern A/B
Associative or nonsense mode	Higher-temperature, weakly anchored internal generation with relaxed coherence constraints	Strong later experiment	Easy to mistake for noise without a baseline
Parameter-space sleep	Offline regularization, renormalization, or compression without narrative-like rollouts	Interesting but probably closer to dormant or a later probe	Mirror sees less “content,” so it may answer a different question

The recommended scope sequence is therefore: MVP = replay plus counterfactual recombination, Phase 2 = add associative/noise-annealed channel, and defer parameter-space sleep to either dormant-state work or a separate follow-on experiment. That sequence best matches the brief’s “build to understand” stance, because it gives the mirror structured offline content that is neither mere replay nor mere planning, while postponing the highest-ambiguity mode until there is something to compare it against. 
 

Because external end users are unspecified, the most plausible target users are internal research personas:

Persona	Primary job to be done	Why they matter
Builder-researcher	Decide whether Probe 3 has taught the project anything real about dreaming and subjectivity	Primary decision-maker for scope and interpretation
ML systems engineer	Implement the state scheduler, dream engine, telemetry, and experiment harness	Makes the probe reproducible rather than anecdotal
Mirror analyst	Inspect dream telemetry, compare criteria surfaces, and detect coherence or degeneracy	Needed because the mirror is the official reader
Safety/privacy reviewer	Bound builder-context capture, review one-way invariants, and minimize leakage	Important because the brief couples runtime behavior to real-world context

Requirements and success criteria
Concern A and Concern B in the uploaded brief should be treated as requirements, not commentary. Concern A means the dream system must generate mirror-readable structure beyond ordinary GRU baseline coupling under prior-only rollout; Concern B means dream rollouts must be distinguishable from waking planning rollouts in either provenance, control policy, perturbation regime, or downstream effects. Dreamer’s and PlaNet’s successes make these concerns stronger, not weaker, because they confirm that imagined rollouts are normal machinery in world-model agents. 
 

Load-bearing decision	Why it constrains everything downstream	Recommended stance
The definition of a dream	Determines telemetry, mirror criteria, and ablation design	Dream = replay-anchored recombination plus distinct perturbation regime
Gradient flow from dreams	Changes whether Probe 3 is about observation, consolidation, or covert optimization	MVP should update world-model-side components only, not the actor
Mirror-to-runtime signal	Determines whether quality-based idling breaks the one-way invariant	Allow only coarse content-blind process control: continue, idle, stop, resample
Telemetry schema	Determines whether the mirror can distinguish planning, replay, and dream sessions	Amend schema before implementation, do not treat current DreamRollout as sufficient
Trigger and exit policy	Shapes months-long runtime behavior and cost	External trigger plus internal guardrail is the cleanest first design
Associative mode timing	Affects interpretability and thesis risk	Add only after a replay-recombination baseline exists

A cautious gradient policy is especially important. Sleep replay, replay disruption experiments, and generative replay all support the idea that offline processing can matter for memory or model stability without making “dreaming” simply another action-optimization loop. For that reason, the MVP should permit at most world-model-side consolidation updates from dream sessions and keep the actor frozen during dream-state learning. A later ablation can test actor updates explicitly, but starting with them would entangle the probe with the very self-optimization concern the brief wants to avoid. 
 

Functional requirement	Meaning in this project
Four-state scheduler	Runtime must explicitly represent waking, dreaming, dormant, and paused
Dream mode registry	Runtime must support at least replay-recombination mode and a clean slot for associative mode
Distinct provenance	Every dream session must record how it differs from waking planning and prior-only controls
Replay anchor selection	Dream sessions must be seeded from replayable waking traces or latent anchors
Counterfactual brancher	System must generate novel continuations from replay anchors under a distinct regime
Mirror-readable telemetry	Dream outputs must be legible at session and sequence level to the existing mirror
Bounded idling control	Mirror may request coarse process actions, never content injection
Reproducibility package	Every session must record seeds, checkpoints, schema version, and trigger context
Evaluation harness	System must run pure-prior, planning-rollout, and shuffled-time controls side by side

Nonfunctional requirement	Recommended target	Why
Performance	Dream mode should outpace waking environment throughput; exact target is unspecified, but the offline loop should be materially faster than real-time environment stepping	Otherwise background change will be too weak to study
Scalability	Single-node first; clean path to multi-GPU and distributed analytics if needed	Prevents premature platform work
Security	Append-only telemetry, least-privilege access, signed artifacts/checkpoint hashes, segmented mirror/runtime permissions	Protects one-way boundary and experiment integrity
Privacy	Store only coarse builder-context signals unless stronger justification exists	Desktop/life-pattern coupling can become personal data
Compliance	Minimal baseline if local-only and single-user; stronger controls if cloud-hosted collaboration or identifiable behavioral telemetry is retained	Regulatory burden depends on deployment, which is still unspecified
Reproducibility	Deterministic seed and checkpoint tracking for every dream session	Essential because “dream evidence” is otherwise too easy to over-interpret
Reliability	Idling and stop conditions must protect against runaway offline loops	Runtime may run for months
Observability	Metrics, traces, and logs must correlate dream sessions, mirror verdicts, and runtime actions	Necessary for debugging boundary violations

A bounded control signal from mirror to runtime can be made compatible with the spirit of self-opacity if it remains content-blind. In practical terms, that means the mirror may return only a small finite-state process signal such as continue, idle, stop, or resample, never a latent vector, never a textual dream summary, and never any policy advice. Architecturally, this is the least-privilege and zero-trust interpretation of the brief’s open question about quality-based idling. 
 

The success criterion is best operationalized as a stack of tests, not one test. The minimum successful outcome is that dream sessions are distinguishable from constant/noise floor, from pure prior-only controls, and from waking-time planning rollouts. A stronger outcome is that mirror descriptions stay coherent and stable under reseeding and paraphrase. The strongest outcome is limited, post-dream change in subsequent waking behavior or model calibration without introducing actor-side self-optimization. That hierarchy fits the brief’s emphasis on “evidence of it” while keeping more ambitious findings clearly optional. 
 

Architecture, integrations, and stack
Because the brief couples dream frequency to real-world context such as desktop on/off patterns, a fully cloud-native design is not the best first choice. A local-first or hybrid topology gives better control over the workstation event source, lower privacy risk, and simpler preservation of the mirror/runtime boundary. Full cloud only becomes advantageous if long-running multi-GPU experiments or external collaboration become central. 
 

Deployment option	What runs where	Best use case	Advantages	Disadvantages
Local-first	Io runtime, scheduler, mirror, metadata DB, and analytics on one workstation or lab server	Earliest MVP, strongest privacy posture	Lowest operational complexity; easiest desktop coupling	Weakest burst capacity and collaboration
Hybrid local-first	Io runtime and scheduler local; metadata/artifacts/analytics optionally remote; cloud used only for burst training or backup	Recommended	Preserves external-trigger coupling while allowing cheap scale-out	Slightly more integration work
Full cloud	Runtime, telemetry, mirror, and analytics in cloud	Team-scale experimentation or heavy continuous compute	Easiest to scale compute and storage	Harder desktop-coupling story; broader privacy/compliance surface

Desktop and builder-context sensor

State scheduler

Waking runtime

Dream engine

Dormant policy

Paused state

Gridworld environment

Replay buffer

Telemetry bus

Counterfactual recombination

Associative perturbation channel

Metadata store

Parquet trace store

Mirror reader

Coherence and admissibility verdicts

Bounded process-control signal



Show code
The recommended stack is deliberately conservative. PyTorch remains the best fit because your inherited substrate is already in the Dreamer/PlaNet family and PyTorch offers straightforward single-node work with a clean DistributedDataParallel path if the model later needs multi-GPU training. FastAPI plus OpenAPI is a good control-plane choice because it keeps orchestration and internal tooling simple. PostgreSQL should hold session metadata and mirror verdicts; Parquet plus DuckDB should hold large dream traces and make offline analysis cheap; MLflow is the best default tracker if you want zero license cost; Weights & Biases is a valid optional commercial layer if you want hosted collaboration; OpenTelemetry plus Prometheus/Grafana covers observability well without vendor lock-in. 

Layer	Recommended choice	Open-source alternative	Commercial option	Rationale
Runtime and model code	PyTorch, Python, Hydra	JAX or pure argparse/YAML stack	Managed notebooks are optional, not required	Fast iteration, broad ecosystem, easy scale-up
API/control plane	FastAPI + OpenAPI	gRPC or Flask	Managed API gateways later if exposed externally	Strong typing, easy internal services
Schema validation	Pydantic + JSON Schema	marshmallow or dataclasses	None needed for MVP	Strong contract discipline for telemetry evolution
Metadata database	PostgreSQL	SQLite for very first prototype	Managed Postgres if cloud-first later	Good for relational metadata, partitioning, replication
Trace and artifact store	Parquet + DuckDB on local/object storage	Polars/Arrow stack	Managed lakehouse only if scale demands it	Cheap, columnar, analysis-friendly
Experiment tracking	MLflow	Plain filesystem plus Git tags	W&B cloud or self-managed	MLflow keeps MVP license-free; W&B is useful if collaboration grows
Observability	OpenTelemetry + Prometheus/Grafana	Plain logs only	Datadog	OTel keeps signals portable; Datadog is optional convenience
Orchestration	Docker Compose locally; Ray only if parallel rollouts become bottleneck	systemd/supervisord	Kubernetes for multi-node or self-managed W&B later	Avoid K8s until justified

FastAPI is explicitly documented as a high-performance Python API framework based on standard type hints; Pydantic can emit JSON Schema; PostgreSQL supports both declarative partitioning and logical replication; DuckDB reads and writes Parquet efficiently; MLflow provides experiment tracking and model registry; OpenTelemetry is a vendor-neutral telemetry framework; Ray scales generic Python code when needed; Docker Compose simplifies multi-container local stacks; Kubernetes Deployments and StatefulSets are useful once stateless and stateful services need independent lifecycle management. 

The integration surface is small but important:

Integration point	Dependency	Why it matters
Desktop/activity sensor	OS event hooks or a tiny local daemon	Needed because wake/dream ratio is externally coupled
Gridworld environment	Existing Probe 1 environment	Supplies replay anchors and waking controls
Replay buffer	Existing telemetry and episode store	Seeds replay-anchored dreams
Mirror ingestion	Existing mirror schemas and data readers	Must remain backward-compatible or versioned
Artifact/checkpoint registry	Filesystem/object storage plus metadata DB	Required for reproducibility and audits
Optional observability backend	OTel collector or Prometheus/Datadog	Helps debug state-transition errors and boundary violations
Optional bounded control IPC	Minimal authenticated local queue or RPC	Must remain process-level only

For commercial tooling, W&B now supports cloud-hosted and self-managed options; its Pro plan starts at $60/month, and its academic Pro program includes 200 GB of cloud storage for qualifying academic users. Datadog Infrastructure Pro starts at $15 per infra host per month when billed annually. Both are useful but optional; neither should drive the architecture. 

Data model and telemetry
The current Phase 0 DreamRollout shape is not sufficient for the stronger version of Probe 3. The uploaded brief says the mirror can already read DREAM_ROLLOUT and its sequence_h, but it also explains why that would fail on pure prior rollout: within-latent coupling from the GRU baseline is not enough. That means the schema has to carry provenance and regime information that lets the mirror distinguish at least four things: replayed content, recombined dream branches, waking planning rollouts, and prior-only controls. Without that, the mirror can only see that “something recurrent happened,” not what kind of something it was. 

Proposed entity or field	Why it should exist
dream_session_id	Groups multiple rollouts under one dream episode
dream_mode	Tells the mirror whether this was replay-only, recombination, associative, or control
anchor_type	Distinguishes replay anchor, current-state planning anchor, random prior anchor, or stitched hybrid
anchor_source_refs	Points to episode IDs and step ranges used as seed material
planning_overlap_flag	Explicitly marks whether the rollout reused planning codepath or a dream-only codepath
policy_coupling_mode	Records whether actor was disabled, weakly conditioned, or active
temperature_or_noise_schedule	Makes associative perturbation reproducible and auditable
gradient_policy	States whether no gradients, world-model-only gradients, or broader updates were enabled
termination_reason	Needed to distinguish mirror idling, quota stop, trigger reversal, or natural completion
checkpoint_hash and rng_seed	Reproducibility core
trace_refs	Stores references to full latent/action traces in Parquet or object storage
mirror_sampled with summary stats	Separates all sessions from those subjected to deeper mirror review

A useful design pattern is relational metadata plus columnar trace blobs. Use PostgreSQL tables for runs, sessions, transitions, verdicts, and control signals; use Parquet objects for full latent trajectories, world-event streams, and sampled dream traces; query the trace lake with DuckDB for ad hoc analysis. This split is well supported by the underlying tools and minimizes the temptation to cram long tensor arrays into relational rows. 

contains

contains

contains

produces

records

contains

yields

contains

evaluated_by

emits

seeds

runs_under

may_emit

RUN

uuid

run_id

string

schema_version

string

git_sha

datetime

started_at

string

deployment_mode

STATE_TRANSITION

EPISODE

DREAM_SESSION

uuid

dream_session_id

uuid

run_id

string

dream_mode

string

anchor_type

string

policy_coupling_mode

string

gradient_policy

string

checkpoint_hash

int

rng_seed

datetime

started_at

datetime

ended_at

string

termination_reason

MODEL_CHECKPOINT

BUILDER_CONTEXT

AGENT_STEP

REPLAY_SEGMENT

DREAM_ROLLOUT

uuid

rollout_id

uuid

dream_session_id

int

sequence_index

int

rollout_length

string

trace_ref

string

noise_schedule

bool

mirror_sampled

MIRROR_EVALUATION

uuid

evaluation_id

uuid

dream_session_id

float

coherence_score

float

repetition_score

float

intelligibility_score

string

verdict

datetime

created_at

WORLD_EVENT

CONTROL_SIGNAL

uuid

signal_id

uuid

evaluation_id

string

signal_type

datetime

applied_at



Show code
The exact cadence of stored dream traces is still unspecified, so the right policy is to separate session metadata from full-trace sampling. A good starting point is: store metadata for every dream session; store full latent/action traces for every session that triggers mirror concern or idling, plus a statistically sampled subset of non-triggering sessions for baselining; always store seeds, checkpoint hashes, anchor references, and mode labels. That gives you months-long storage sustainability while preserving enough ground truth to test whether the mirror is seeing genuine structure rather than hallucinating narratives from sparse summaries. The recommendation is a design choice, not an inherited commitment. 
 

Delivery effort and cost
The shortest credible path is one probe-quality MVP, not a platform rewrite. If the team over-invests in cloud orchestration or multi-user tooling before answering the Probe 3 question, the work will drift away from the project’s stated discipline. A practical build sequence is to freeze the research commitments first, then amend schema, then implement the dream runtime, then integrate mirror-guided idling and evaluation. 

Phase	Main milestone	Rough person-months
Decision freeze	Commit dream taxonomy, trigger/exit policy, gradient policy, and success tests	0.5–1.0
Telemetry amendment	Version schema, add provenance fields, wire metadata store and trace store	0.5–1.0
Dream runtime MVP	Build replay-anchor sampler, recombination engine, and distinct dream codepath	1.5–2.0
Mirror integration	Add bounded control channel, coherence/idling logic, and observability	1.0–1.5
Evaluation and ablations	Run pure-prior, planning-rollout, shuffled-time, and no-gradient controls	1.0–1.5
Hardening and docs	Reproducibility packaging, retention policies, operator docs, decision memo	1.0–1.5
Total MVP		5.5–8.5

A more robust version that adds associative mode, more formal dashboards, cloud burst support, and stronger collaboration hygiene is closer to 8.5–12 person-months. The effort is research-heavy because each stage needs control experiments and interpretation support, not just code. 
 

Jun 07
Jun 14
Jun 21
Jun 28
Jul 05
Jul 12
Jul 19
Jul 26
Aug 02
Aug 09
Aug 16
Aug 23
Decision freeze and acceptance tests
Schema amendment and telemetry plumbing
Replay-anchor dream MVP
Mirror control and idling integration
Ablations and comparative runs
Reproducibility, retention, and docs
Research commitments
Data plane
Runtime
Evaluation
Hardening
Illustrative Probe 3 delivery timeline


Show code
Infrastructure cost is comparatively small. RunPod’s official pricing currently lists L4 at $0.44/hr, A100 PCIe at $1.19/hr, and standard network storage at $0.05–$0.07/GB-month. Datadog Infrastructure Pro starts at $15/host-month billed annually. W&B Pro starts at $60/month and the academic Pro program includes 200 GB of cloud storage for eligible academic users. Those figures make burst experimentation cheap enough that cost should not be the reason to under-instrument the MVP. 

Scenario	Infra assumptions	Estimated monthly infra	Licensing	Staffing estimate
Local-first OSS	Existing workstation; PostgreSQL/DuckDB/MLflow/Prometheus/Grafana local; minimal offsite backup	$20–$100/mo incremental ops; if new hardware is needed, add an illustrative assumption of $6k–$12k capex, or about $170–$330/mo amortized over 36 months	$0 license cost with OSS stack	At an illustrative assumed blended cost of $18k per person-month, MVP staffing is about $99k–$153k
Hybrid recommended	RunPod L4 for 160 GPU-hours/month, 1 TB storage, one monitored host, optional W&B Pro	$135–$215/mo	Optional W&B or Datadog only	Same staffing range
Cloud-heavier research	RunPod A100 for 160–730 GPU-hours/month, 1 TB storage, one monitored host, optional W&B Pro	$255–$1,014/mo	Optional W&B or Datadog only	Same staffing range; likely more engineering toil

Those staffing numbers are assumptions, not market quotations. They are included because the user requested a cost estimate; the only sourced costs above are infrastructure and optional software charges. In practice, staffing will dominate total spend for any serious version of Probe 3. 

Risks, feasibility, and reference implementations
The project is technically feasible, but the biggest failure mode is not implementation failure. It is interpretive overreach: mistaking routine latent dynamics, planning rollouts, or logging artifacts for dream evidence. That is why the MVP needs explicit controls and why the dream mode must be engineered to be meaningfully distinct from planning before the mirror ever sees it. 
 

Risk	Why it matters	Recommended mitigation
Dream/planning confound	Waking actor already imagines futures	Separate codepaths, provenance flags, and planning-control runs
GRU-baseline false positives	Mirror may read ordinary recurrent coupling as dream content	Compare against prior-only and shuffled-time controls
Covert self-optimization	Offline gradients can reintroduce exactly what the stance rejects	Freeze actor during MVP dream learning
Mirror over-interpretation	Natural-language summaries may sound more meaningful than substrate warrants	Keep citations, reproducible trace refs, and quantitative controls attached to every verdict
Schema drift	Mirror readers are already pinned to existing schemas	Version schema once, with explicit compatibility plan
Privacy leakage	Desktop/life-pattern coupling may become personal behavioral data	Minimize and coarse-grain builder-context collection
Runaway offline loops	Dream sessions can become open-ended and expensive	Mirror-driven but content-blind idling plus hard runtime caps
Premature platforming	Moving to full cloud/K8s too early obscures the research question	Keep MVP local or hybrid and narrow in scope

If personal data is ever present, especially if builder-context signals are retained or shared, data minimization and purpose limitation become relevant legal principles under GDPR-like regimes. For the software itself, least privilege, zero-trust segmentation, and standard application-security baselines such as OWASP ASVS are appropriate even for a small research system, because the mirror/runtime boundary is central to the project’s epistemic validity, not merely its operations. 

The strongest representative references for later building are these:

Reference implementation or paper	What it contributes to Probe 3	Primary source
PlaNet	Shows why the inherited substrate mixes deterministic and stochastic latent state for planning from pixels	
Dreamer	Shows how latent imagination can train behavior, and therefore why planning-vs-dream distinction is crucial here	
DreamerV3	Shows the world-model family is robust and scalable across many domains	
Deep Generative Replay	Gives the clearest ML precedent for replay-like offline consolidation without storing all prior data forever	
Wilson and McNaughton	Canonical evidence that waking experience is re-expressed during sleep	
Ego-Stengel and Wilson	Causal evidence that disrupting ripple-associated activity during rest impairs learning	
Hobson and Friston	Predictive-processing account of dreaming as internally generated virtual reality and offline model optimization	
Raichle default-mode work	Suggests that internally generated structured activity during rest is not reducible to noise	
Smallwood and Schooler review	Strong frame for internally generated, task-decoupled thought and why associative modes matter	
Google DeepDream	Useful computational analogy for associative amplification modes	
Lucid-dreaming review literature	Evidence that lucid control is entangled with metacognitive access, making it a weak first fit for self-opacity	

What would be genuinely surprising is if pure prior-only rollout, with no replay anchor, no distinct perturbation regime, and no special provenance, turned out to be enough for the mirror to produce stable, coherent, dream-specific descriptions that also predicted later waking changes. That would run against the uploaded brief’s own Concern A/B analysis and against the basic fact that imagined rollouts are already everyday machinery in the Dreamer family. A second surprising outcome would be if a lucid-control-first variant fit the project better than self-opaque offline processing, because the literature on lucidity points toward explicit metacognitive access rather than the ingredients-only stance imposed here. 
 

Next steps and prioritized backlog
The immediate next step is not coding. It is to produce a short Phase 0 decision memo that freezes the load-bearing commitments this report identifies: the dream taxonomy, the gradient policy, the telemetry amendment, the trigger/exit rules, and the success tests. Until those are fixed, implementation effort will mostly create ambiguity. That maps directly to the open questions in the uploaded brief. 

Priority	Development task	Why it should happen in this order
P0	Freeze the MVP answer to “what is a dream here?”	All downstream implementation depends on it
P0	Decide gradient policy explicitly	Prevents drift into covert self-optimization
P0	Amend and version DreamRollout/session schema	Mirror evidence is impossible without richer provenance
P0	Define success tests and control experiments	Prevents post hoc interpretation
P1	Implement state scheduler for waking/dreaming/dormant/paused	Required runtime backbone
P1	Build replay-anchor sampler and counterfactual brancher	Core dream engine for MVP
P1	Add provenance logging: anchor refs, seeds, checkpoint hashes, policy-coupling mode	Core reproducibility requirement
P1	Implement bounded mirror process-control channel	Needed for quality-based idling
P1	Add observability and session dashboards	Necessary to debug long-running offline periods
P1	Run ablations: prior-only, planning-rollout, shuffled-time, no-gradient, world-model-only-gradient	These are the minimum scientific controls
P2	Add associative/noise-annealed dream channel	Only after baseline interpretation is stable
P2	Evaluate whether parameter-space “sleep” belongs in dormant or later work	Promising, but likely a different question
P3	Add cloud burst support, managed tracking, or broader collaboration tooling if justified	Infrastructure should follow evidence, not precede it

The recommended near-term deliverable after that backlog is a single, narrow MVP: replay-anchored recombination, actor frozen during dreams, world-model-side updates only if explicitly chosen, full provenance telemetry, mirror-readable summaries, and three mandatory controls. If that MVP cannot produce evidence distinguishable from planning and prior-only baseline, the project will have learned something important without overbuilding. If it can, then associative mode, dormant-state work, and richer deployment options become justified next steps. 
 