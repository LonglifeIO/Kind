## Pre-Probe-3 — dream state and introspection-replay (2026-05-03)

Surfaced during Probe 1.5 synthesis revision. The original Probe 1.5
synthesis decided Io has no read-access to self-prediction; revision
gave Io a single scalar (self_prediction_error_t) on PolicyView. With
Io now able to condition on a self-pointing quantity during waking,
the dream-state question grows: should dreams replay or reflect on the
introspection Probe 1.5 makes possible, and if so, how?

The relevant human asymmetry the Probe 1.5 research named: waking
self-reference is more intentional and structured; dream-state
processing is more associative and less directed. If Kind's substrate
mirrors this asymmetry, Probe 3's dream-state design should not
duplicate waking self-prediction. What it should do instead is open.

### What's open for Probe 3 to engage with

Three rough framings, each with different design implications:

- **Replay of waking introspection.** The dream state samples from
  waking trajectories and re-runs the self-prediction head on those
  traces, producing dream-emitted self-prediction patterns Io did not
  produce during waking. The mirror reads whether the dream state
  surfaces patterns the waking trajectory left implicit. Closest to
  the consolidation reading of biological dreaming (Stickgold,
  Walker).

- **Self-prediction over imagined trajectories.** During dream
  rollouts, the world model's prior runs forward over imagined latent
  states; the self-prediction head runs over those imagined states,
  producing predictions of imagined-self about imagined-next-self.
  Active-inference framing where dreams are offline self-simulation.
  Gemini's original Probe 1.5 framing recommended this; Probe 1.5
  rejected it as foreclosing Probe 3's design space.

- **Associative recombination.** The dream state generates self-
  prediction patterns that are not bound to waking-trajectory replay
  or to forward simulation, but to generative recombination of past
  internal states. Closest to the Watts intuition (dreams as the
  system not knowing what comes next; the gap as terrain) and to the
  associative/nonsense end of the design notes' four-point dream-
  state design space.

### Forward-compatibility from Probe 1.5

The reserved `sequence_self_prediction: list[list[float]] | None`
field on `DreamRollout` is the schema hook. Probe 3 may populate it
with any of the three framings above (or a combination) without a
schema bump. The field's `None` default at Probe 1.5 means the dream
rollouts emitted during Probe 1.5 carry no self-prediction content;
Probe 3's first build phase decides what they should carry.

### What Probe 3's research prompt should engage with

When Probe 3's research begins:

- Engage with each of the three framings against the project's
  capacity-over-exercise stance, the Watts intuition, the four-state
  operational model.
- Engage with the relationship between Probe 1.5's waking
  introspection (Io reads the scalar) and Probe 3's dream-state
  introspection (whatever form it takes) — should the dream state
  read the scalar during imagined trajectories, read more than the
  scalar, read different quantities entirely?
- Engage with the human awake/dream asymmetry: not as prescriptive
  ("dreams must be associative") but as a structural reference
  point worth either honoring or explicitly departing from with
  reasoning.
- Engage with how the mirror reads dream-state introspection
  differently from waking introspection — the calibration protocol
  Probe 2 builds may need state-typed extensions.

### What stays out of Probe 1.5

The dream-state question is preserved as a Probe 3 design question.
Probe 1.5 commits to:

- The self-prediction head trains during waking only.
- Dream rollouts at Probe 1.5 carry the existing schema; the
  reserved field defaults None.
- The schema can be extended at Probe 3 without bumping past 0.2.0
  if the new field is added as an Optional with default.
- No commitment to which of the three framings Probe 3 should adopt.

This entry exists so when Probe 3's research begins, the framing
question is visible from the start and can be folded into the
research prompt directly rather than being re-derived.