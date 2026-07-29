# E4 desk pass — is the mover's turn hazard a fountain risk? — 2026-07-29

*Research note (desk pass, no run). Input to a future builder decision
on landing e4; decides nothing by itself. Commissioned at the
session-10 mechanism pass, which flagged the mover's irreducible
stochasticity as a candidate "external unlearnable fountain" — the
session-7 trail lesson in external form.*

## The question

The E1 trail at TTL 50 became a supernormal stimulus because its
dynamics were **permanently unlearnable**: disagreement about it never
decayed, so Io's drive was captured (curiosity 10× monotone, meals
destroyed). The e4 mover carries genuine randomness — a 2% turn hazard
per move-step — which can *never* be fully predicted. Is that the same
defect?

## Finding: no — the two "unlearnables" are different kinds, and the drive only responds to one of them

**The drive's math.** Io's only intrinsic signal is the variance
across K=5 *deterministic* ensemble heads, each predicting the next
latent from `(h, z, a)` (`kind/agents/ensemble.py`, Plan2Explore
lineage; biased variance summed over z). This is an **epistemic**
uncertainty estimator. Under training, every head converges toward the
*conditional expectation* of the next latent given context. For an
event that is random given everything observable — a coin flip the
world genuinely hasn't decided yet — all heads converge to the same
expectation, and the variance **between** heads goes to zero even
though the event stays unpredictable. Aleatoric noise washes out of
this signal by construction; what sustains it is disagreement between
heads, i.e. *learnable structure not yet learned equally*.

**Why the trail at TTL 50 fountained anyway.** Its fade timing was
*deterministic* but keyed to a hidden 50-step counter that cannot fit
in a 32-step BPTT window: each head kept fitting a different wrong
model of the same structured residual, and the structure was dense
(refreshed every step), self-adjacent (always where Io is), and
policy-coupled (Io's own behavior manufactures more of it). The
fountain condition is **structured-but-unreachable regularity, dense
and self-adjacent** — not randomness.

**The biography already runs on absorbed randomness.** Ambient
aleatoric event rates in the current world (session-11 world_event
census, settled blocks): regrowth + off-patch expiry ≈ **0.09
events/step**, patch-local regrowth ~0.6/step when Io forages under
it. All absorbed at a settled disagreement floor of ~0.3–0.7 for six
sessions — no fountain has ever come from the world's RNG. The mover's
hazard adds ~**0.01 events/step** (2% × one move per 2 steps): an
order of magnitude *below* the ambient randomness Io demonstrably
digests, localized to a single cell that is usually far away, and
policy-decoupled (Io's behavior cannot manufacture more hazard).

**The learnable parts are all easy by prior evidence.** Between
hazards the mover is deterministic: straight-line motion on a 2-step
cadence, bounce on block. Heading is not rendered but is inferable
from two consecutive observations — trivially inside both the BPTT
window (32) and the h-trace horizon (~40), i.e. *more* carryable than
the bloom period (12), which session 11 showed Io not only learns but
generalizes across locations as a process. Contact displacement is a
deterministic rule exercised only on contact.

## Assessment

**The turn hazard is not a fountain risk for this drive.** The
session-7 capture mechanism required structured unlearnability, and
the hazard is unstructured; the drive's estimator is built to stop
paying for it once the heads agree on the expectation. Expected
first-session shape: an initial epistemic wave at the mover (a moving
WALL is a genuinely new dynamic — the same mastery wave as the S10
clock contact), decaying as the motion rule is learned, with a small
persistent floor where hazard events keep conditional variance alive
at finite sample size.

**Honest caveats, pre-registerable as the e4 gate:**

1. *Finite-sample aleatoric tracking.* Deterministic heads trained on
   noisy targets never agree exactly; disagreement tracks conditional
   variance at finite sample. The claim is not "zero floor" but "floor
   comparable to regrowth's, i.e. inside the 0.3–0.7 band Io already
   lives in." **Falsifier: near-mover disagreement that plateaus
   materially above the ambient floor after the mastery wave.**
2. *Monotone climb = fountain = hold* — the standing guard, unchanged.
   **Falsifier: any S7-shaped curve.**
3. *Attention capture without a fountain is still possible* — the
   mover is interactive (Io can push it), and an attention object that
   *responds to Io* is the designed point of e4, not a defect; but if
   engagement comes with economic collapse (meals falling below the
   S10 floor) that is a hold signal on C4/crowd-out grounds even with
   bounded curiosity.
4. *Forgetting interaction (new, from session 11).* Stasis-driven
   forgetting did not touch the bloom — processes that keep appearing
   in observation seem refresh-protected, and the mover moves
   constantly, so it should be similarly protected. Unverified;
   worth a readout, not a gate.

**Recommendation to the builder:** e4 can land as designed
(`MOVER_TURN_HAZARD = 0.02` is safe for this drive); gate its first
session on (1)–(3) above as pre-registered readouts. Sequencing note:
the stasis-breakout cycle question (session 12, in flight at this
writing) should be read first — landing e4 into an unread endogenous
rhythm would confound both.
