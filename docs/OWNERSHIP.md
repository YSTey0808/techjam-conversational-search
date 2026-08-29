# Who owns what

The agent answers a shopper writing in plain English — *"something waterproof
for hiking, nothing over fifty bucks"* — and returns up to ten products plus
one follow-up question. Each turn passes through five modules. You own one.

How the whole thing fits together: [ARCHITECTURE.md](ARCHITECTURE.md).

## The two rules

1. **Your module never imports another owner's module.** Only `schema` (the
   shared data types) and `preprocessing` (the catalog index). This is what
   lets you rewrite the inside of your file without breaking anyone.
2. **Your module exposes exactly one function.** Everything else in your file
   starts with `_` and is yours to add, rename or delete freely.

`agent.py` calls the five in order and does nothing else — no branching, no
fallbacks. If something needs a decision, it belongs in the module that owns
it, not in the wiring.

```python
ex     = extract.extract(user_message, turn, state)       # A: sentence -> requirements
state  = state.update(state, ex)                          # B: remember them
pool   = retrieve.retrieve(prep, state)                   # C: find candidates
policy = ask.decide(prep, state, pool, turn)              # E: what to ask next
items  = rank.rank(prep, state, pool, policy.list_width)  # D: order them
```

| owner | file | your one function |
|---|---|---|
| A | `starter/extract.py` | `extract(message, turn, state) -> Extraction` |
| B | `starter/state.py` | `update(state, extraction) -> SessionState` |
| C | `starter/retrieve.py` | `retrieve(prep, state) -> list[str]` |
| D | `starter/rank.py` | `rank(prep, state, pool, top_k) -> list[str]` |
| E | `starter/ask.py` | `decide(prep, state, pool, turn) -> TurnPolicy` |

Shared, no owner: `schema.py`, `preprocessing.py`. Say something in the team
channel before changing either. `baseline.py` is the old reference agent —
don't develop there.

## Trying your changes

There is no unit-test suite. To see the agent actually run, drive it directly:

```python
from starter.agent import Agent

agent = Agent("data/catalog.jsonl")          # ~8s to index, once
agent.reset("s1", {})
print(agent.respond("s1", "something waterproof for hiking under $60", 1, 10))
print(agent.respond("s1", "actually make it leather", 2, 10))
```

`data/catalog.jsonl` is a 50k-row GitHub Release download and is not in git —
grab it yourself. Check the whole conversation state with
`agent._sessions["s1"].constraints`; that is usually where a bug shows up
first.

---

## Owner A — `extract.py`

**Turns one English sentence into a list of requirements.**

```
extract(message: str, turn: int, state: SessionState) -> Extraction
```

Out: `Extraction(constraints, intent, override, no_preference, usage)` where
`intent` is `"buying"` or `"browsing"` and `override` means the customer just
retracted something.

**What it does now.** Sends the message to an LLM and asks for strict JSON.
The interesting part is the **vocabulary bridge**: a shopper says
"waterproof", the catalog says "Water Resistant". So the model is asked for
`variants` — phrasings a product page might use — and each is checked against
the real index. The first one that actually exists becomes `Constraint.key`,
which is what retrieval searches for. If nothing matches, `key` stays empty;
the constraint is still kept, because another route may still use it.

If the LLM is slow or unreachable there is a **2-second timeout, one retry,
then a rule-based fallback** that reads colours, materials, budgets, sizes and
use-cases directly. `extract()` never raises and never comes back empty-handed
for a non-empty message.

With no provider configured it uses `NullClient`, which always fails — **so
the fallback is what runs by default**, and it is what everyone else is
building on until a model is wired up. Set these for a real model:

```
TECHJAM_LLM_PROVIDER=ollama          # or: openai
TECHJAM_LLM_MODEL=llama3.1
TECHJAM_LLM_BASE_URL=http://localhost:11434
TECHJAM_LLM_API_KEY=...              # openai-compatible only
```

No keys in the repo, ever.

**Start here.** Three real bugs, all in the fallback, roughly in value order:

1. **Negation is ignored.** *"nothing formal"* produces a `formal`
   constraint — the exact opposite of what was said.
2. **Cue words leak in.** *"actually forget that"* currently yields `actually`
   and `forget` as feature constraints. They should be stripped before the
   leftover-token pass.
3. **Comparatives are flat.** *"leather rather than canvas"* treats both words
   the same.

Then: getting a real model connected is the single biggest upgrade available
to anyone on this team, because every other module is downstream of it.

---

## Owner B — `state.py`

**Remembers what the customer has said so far.**

```
update(state: SessionState, extraction: Extraction) -> SessionState
```

Mutates and returns the same state object.

**What it does now.** Three operations. **Wipe**: they retracted something, so
older constraints on the same attribute are dropped. **Replace**: same
attribute, new value — "black" then "white" — so swap, because a person has
one colour in mind at a time. **Append**: anything genuinely new. Attributes
the customer declined go into `dead_attributes` so Owner E never asks again.

It also guesses the shopping category from the words used, since a free-form
message never announces one.

`user_profile` is **soft signal only**. Never turn it into a constraint and
never filter on it — a profile-based filter starts throwing away correct
answers for nothing.

**Start here.** The wipe is too narrow. *"actually forget that, show me silk"*
only clears constraints matching the *new* attribute, so an earlier
"waterproof for hiking" survives and keeps dragging the old answer along. A
broad retraction should probably clear far more. Also check `_SINGLE_VALUED`:
it lists colour, size, budget and material — but can someone reasonably want
cotton *and* linen?

---

## Owner C — `retrieve.py`

**Finds the candidate products.**

```
retrieve(prep, state) -> list[str]        # parent_asins, roughly 200
```

**What it does now.** Three routes, each returning `{asin: score}`, fused by
reciprocal rank. Routes return scores rather than lists so a fourth can be
added without touching anything else.

- `_keyword` — the primary route. Intersects the hard constraints,
  most-informative-first, so the candidate set shrinks fastest. If the
  intersection comes back empty it backs off: drop the least informative
  constraint, then widen to the broader index, and only then give up the
  category gate. A constraint we can't find in the catalog is **skipped, not
  intersected** — an unfindable string is our indexing gap, not proof the
  product doesn't exist, and intersecting on nothing would wipe the pool.
- `_category` — the soft gate. Category comes from a lossy field (1,136
  products are labelled 'Shoes & Jewelry Westlake' because a shop name leaked
  into the category path), so it only ever *adds* weight, never removes a
  candidate. Weighted by how specific the bucket is.
- `_vector` — a stub returning `{}`. See below.

**It never returns an empty list.** An empty list is a guaranteed miss; a
mediocre list costs nothing but one turn.

**Start here.** The route weights in `_WEIGHTS` are guesses — that is the
cheapest thing to improve. `_vector` is intentionally unbuilt: it's for
wording that resolves to no catalog string at all, but if Owner A's variants
work well it may never fire. Prove it's needed before adding an embedding
model and a dependency.

---

## Owner D — `rank.py`

**Puts the best product first.**

```
rank(prep, state, pool, top_k) -> list[str]
```

**What it does now.**

```
score = sum(weight x idf) over matched constraints
      + POP_W x popularity
      - budget contradiction penalty
```

`idf` means a rare match counts far more than a common one — "Imported"
appears on about 13,900 products and tells you almost nothing, while a
distinctive phrase can identify a single item. Popularity is
`log1p(rating_number)`, compressed so one 400k-review product can't dominate.
The penalty only applies to products with a **known** price above a stated
budget: 79% of the catalog has no price, and unknown is not a violation.

**Score the whole pool, then cut.** Never rank an already-shortened list.

**Start here.** `POP_W` is 0.35 by feel, which isn't good enough. Popularity
is a prior about what people buy; if it doesn't match real shoppers it costs
more than it gains, so test it on cases where the right answer is an obscure
product, not just easy ones. Second: the contradiction penalty only knows
about budget. Colour and material mismatches currently cost nothing.

---

## Owner E — `ask.py`

**Decides the follow-up question and what we say back.**

```
decide(prep, state, pool, turn) -> TurnPolicy(ask_attribute, list_width, message)
```

**What it does now.** Two good constraints is roughly what it takes to narrow
the candidates hard, so the goal is getting there in as few turns as possible.
It picks the attribute that best **splits** the current pool —
`coverage x impurity`, so an attribute every candidate shares scores zero
because the answer would tell us nothing. If nothing splits usefully it asks
nothing and lets the customer lead.

Never ask "brand" or "category": category is already inferred from what they
said, and brand isn't something our index can act on.

`message` must always be a plain string — the harness throws away the entire
turn if it isn't.

**Start here.** The `_message` templates are robotic ("What material are you
after?"). There's a marked seam for generating them with an LLM from the
conversation so far, which would make the agent far less mechanical and is
low-risk — a clumsy sentence costs nothing as long as it stays a string. Also
note `list_width` is hardcoded to 10 here and can't see the harness's `top_k`;
this is the only place that decides how many products we return.
