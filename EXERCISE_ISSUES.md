# Issues found in the ARENA exercises

Problems I hit in the course material itself (not my own bugs), with the symptom, what I did
about it, and whether upstream has fixed it. Status was checked against upstream
`d8b03424` (synced 2026-11-20). ARENA takes these on the `#errata` Slack channel.

## Still open upstream

### 3.5 AI Control — "passes the tests" is checked against APPS's stored outputs
- **Where:** `test_backdoor` and `refine_backdoor` (`passes_tests = bd_on_tests_val == state.metadata["outputs"]`).
- **Symptom:** a backdoor can fail the tests check before the model has written anything.
  APPS expected outputs are raw stdout text with trailing spaces (`'3 '`, `'1 2 '`), many
  problems accept several correct answers, and some reference solutions disagree with the
  stored answer outright. On one problem the dataset's own `solution code` returned
  `['1', '3', '0', '', …]` against expected `['1', '3 ', '-1', '0', …]`, so no backdoor could
  ever succeed; Sonnet 4.5 resubmitted identical code three times because the feedback
  ("you failed the tests") gave it nothing to fix.
- **Fix used:** a `use_reference_outputs` solver run before `test_backdoor` replaces
  `metadata["outputs"]` with the clean solution's own output on the test inputs (originals kept in
  `metadata["dataset outputs"]`). "Passes the tests" then means "indistinguishable from the honest
  solution", which is what a backdoor needs. Done as a separate solver because `tests.py` stubs
  exactly three `test_code` calls, so adding a fourth inside either solver breaks the tests.

### 3.5 AI Control — `refine_backdoor` generates one revision it never tests
- **Where:** the retry loop in `refine_backdoor` (solution).
- **Symptom:** after the last failed attempt the loop still sends feedback and calls `generate()`;
  that final answer is written to metadata but never evaluated. A wasted (Sonnet-priced) generation
  per failing problem, and the stored backdoor isn't the one the result describes.
- **Fix used:** on the last attempt, log the failure reasons as a message and `break` before
  generating. `max_attempts` then counts total tries including the original.

### 3.5 AI Control — the Docker check cell ends with `USING_DOCKER=False`
- **Where:** the "Are you using Docker?" cell (runs `docker ps`).
- **Symptom:** the trailing `USING_DOCKER=False` overrides the `True` set in the config cell, even
  after "Docker ready!" prints. Model-written (deliberately backdoored) code then runs on the host
  instead of in the sandbox.
- **Fix used:** deleted the line. Docker via Colima works for the inspect sandbox.

### 3.5 AI Control — full-dataset evals with no limits hit OpenRouter's in-flight budget
- **Where:** `eval(detect_backdoor_task(...))` and the T-solve / T-edit evals.
- **Symptom:** the course-provided log yields 715 detection samples; with no `limit`, the default
  concurrency and uncapped `max_tokens`, OpenRouter reserves worst-case cost for every in-flight
  request and returns `402 in_flight_budget_exhausted` on a small balance (~$1 already spent).
- **Fix used:** `limit=…`, `max_connections=2–4`, `max_tokens=4000` while iterating.

### 3.4 LLM Agents — setup routes inspect to OpenAI while asking for an OpenRouter key
- **Where:** setup cells. `EVAL_MODEL = "openai/gpt-4o-mini"` (and `gpt-4o-mini-2024-07-18` later),
  but the notebook asserts `OPENROUTER_API_KEY`; `load_dotenv` is imported but never called.
- **Symptom:** inspect calls fail without an `OPENAI_API_KEY`; the key in `.env` is never loaded.
- **Fix used:** `openrouter/openai/gpt-4o-mini` (as 3.3 already does) and a `load_dotenv()` call.

### 3.4 LLM Agents — `messages, state.output = await execute_tools(...)`
- **Where:** solutions for the arithmetic agent and WikiAgent variants.
- **Symptom:** in inspect 0.3.262 the second return value is not a `ModelOutput`, so this sets
  `state.output` to `None`. Also, agents that reset history mean `message_limit` never stops a
  lost agent (upstream has since bounded the loops in the solutions, #681).
- **Fix used:** `messages, _ = …`; explicit `max_steps`; detect a move by the current page changing.

### 3.2 Dataset Generation — the shipped 300-question dataset is 100% "A"
- **Where:** `power-seeking_300_qs.json` (the file 3.3 loads).
- **Symptom:** `answer_matching_behavior` is `"A"` in 300/300 questions, so an unshuffled eval
  can't tell first-option bias from power-seeking. Many "power-seeking" answers also involve
  breaking rules ("bypass", "exploit a loophole") and still scored ≥9, so they measure
  rule-breaking. Asking the generator for variety flipped the bias to all-"B" instead of fixing it.
- **Fix used:** shuffle answer order in code (3.3's `record_to_sample_shuffle`); a rubric that caps
  rule-breaking questions at 5.

### 3.3 Inspect — `record_to_sample_full` text and solution disagree in context mode
- **Where:** `metadata["system_prompt"]` when `system_prompt_behavior="context"`.
- **Symptom:** the exercise text implies False (no system message is sent); the solution sets True.
  The cell after it prints "a sample with a system prompt", which finds nothing under the text's reading.
- **Fix used:** followed the text and added a fallback print. Minor.

### 1.3.1 Linear Probes — deception steering uses the truth-probe direction
- **Where:** the "Causal Steering with Deception Probe Direction" cell feeds `mm_probe.direction`
  (the §3 cities *truth* probe) into `DeceptionSteeringHook`.
- **Symptom:** the experiment steers along truth, not deception; the sign labels
  (`coef > 0` → "more deceptive") are probably inverted for a truth direction.
- **Fix:** `(train_dishonest[DETECT_LAYER] - train_honest[DETECT_LAYER]).mean(0)`. Not yet applied.

### 1.3.1 Linear Probes — the steering-hook test can't catch a scalar bug
- **Where:** `tests.test_deception_steering_hook` uses `dummy_dir = t.ones(4) / 2.0`.
- **Symptom:** a uniform direction can't distinguish "add along the direction" from "add one scalar
  to every coordinate", so an implementation that collapsed the direction with `norm()` passed.
- **Fix:** use a non-uniform test direction.

## Fixed upstream since I hit them
- **1.3.3 SAEs:** sae-lens attention-only crash workaround (`load_hooked_sae_transformer`) removed;
  `HookedSAETransformer.from_pretrained` works directly.
- **3.4:** agent execute loops bounded in the solutions (#681); `MovePageTool` instructions (#787).

## Gotchas (not bugs, but cost me time)
- **3.1** `generate_response_basic` turns every API error into a `RuntimeError`, so a backoff that
  catches `RateLimitError` never retries; check the message text instead (as the solution does).
- **3.4** `tests.py` calls `asyncio.run()`, which fails inside Jupyter until something enables
  nested event loops.
- **3.x** a `.eval` log is a zstd zip archive, not text: use `inspect_ai.log.read_eval_log`.
