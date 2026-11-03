# mmaas.ai: makemore as a service

A free site where you paste a list of names and get new names in the same style. It's a
base model trained on names from many languages, fine-tuned on the fly to your list.

Written 2026-11-01. Costs and speeds are estimates, extrapolated from the local makemore
sweeps. Check current prices before buying anything.

## Goals

- Put it up and forget about it.
- No accounts, no billing page, no database.
- A fixed monthly bill. If traffic gets huge, it should fail gracefully: the site stays
  up, and people past capacity get a clear message right away. Scaling up gets solved
  then.
- Free at launch. Add ads only if real traffic shows up.

## Where things stand

- Best model so far: dev loss 1.9412, using GELU, hidden 800, dropout 0.2, weight decay
  0.1, lr 3e-4. That's one seed, on English-style names only.
- GELU's train/dev gap is 0.25, so it's overfitting and may want more dropout.
- `round3` in `makemore_experiment.py` tests that: lr {2e-4, 3e-4, 5e-4} x dropout
  {0.2, 0.3, 0.4}.
- The val split is still untouched. Keep it that way until a final config is chosen.
- Sweeps don't save weights yet.

## Design

**Base model**

| | Current | Multi-language target |
|---|---|---|
| Characters | 27 | about 300 |
| Embedding size | 24 | 32 |
| Hidden size | 800 | 800-1024 |
| Parameters | 2.6M | 3-5M |
| Size on disk | 10.5 MB | 12-20 MB |

- Add a small learned language embedding, or the model will blend languages together.
- Leave out Chinese and Japanese at first. They have thousands of characters and very
  short names, so this design doesn't suit them.

**Fine-tuning**

- Freeze the base model and train only the last layers.
- Compute the frozen layers' output once, then train on those saved values.
- Target: a few seconds on a CPU. The full-model fallback is about 30s.
- Generate a few sample names every couple of seconds so users watch them drift toward
  their style.
- Keep tuned weights in memory only, and delete them once names are generated.

**Serving**

- One Python web process, on CPU only. No GPU.
- Cap each request at about 500 names, plus a simple per-IP rate limit.
- Store nothing that users paste in.

**Overload behavior**

The rule: every request gets a fast answer, even if that answer is "not right now".
Work that can't start soon is refused, never left to pile up.

- **Keep the web process free.** Training runs in separate worker processes, never in
  the web process, so a full server can still answer instantly.
- **Fixed capacity.** 1-2 workers at a time, loaded once at startup, so memory use stays
  flat under any amount of traffic.
- **Short, capped queue.** About 10 waiting jobs, each showing its place in line. When
  the queue is full, return `503` with `Retry-After` straight away, and show a friendly
  "we're busy" message.
- **Fall back to the base model.** When fine-tuning is full, still offer names from the
  untuned base model. That's cheap enough to keep working long after fine-tuning fills
  up, so most people still get something.
- **Time limits.** Kill any job after about 60s, and drop jobs from the queue after about
  30s of waiting.
- **Cancel abandoned jobs.** If someone closes the tab, stop their job, so that capacity
  isn't spent on nobody.
- **Serve the page separately.** Caddy serves the page itself, so it still loads if the
  app is struggling. If the app is down, Caddy shows a static "back soon" page instead of
  a timeout.
- **Recover without help.** `systemd` restarts anything that crashes, and sets a memory
  limit so a runaway process dies alone instead of taking the Droplet with it.
- **Optional:** put Cloudflare's free tier in front, to absorb traffic to the static page
  and block abusive bots.

## Costs

| Item | Estimate |
|---|---|
| One fine-tune | about $0.0005 of compute |
| Generating 100 names | about $0.00001 of compute |
| Base model training | free on the Mac, or $2-5 on a rented GPU |
| Tuning the base model | free over a few days, or $50-200 on a rented GPU |
| DigitalOcean Droplet | about $24/month for 2 vCPU / 4 GB, about $48/month for 4 vCPU / 8 GB |
| `.ai` domain | often around $70+ with a multi-year minimum |

Start on the 2 vCPU / 4 GB Droplet. It's fixed-price apart from bandwidth overage, which
should be negligible for a site this small. Resizing later is easy.

## Steps

### 1. Prove the idea, before building anything

This is the go/no-go step. It uses the English model and needs no new data.

1. Run `round3` and pick a config.
2. Add weight saving to `run_config`.
3. Train the base model and save it.
4. Write the frozen fine-tune as a script.
5. Test it on small, distinctive lists of 50-500 names, such as Italian surnames or
   fantasy names. Hold out part of each list.
6. Measure:
   - Does the tuned model score better than the base model on the held-out names?
   - What share of generated names are new, rather than copies of the input list?
   - Do the names look right to a person?

**Stop here if the output is mostly copies, or no better than the base model.** A site
can't fix that.

### 2. Build the multi-language dataset

1. Collect names from clean sources, such as census records and Wikidata. Check each
   license, and avoid datasets that come from leaks.
2. Aim for about 1M names, tagged by language.
3. Normalize Unicode, remove duplicates, and balance languages so English doesn't dominate.
4. Choose the launch languages and the final character set.

### 3. Train the base model

1. Add the language embedding and the larger vocabulary to `create_model`.
2. Run a short sweep in a new `rounds` entry. Expect about 12 hours per full run on one
   core, or rent a GPU for a few dollars.
3. Choose the final config on dev loss.
4. Evaluate val once, on the final model only.
5. Export the weights and character tables.

### 4. Build the app

1. Make a small Python web app (FastAPI) with a single page: paste names, watch samples
   appear, copy the results.
2. Add the worker processes, capped queue, time limits, base-model fallback, request cap
   and rate limit from "Overload behavior".
3. Keep tuned weights out of disk and logs.

### 5. Deploy on DigitalOcean

1. Create the Droplet and lock it down: SSH keys only, firewall on ports 22, 80 and 443.
2. Install the CPU-only build of PyTorch.
3. Upload the base model with the app.
4. Run the app under `systemd`, with restart on failure.
5. Put Caddy in front for automatic HTTPS.
6. Buy the domain and point it at the Droplet. Check that `mmaas.ai` is available first.
7. Turn on DigitalOcean's free CPU and memory alerts, and enable backups or take a snapshot.
8. Load test the live Droplet at about 10x its capacity. It only passes if:
   - The page keeps loading quickly.
   - Requests over capacity get the busy message in under about a second, with no timeouts.
   - The base-model fallback keeps working.
   - Memory stays flat.
   - When the test stops, it recovers on its own with no restart.

### 6. Launch

1. Add a short privacy note saying that pasted names aren't stored.
2. Share it somewhere small first, and watch it for a day.
3. Then leave it alone.

## Later, only if people use it

- Ads, which bring a cookie consent banner for EU visitors. AdSense approval isn't
  guaranteed, and it pays out only at $100.
- A larger Droplet, or a job queue.
- Chinese and Japanese support.

## Open questions

- Does fine-tuning on a small list actually work? Step 1 answers this.
- Why would someone use this over asking ChatGPT? The best answer is probably "trained on
  your own list".
- Which languages make the launch list?
