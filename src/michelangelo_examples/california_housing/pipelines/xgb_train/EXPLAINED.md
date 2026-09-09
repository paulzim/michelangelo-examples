# California Housing XGBoost, Explained

A plain-English walkthrough of what the [`xgb_train`](README.md) pipeline
does and why, for readers who are new to Michelangelo, XGBoost, or
distributed ML pipelines in general. For the technical reference — how to
run it, environment variables, task decorators — see this pipeline's own
[`README.md`](README.md).

## The one-sentence version

It's a worked example that teaches a model to guess a house's price from a
few basic facts about it (location, age, number of rooms, etc.), and shows
how to run that training job on Michelangelo.

## The problem it's solving

The data is the "California Housing" dataset — a well-known, small, public
dataset of 20,640 California homes from the 1990 census. Each row has
things like: how many rooms, how old the building is, median income in
that area, latitude/longitude, and the actual sale price. The "target"
being predicted is that price.

This dataset isn't interesting on its own — it's a stand-in. Any real
Michelangelo pipeline for, say, predicting ETAs or fraud risk has the same
basic shape (some input columns, one number to predict), so this is a
small, fast way to see the *plumbing* of a Michelangelo pipeline before
pointing the same pattern at a real, larger dataset.

## What XGBoost is, without the math

XGBoost is a popular algorithm for exactly this kind of "predict a number
from some columns" problem. Loosely: it builds a sequence of simple
decision trees ("is median income above $50k? is the house older than 30
years?"), where each new tree focuses on correcting the mistakes of the
ones before it. Combine a few hundred of these small, individually-weak
trees and you get one strong predictor. It's fast, hard to beat on
spreadsheet-shaped data, and widely used in industry.

## The pipeline, like a recipe

The whole job is broken into four steps that run one after another, each
one handing its output to the next:

1. **`feature_prep`** — fetch the raw housing data and split it into a
   "training" pile (used to teach the model) and a "validation" pile (kept
   aside to honestly check how well it learned, since testing on data the
   model already saw would be cheating).
2. **`preprocess`** — clean/reshape that data into the exact numeric format
   the training step expects. Runs on Spark, the same big-data processing
   engine commonly used for large real-world datasets.
3. **`train`** — the actual learning step. Feeds the training pile to
   XGBoost and produces a trained model. Runs on Ray, a framework for
   spreading a computation across multiple machines instead of one — here
   it's mostly there to prove the *distributed* code path works, since a
   dataset this small could train on one laptop core in seconds.
4. **`push_step`** — takes the finished model plus a short report card of
   how accurate it was, and files them away in Michelangelo's model
   registry, a catalog other services/people can browse to find and deploy
   trained models.

## Under the hood: what each step is actually doing

Each of the four steps runs as its own isolated unit of work — the
platform spins up dedicated compute just for that one step (a small Ray
cluster, or a small Spark cluster), runs it, and tears it back down. Steps
never share a running process or in-memory data with each other; instead,
each one writes its output data to shared storage (an S3-compatible object
store, e.g. MinIO or AWS S3) and passes along a lightweight *handle*
(`DatasetVariable`) that just remembers where that data lives. The next
step reads the handle and pulls the real data back down itself. This is
why Ray and Spark — two totally different distributed-computing engines —
can hand work back and forth to each other without ever needing to talk to
each other directly.

### `feature_prep`

- Runs on a small dedicated Ray cluster (this pipeline asks for 2 CPUs on
  the head node plus 1 worker).
- Looks for a CSV file bundled inside the installed Python package first;
  only reaches out over the network (via scikit-learn's downloader) if
  that file is missing. This is a deliberate fallback for environments
  with restricted internet access.
- Loads the CSV into an ordinary pandas DataFrame, then converts it with
  `ray.data.from_pandas(...)` into a *Ray Dataset* — a table that Ray
  knows how to split into chunks and spread across many machines. At
  ~20,000 rows this dataset is far too small to need that, but it's the
  same code path a multi-terabyte dataset would use, which is the point.
- Keeps only the 8 feature columns plus the price column, and randomly
  splits off 25% of rows as a validation set that the model will never be
  trained on — so the accuracy reported at the end reflects genuine
  generalization, not memorization.
- Never returns the actual data to the workflow engine. Each dataset is
  wrapped in a `DatasetVariable` and written out to storage; only the
  lightweight handle is passed forward.

### `preprocess`

- Runs on a separate, dedicated Spark cluster (1 driver, 1 executor by
  default here) — a different engine from the previous step, started
  fresh and unaware anything ran before it.
- First thing it does is read back the exact data `feature_prep` just
  wrote, except now as a *Spark* DataFrame instead of a *Ray* Dataset —
  same underlying files in storage, read through a different distributed
  engine's API.
- The transformation itself is intentionally simple for this tutorial:
  cast every selected column to `float` (Spark's `.cast("float")`). A
  production pipeline built from this template would do real feature
  engineering here (bucketing, joins, normalization); this step exists
  mainly to prove that plug-in point works.
- Writes the result back out the same handle-based way, producing a fresh
  pair of train/validation `DatasetVariable`s for the next step to pick up.

### `train`

- Runs on its own Ray cluster again, this time sized for actual training:
  1 head node plus 2 worker processes.
- Pulls the Spark-processed data back into Ray's world, and defensively
  drops any leftover `uuid`/`datestr` columns so they're never accidentally
  treated as model inputs.
- Before training starts, it asks Ray how much CPU the cluster it's
  actually running on has right now (`ray.cluster_resources()`), sets
  aside half of that for Ray's own data-shuffling overhead, and divides
  the rest to decide how many training workers it can afford. This is
  what lets the same code run unmodified on a small sandbox cluster or a
  much larger production cluster, rather than a hardcoded worker count.
- The actual learning happens in a small function Ray copies out and runs
  once per worker, in parallel:
  - Each worker is handed only its own slice of the training rows (never
    the full dataset).
  - That slice is converted to an `xgboost.DMatrix` — XGBoost's own
    specialized data format, pre-sorted and pre-bucketed to make building
    decision trees fast.
  - `xgboost.train(...)` runs with this pipeline's chosen settings (10
    rounds of boosting, trees capped at depth 5). Ray's `XGBoostTrainer`
    keeps all the workers coordinated so they jointly build *one*
    consistent model — each round, workers exchange summary statistics
    about their slice of the data so every worker's trees reflect the
    whole dataset, not just their own shard. That coordination step is
    what "distributed training" actually refers to here.
  - After every round, a callback reports the current accuracy numbers
    and saves a checkpoint back to the main training process.
- Once every worker finishes, the step gets back a result containing
  where the final model file was saved (an XGBoost-native `model.ubj`
  file) and the final round's metrics — including a validation RMSE (root
  mean squared error, roughly "how far off were the guesses, on average,
  measured in the same units as price") that's a good number to watch as
  a sanity check on the model's accuracy.

### `push_step`

- Runs on Spark one more time — mainly for convenience (file I/O, talking
  to other services), not because anything here needs distributed
  processing.
- The fiddliest part: Ray writes its checkpoint path without a storage
  prefix (e.g. `default/ray_results/run-...` instead of
  `s3://default/ray_results/run-...`) — it expects the caller to already
  know which storage system it's on. This step re-attaches the correct
  prefix, then uses `fsspec` (a library that gives one common file API
  whether the data's on local disk or in S3) to find the model file under
  that path and download it locally.
- Also reloads the same train/validation data one more time as plain
  pandas tables, so exact copies of what the model was trained on get
  archived alongside it — handy later for debugging an odd result.
- Picks where things get saved based on whether this is a real run or a
  local/CI one: remote object storage if configured, otherwise plain
  local files.
- Connects to Michelangelo's real model registry over gRPC if it's been
  told an endpoint to use — or falls back to a fake, non-persistent
  in-memory registry if not, which is only useful for local unit tests.
- Bundles everything — the model (tagged as an XGBoost regression model),
  a formal evaluation report of its accuracy metrics, and the two
  datasets — into one request, and hands it to a shared "push" helper the
  platform provides, which does the actual uploading and registry
  bookkeeping and reports back success or failure for each of the four
  items individually.

## What Michelangelo adds on top

Michelangelo is a platform for training and serving ML models. Instead of
everyone hand-rolling their own scripts to spin up compute, track
experiments, and store models, teams describe their pipeline once (the
four steps above) and the platform handles: renting the right amount of
compute for each step, running steps in the right order, retrying
failures, and keeping a permanent, searchable record of every model
that's ever been trained. This repo (`michelangelo-examples`) is a small
public collection of "here's how you'd actually write one of these"
reference pipelines — `xgb_train` is one of them.
