# Saile et al. 2024 — Read-only Survey

**Paper:** "Client-Side Adaptation to Concept Drift in Federated Learning" (FLTA 2024)
**Repo path:** `../concept-drift-adaption-saile/` (sibling of `fl_experiment/`)
**Survey date:** 2026-06-04
**Survey type:** Read-only. No code modified, no installs, no runs.

---

## File tree (summary, excluding `.git`, `__pycache__`)

```
concept-drift-adaption-saile/
├── README.md
├── LICENSE
├── main.py                              ← entry point (single file, argparse-based)
├── requirements.txt                      ← torch 1.12.0, torchvision 0.13.0, tb 2.12.1, ...
├── scripts/
│   ├── cifar10_scripts/  (6 shell scripts: base + sudden+adapt + inc+adapt)
│   ├── fminst_scripts/   (6 shell scripts)
│   └── minst_scripts/    (6 shell scripts)
├── src/
│   ├── algorithm/  (basealgorithm.py, fedavg.py, fedsgd.py, fedprox.py, learningrate_estimator.py)
│   ├── client/     (baseclient.py, fedavgclient.py, fedsgdclient.py, fedproxclient.py)
│   ├── server/     (baseserver.py, fedavgserver.py, fedsgdserver.py, fedproxserver.py)
│   ├── datasets/   (cifar10_concept_drift.py, mnist_concept_drift.py, concept_drift_transforms.py, + LEAF/torchvision/torchtext parsers)
│   ├── loaders/    (data.py, model.py, split.py)
│   ├── metrics/    (basemetric.py, metricszoo.py)
│   ├── models/     (twocnn.py, lenet.py, resnet.py, vgg.py, ...)
│   └── utils.py
└── (no log/ or result/ dirs — generated at runtime)
```

## README (full text quoted)

> # Concept Drift Adaption
> Client-Side Adaptation to Concept Drift in Federated Learning
>
> This repository contains the code to reproduce the figures of our paper "Client-Side Adaptation to Concept Drift in Federated Learning".
>
> ## Original Codebase
> This project is based on Federated-Learning-in-Pytorch by Seok-Ju Hahn, which is licensed under the MIT License.
>
> ## Requirements
> * To guarantee compatibility use python version 3.10.10.
> * The required libraries can be installed using `pip3 install -r requirements.txt`.

Brief, no detail. The repo is a fork of `vaseline555/Federated-Learning-in-PyTorch` with their drift-adaptation additions bolted on.

---

## 1. Recovery-time metric (highest-priority item)

**There is no recovery-time computation anywhere in this repository.** Verified two ways:

```
$ grep -r -i -E "recover|recovery|regain" \
    src/ main.py scripts/ → no matches
```

(Recursive grep across the whole repo returned `No files found`.)

The server only logs raw `acc1` per round to TensorBoard and dumps the same to a JSON result file in `finalize()` (`src/server/fedavgserver.py:431-444`):

```python
def finalize(self):
    logger.info(...)
    with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file:
        results = {key: value for key, value in self.results.items()}
        json.dump(results, result_file, indent=4)
    torch.save(self.model.state_dict(), ...)
```

**Implication:** the "recovery time" / "time to regain original accuracy" reported in the paper is computed **post-hoc, outside the repo**, by hand or with a separate analysis script not in this codebase. There is no in-repo definition of recovery threshold, no consecutive-rounds requirement, no formula. If we want to use that metric as a comparison axis, we have to invent the definition ourselves and apply it consistently — there is no canonical reference implementation to copy.

## 2. Accuracy metric(s)

They report whatever metric the user passes via `--eval_metrics`. The CIFAR-10 scripts pass `--eval_metrics acc1 precision recall` (`scripts/cifar10_scripts/cifar10_iid_steplr_99_1_drift_sudden.sh:2`).

`acc1` is standard top-1 accuracy via sklearn (`src/metrics/metricszoo.py:15-37`):

```python
class Acc1(BaseMetric):
    def collect(self, pred, true):
        ...
    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()
        if scores.size(-1) > 1: # multi-class
            labels = scores.argmax(-1).numpy()
        ...
        return accuracy_score(answers, labels)   # sklearn
```

There are **two evaluation modes** controlled by `--eval_type {local, global, both}` (`main.py:164-171`):

- `global` → server holdout set, central eval (`src/server/fedavgserver.py:302-339`, `_central_evaluate`)
- `local` → each non-participating client evaluates the broadcast model on its own holdout set
- `both` → both above are run each `--eval_every` round

The CIFAR-10 scripts use `--eval_type both --test_fraction 0`. **`--test_fraction 0` is critical** — it means each client has zero local holdout (their `_train_only` flag fires), so `local` eval is a no-op and only the `global` server-side eval actually produces numbers (`src/client/fedavgclient.py:104-106`):

```python
def evaluate(self):
    if self.args._train_only: # `args.test_fraction` == 0
        return {'loss': -1, 'metrics': {'none': -1}}
```

So for CIFAR-10 they effectively report **central server-side top-1 accuracy on a global holdout** — NOT per-client like FedCCFA, just the global model on the central test set.

## 3. Data partition

**CIFAR-10 scripts use IID only.** All `scripts/cifar10_scripts/*.sh` pass `--split_type iid`. The IID code (`src/loaders/split.py:21-30`):

```python
if args.split_type == 'iid':
    shuffled_indices = np.random.permutation(len(dataset))
    split_indices = np.array_split(shuffled_indices, args.K)
    split_map = {k: split_indices[k] for k in range(args.K)}
    return split_map
```

The repo also supports Dirichlet (`diri`), unbalanced, pathological, and pre-defined splits — but the published CIFAR-10 shell scripts don't exercise them. K=100 clients, C=0.1 (10% sampled per round).

## 4. Drift implementation

Two flavors. Both are coded into the **dataset class** (not the training loop), and the trigger is in the server's `update()`.

### Sudden — only swaps 4 classes (3↔7 and 9↔8)

`src/datasets/cifar10_concept_drift.py:28-41`:

```python
def __getitem__(self, index: int) -> Tuple[Any, Any]:
    img, target = super().__getitem__(index)
    if self.sudden_drift:
        if target == 3:
            target = 7
        elif target == 7:
            target = 3
        elif target == 9:
            target = 8
        elif target == 8:
            target = 9
    return img, target
```

Triggered by setting `self.sudden_drift = True` on the dataset (`src/server/fedavgserver.py:345-347`):

```python
elif self.args.drift_start == self.round:
    self.clients[0].training_set.subset.dataset.dataset.dataset.sudden_drift = True
    self.server_dataset.dataset.sudden_drift = True
```

**Both client training data AND server test data are affected** by toggling the flag on the shared underlying dataset — there is only ONE underlying dataset instance under all the wrapping subsets. So **drift is uniform across all clients** (not distributed per-group like FedCCFA or our setup), and the server test set drifts too — the metric is "accuracy on the new post-drift distribution that the model must learn to serve."

### Incremental (called "soft" / "hard") — Gaussian blur ramp

`src/datasets/cifar10_concept_drift.py:43-51`:

```python
def increase_drift_stage(self):
    if self.drift_stage < self.total_stages:
        self.drift_stage += 1
        self.transform = torchvision.transforms.Compose([
            self.original_transform,
            AddGaussianNoise(mean=0, std=self.drift_stage / self.total_stages, mode=self.drift_mode)
        ])
```

Where `AddGaussianNoise` is actually `GaussianBlur` despite the name (`src/datasets/concept_drift_transforms.py:14-24`):

```python
if self.mode == 'soft':
    tensor = torch.clamp(GaussianBlur(7, sigma=(2 * self.std))(tensor), 0, 1)
    return tensor
if self.mode == 'hard':
    tensor = torch.clamp(GaussianBlur(11, sigma=(5 * self.std))(tensor), 0, 1)
    return tensor
```

Drift is in the **input distribution** (feature drift via Gaussian blur), not the labels. Progressive over `drift_duration` rounds.

This is a fundamentally different drift type from our label-swap. **They model image-quality degradation; we model semantic relabeling.**

## 5. LR-adaptation mechanism (their contribution)

**Two estimators** in `src/algorithm/learningrate_estimator.py`:

### (a) `LearningrateEstimatorModel` (server-side, "original" mode)

Operates on the **flattened global-model parameter vector**. This is mathematically identical to FedCCFA's `AdaptiveFedAvgServer.cal_adaptive_lr` — we already ported this as our `AdaptiveFedAvg`. Code:

```python
def estimate(self, model, current_round, base_lr):
    if self.model_ema is None:
        self.initialize(model)
    model_vec = model_to_vec(model)
    # ema on model vector (mean)
    self.model_ema = self.b1 * self.prev_model_ema_na + (1 - self.b1) * model_vec
    self.prev_model_ema_na = copy.deepcopy(self.model_ema)
    self.model_ema = self.model_ema / (1 - pow(self.b1, current_round))   # bias correction
    # ema on variance
    self.variance_ema = (self.b2 * self.prev_variance_ema_na + (1 - self.b2) *
                         np.mean((model_vec - self.prev_model_ema) * (model_vec - self.prev_model_ema)))
    ...
    self.variance_ema = self.variance_ema / (1 - pow(self.b2, current_round))
    # variance ratio
    if self.prev_variance_ema == 0:
        ratio = 1
    else:
        ratio = (self.variance_ema / self.prev_variance_ema)
    self.variance_ratio_ema = self.b3 * self.prev_variance_ratio_ema_na + (1 - self.b3) * ratio
    self.variance_ratio_ema = self.variance_ratio_ema / (1 - pow(self.b3, current_round))
    ...
    lr = min(self.initial_lr, base_lr * self.variance_ratio_ema)
    return lr
```

Three EMAs (mean, variance, ratio), three β parameters, bias-corrected, final LR is `min(initial_lr, base_lr * variance_ratio_ema)`. **Identical formula to FedCCFA AdaptiveFedAvgServer**, with β1/β2/β3 controllable via the `--b1/--b2/--b3` CLI flags. The CIFAR-10 adapt scripts try three β triples (e.g., 0.7/0.3/0.7, 0.5/0.5/0.5, 0.5/0.5/0.9).

Called server-side in their `update()` (`src/server/fedavgserver.py:374-375`):

```python
if self.args.drift_adaptation and self.args.drift_adaptation_mode == "original":
    self.client_lr = self.adapted_lr = self.original_estimator.estimate(
        self.model, self.round, self.lr_scheduler.get_last_lr()[-1]
    )
```

### (b) `LearningrateEstimatorLoss` (client-side, "custom" mode) — this is the actually novel one

Operates on **per-client loss**. Same three-EMA structure but driven by a scalar loss value, not a parameter vector:

```python
def estimate(self, loss, current_round, base_lr):
    self.loss_ema = self.b1 * self.prev_loss_ema_na + (1 - self.b1) * loss
    ...
    self.loss_ema = self.loss_ema / (1 - pow(self.b1, current_round))
    self.variance_ema = (self.b2 * self.prev_variance_ema_na + (1 - self.b2) *
                         (loss - self.prev_loss_ema) * (loss - self.prev_loss_ema))
    ...
    self.variance_ema = self.variance_ema / (1 - pow(self.b2, current_round))
    if self.prev_variance_ema == 0:
        ratio = 1
    else:
        ratio = (self.variance_ema / self.prev_variance_ema)
    self.variance_ratio_ema = self.b3 * self.prev_variance_ratio_ema_na + (1 - self.b3) * ratio
    self.variance_ratio_ema = self.variance_ratio_ema / (1 - pow(self.b3, current_round))
    ...
    lr = min(self.initial_lr, base_lr * self.variance_ratio_ema)
    return lr
```

### The extra per-round communication stage

In `custom` mode, BEFORE the normal sampling+training round, the server makes a **full broadcast to all K clients** for a loss-only forward pass (`src/server/fedavgserver.py:362-366`):

```python
if self.args.drift_adaptation and self.args.drift_adaptation_mode == "custom":
    idx = [c.id for c in self.clients]
    self._broadcast_models(idx)
    self.client_lr = self._request(idx, lr_update=True)
    self._cleanup(idx)
```

Each client then runs `update_estimator()` (`src/client/fedavgclient.py:80-101`):

```python
@torch.inference_mode()
def update_estimator(self):
    self.model.eval()
    self.model.to(self.args.device)
    if self.estimator.id is None:
        self.estimator.id = self.id
    batch_count = 0
    loss_arr = []
    while batch_count <= 50:
        inputs, targets = next(iter(self.train_loader))
        ...
        outputs = self.model(inputs)
        loss_arr.append(self.criterion()(outputs, targets).item())
        batch_count += self.args.B
    loss = np.mean(loss_arr)
    self.args.lr = self.estimator.estimate(loss, self.round, self.args.lr)
    return self.args.lr
```

Each client does up to 50 batches worth of forward-only loss computation on its OWN training set, averages, feeds into its OWN per-client loss-based estimator, returns the adjusted LR. The server averages those LRs across all K clients and uses that as the client LR for the next training round. **This adds one round-trip per training round** — substantial communication overhead — and is what they call "client-side adaptation."

## 6. Training config for CIFAR-10

From `scripts/cifar10_scripts/cifar10_iid_steplr_99_1_drift_sudden_adapt_737.sh:2`:

```
--dataset CIFAR10 --split_type iid --test_fraction 0
--model_name TwoCNN --resize 24 --randhf 0.5 --randjit 0.5 --hidden_size 32
--algorithm fedavg
--K 100 --C 0.1 --R 500 --E 3 --B 50
--optimizer SGD --lr 0.2 --lr_decay 0.99 --lr_decay_step 1 --criterion CrossEntropyLoss
--beta 0
--drift_start 200 --drift_duration 50 --drift_mode sudden
--b1 0.7 --b2 0.3 --b3 0.7
--eval_type both --eval_every 1 --eval_metrics acc1 precision recall
```

| Field | Value |
|---|---|
| Model | TwoCNN (custom small CNN, hidden_size=32) |
| Input resize | 24×24 (NOT 32×32) |
| Augmentation | Horizontal flip + jitter (no random crop) |
| Rounds | **500** |
| Local epochs | **3** |
| Batch size | **50** |
| LR base | **0.2** with step decay 0.99 every 1 round (effectively per-round multiplicative 0.99) |
| Server momentum (β) | 0 (no server momentum) |
| Clients | 100 |
| Participation | 10% per round |
| Drift start | round **200** (of 500) |
| Drift duration | 50 rounds |
| Eval frequency | every round |

**CIFAR-10 is present** with multiple drift settings. The repo also has Fashion-MNIST and MNIST equivalents with their own scripts.

## 7. Baselines

```
$ ls src/algorithm/
basealgorithm.py
fedavg.py
fedprox.py
fedsgd.py
learningrate_estimator.py
```

Only three federated algorithms are implemented: **fedavg, fedprox, fedsgd**. **No Power-of-Choice in the repo.** The CIFAR-10 scripts all use `--algorithm fedavg`. The "baseline vs adapted" comparison they do is:

- **Baseline:** vanilla FedAvg (no `--drift_adaptation`)
- **Adapted variants:** FedAvg + `--drift_adaptation --drift_adaptation_mode original` (server-side model-vec EMA) **or** FedAvg + `--drift_adaptation --drift_adaptation_mode custom` (per-client loss-based)

So their entire experimental comparison is **(plain FedAvg) vs (FedAvg + their LR-adaptation)** at multiple β triples. They do not compare against AdaptiveFedAvg as a separate baseline because their `original` mode IS AdaptiveFedAvg's formula. Power-of-Choice is not implemented.

## 8. Dependencies and framework

`requirements.txt`:

```
torch==1.12.0
torchvision==0.13.0
torchtext==0.13.0
tensorboard==2.12.1
numpy~=1.23.0
pandas==1.5.3
scikit-learn~=1.1.1
transformers==4.27.4
tqdm==4.65.0
matplotlib~=3.8.0
requests~=2.22.0
einops==0.8.0
```

**Framework:** custom server/client/algorithm hierarchy, no Flower / FedML / TFF dependency. The framework is a fork of Hahn's "Federated-Learning-in-PyTorch" with their LR estimator dropped in.

**Python:** 3.10.10 (per README, very specific). **PyTorch:** 1.12.0 — older than FedCCFA's 2.2.x and our 2.9.x. Their `requirements.txt` would pull old wheels; even older than FedCCFA's env.

---

## Integration difficulty assessment

### Where their regime fundamentally conflicts with ours

| Field | Theirs (CIFAR-10) | Ours | Conflict severity |
|---|---|---|---|
| Rounds | 500 | 200 | **High** — recovery dynamics + LR-decay schedule are tuned for 500 rounds with `lr_decay=0.99/round`; running 200 rounds means decay ramp is incomplete |
| Local epochs | 3 | 5 | Low — easily configurable |
| Batch size | 50 | 64 | Low |
| Base LR | 0.2 + step decay 0.99 | 0.01 fixed | **High** — their entire adaptive mechanism is calibrated to start at 0.2 and decay; transplanting to lr=0.01 changes the EMA dynamics entirely |
| Model | TwoCNN, resize 24, hidden=32 | CifarCNN, 32×32, hidden=128 | Medium — different architecture; their estimator math is architecture-agnostic in principle but the parameter-vector dimension affects the variance-ratio numerator |
| Split | IID | **Dir(0.1) non-IID** | **High** — they NEVER ran their CIFAR-10 experiments on non-IID; their per-client-loss estimator may behave very differently when clients have widely varying class distributions |
| Drift type | Uniform sudden swap (3↔7, 9↔8) affecting all clients identically + the server test set | Per-group distributed swap (A: 1↔2, B: 3↔4, C: 5↔6) | **High** — their drift is "everyone's task changed the same way"; ours is "different groups changed differently." Their estimator is designed for the homogeneous case. |
| Drift placement | round 200 of 500 (40% in) | round 100 of 200 (50% in) | Low |
| Sample ratio | C=0.1 (10 of 100 clients/round) | full participation (20 of 20) | **High** — their "custom" mode broadcasts to ALL K clients for the loss-update before sampling; this is the same in either case, but the published numbers were at K=100 with C=0.1. With our K=20, C=1.0 the noise characteristics of the loss-ratio EMA change. |
| Eval | server-side global test set (per-client `test_fraction=0` so no local eval) | global canonical test (and per-client if we add FedCCFA-style metric) | Low — compatible |

### Two paths

**(a) Run their code as-is in their own regime.**
- Spin up Python 3.10.10 + torch 1.12.0 venv (older, more brittle to install). Run their CIFAR-10 sudden + adapt scripts. Get their original numbers.
- Pros: faithful reproduction; their `--drift_adaptation` is their actual method.
- Cons: incomparable to our results. Their setup is IID + 500 rounds + lr=0.2 + uniform-drift, ours is Dir(0.1) + 200 rounds + lr=0.01 + distributed-drift. The numbers from their setup don't tell you how their method would do on OUR problem. And critically, there is **no recovery-time metric in their code** — you'd still have to define and compute it yourself.

**(b) Port their LR-adaptation into our harness as a baseline.**
- ~80 lines of code in our `all_experiments_optimized.py` for `LearningrateEstimatorLoss` + the extra "lr_update round" loop. The math is straightforward and architecture-agnostic.
- Run it under OUR setup: Dir(0.1), 200 rounds, batch 64, lr 0.01, our drift schedule, our test set.
- Pros: head-to-head against OurMethod / FedAvg / Flash / AdaptiveFedAvg on the SAME problem. Apples-to-apples.
- Cons: their β-triples were tuned for their 500-round / lr=0.2 / IID setting. Their custom-mode loss EMA may behave poorly with lr=0.01 (much smaller per-step loss changes → less signal for the variance-ratio EMA). We'd want to sweep β triples to be fair to them.

**Concretely for (b), what would change:**
- Add a `LearningrateEstimatorLoss` class (copy from their file, ~50 lines)
- Add a "Method 5: SaileLossAdaptation" runner in our script that mirrors `run_adaptive_fedavg` but uses the per-client loss EMA + extra communication round
- Hyperparameters to take from them: β1=0.7, β2=0.3, β3=0.7 (their adapt_737 script). lr=0.01 (ours, NOT theirs) — and acknowledge the regime shift as a fair-comparison caveat.
- The model parameter-vector estimator (`original` mode) is **already in our codebase as AdaptiveFedAvg**, so we don't double-port it.

### Recommendation

**Path (b).** Their `LearningrateEstimatorModel` is already in our pipeline (as AdaptiveFedAvg). Their `LearningrateEstimatorLoss` (the per-client loss-driven variant with extra round-trip) is genuinely novel and not yet a baseline in our work. ~80 lines of additional code to add it. The β-triple sweep is a separate decision but cheap (3 values × ~$1.50 each).

**Path (a) is not worth it** because (i) we can't compare to it apples-to-apples even if it succeeds, and (ii) they don't ship the recovery-time computation we actually want to study, so running their code doesn't give us a metric to compare anything on.

Also worth flagging: **their drift model (uniform Gaussian blur or class-pair-swap) is meaningfully different from ours** (distributed per-group label swap). When we run their LR-adaptation method on our drift, we're testing whether the *mechanism* generalizes to a different drift type — that's still a legitimate baseline comparison, but we should frame it that way in the writeup, not as "reproducing their result."

---

## TL;DR

| Question | Answer |
|---|---|
| Recovery-time metric in code? | **No.** Not anywhere. Must be defined and computed externally. |
| Accuracy metric reported? | Central server-side top-1 (`acc1`) on a global holdout. |
| Data partition? | IID only for CIFAR-10 scripts. |
| Drift implementation? | Sudden = swap classes 3↔7 and 9↔8 (uniform across clients). Soft/Hard = progressive Gaussian blur (feature shift). |
| LR-adaptation mechanism? | Two variants: server-side model-vec EMA (= AdaptiveFedAvg formula), OR per-client loss EMA with extra round-trip per round. |
| CIFAR-10 config? | R=500, E=3, B=50, lr=0.2, decay 0.99/rd, TwoCNN, IID, K=100, C=0.1. Drift at round 200 of 500. |
| Baselines? | Only FedAvg, FedProx, FedSGD. No Power-of-Choice. No separate Adaptive-FedAvg (it's the same formula as their `original` mode). |
| Dependencies? | Python 3.10.10, PyTorch 1.12.0, custom (non-Flower) FL framework. |
| Integration path? | **Port only the `LearningrateEstimatorLoss` (per-client loss variant) into our harness as a new baseline. ~80 lines.** The other variant is already covered by our AdaptiveFedAvg. |
