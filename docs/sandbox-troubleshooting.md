# Sandbox troubleshooting

General operational gotchas for running any pipeline in this repo against a
local Michelangelo sandbox (`ma sandbox create`). These aren't specific to
any one pipeline — they come up regardless of which example you're running.
For pipeline-specific setup, see that pipeline's own README.

## `ma sandbox create` fails partway through

`create` calls `k3d cluster create` unconditionally, so retrying a failed
`create` fails immediately with "cluster already exists." If a first attempt
fails partway, use `ma sandbox sync` to continue — it skips cluster creation
and goes straight to the operator/Helm install steps.

Before deleting and retrying anything, check `kubectl get pods
--all-namespaces`. A `helm --wait --timeout` failure on an operator install
(e.g. kuberay-operator, spark-operator) is often a false failure: large image
pulls can exceed the timeout window while the pods are still coming up
healthy in the background. If the previously-failing pods are now `Running`,
`ma sandbox sync` will pick up cleanly — no need for `ma sandbox delete`.

## Sandbox doesn't come back after a machine restart

`ma sandbox start` restarts the k3d nodes but can surface a few restart-only
failure modes before pods are usable again:

- **`ma sandbox start` fails on a missing `.pem` path.** k3d bakes a temp-file
  path for the local CA cert into the cluster's volume mount at creation
  time; the temp file doesn't survive a restart. If the path now exists as a
  directory instead (some OSes recreate it that way), remove it first, then
  recreate the file from the system keychain:

  ```bash
  rmdir <path-from-error>   # only if it's now a directory
  security find-certificate -a -p /Library/Keychains/System.keychain > <path-from-error>
  ma sandbox start
  ```

- **Pods stuck in `CreateContainerConfigError: secret not found` even though
  the secret exists.** k3d's kubelet caches a stale "not found" result from
  before the secret was created; deleting/recreating the pod doesn't clear
  it. Restart the k3s server process instead:

  ```bash
  docker restart k3d-<cluster-name>-server-0
  # wait ~30s, then:
  kubectl get pods -n default
  ```

- **The k3d server container itself is missing** (k3d reports the cluster as
  running, but `kubectl` can't connect at all). This isn't recoverable by
  restarting anything — delete and recreate the cluster
  (`ma sandbox delete && ma sandbox create`).

After any of the above, if only `kube-system` pods are present (no
Michelangelo pods), run `ma sandbox sync` to redeploy the Helm chart and
bring them back — a restart does not restart pod processes on its own.

## MA Studio shows HTTP 415 / "Unable to fetch data"

MA Studio's browser client posts plain `application/json`; Envoy's
`grpc_json_transcoder` filter is what converts that into gRPC for the
apiserver. A 415 usually means that filter isn't configured. Check with:

```bash
kubectl get configmap -n default -o yaml | grep -A2 http_filters
```

It should show `envoy.filters.http.grpc_json_transcoder`. If it's missing or
different, `ma sandbox sync` redeploys the Envoy config from the chart. A
manual ConfigMap patch needs a following
`kubectl rollout restart deployment/michelangelo-envoy -n default` — there's
no checksum annotation on the deployment to trigger this automatically.

## Cadence domain missing (`EntityNotExistsError`)

`ma sandbox sync` registers the Cadence domain as one of its last steps.
If an earlier step in the same sync timed out, an exception can skip domain
registration entirely, and pipeline runs fail immediately with something
like:

```
EntityNotExistsError{Message: Domain default does not exist.}
```

Fix by registering it manually:

```bash
kubectl run cadence-reg --restart=Never --image ubercadence/cli:v1.2.6 \
  --env=CADENCE_CLI_ADDRESS=michelangelo-cadence-frontend:7933 \
  --command -- cadence --domain default domain register --rd 1
kubectl logs cadence-reg   # confirm it registered
kubectl delete pod cadence-reg
```

Note: `--command` is required so `cadence` is treated as the binary being
run, not a doubled argument to the image's entrypoint. Failed PipelineRuns
from before the fix can't be retried in place — delete and reapply the CR.

## Zombie RayCluster CRs accumulate and block new clusters

Each failed or abandoned PipelineRun can leave behind Ray-related CRs in
**two separate CRD groups**: `rayclusters.ray.io` (kuberay) and
`rayclusters.michelangelo.api` (Michelangelo's own tracking CRD). Deleting
only one group leaves the other piling up, which saturates the controller's
reconcile queue and eventually causes new cluster creation to fail (errors
like `internal error: nil (not None) returned from ... create_cluster`).

Clean up both groups, plus any failed pods, before submitting a new run:

```bash
kubectl delete raycluster.michelangelo.api -n default --all
kubectl delete raycluster.ray.io -n default --all
kubectl delete pod -n default --field-selector=status.phase=Failed
```

Safe to run at any time — clusters for an actively-running PipelineRun are
recreated automatically.

## Helm SSA field-manager conflict after `kubectl set image`

If `ma sandbox sync` fails with something like:

```
conflict with "kubectl-set" using apps/v1: .spec.template.spec.containers[name="app"].image
```

a previous `kubectl set image` call took ownership of that field via
server-side apply, and Helm's own SSA can no longer update it (`helm upgrade
--force` doesn't fix this either — it's incompatible with SSA). Delete the
deployment and let Helm recreate it with clean field ownership:

```bash
kubectl delete deployment <name>   # e.g. michelangelo-controllermgr
ma sandbox sync
```

## `HeadPodNotFound` failing every task after the first

If pipeline tasks after the first one fail immediately when a new Ray
cluster spins up, check whether your controllermgr build predates upstream
commit `cfeb1dd0` (`fix(ray): treat HeadPodNotFound as transient
provisioning state`, in `michelangelo-ai/michelangelo`). Without that fix,
a transient `HeadPodReady=Unknown, reason=HeadPodNotFound` status — which
kuberay sets briefly while the head pod is still being scheduled — gets
treated as a fatal terminal error instead of a retry-able one. Check
`kubectl logs` on the controllermgr pod for `HeadPodNotFound` around the
time of failure to confirm.

## RayJob permanently fails with `BackoffLimitExceeded`

A fresh Ray head pod's first boot can take several minutes before its
liveness/readiness endpoints respond — this is normal, and the pod recovers
on its own. The problem is that KubeRay's RayJob submitter defaults to only
`submitterBackoffLimit: 2` (3 total attempts), which exhausts in well under a
minute — long before the head pod has stabilized — so the RayJob then
permanently fails with `BackoffLimitExceeded` even though the cluster would
have come up fine given more time. Submitter pod logs typically show
`ConnectionRefusedError` or `ServerDisconnectedError` against the dashboard
port.

This is specific to `xgb_train` (and any future pipeline going through
Ray Train's distributed trainers) — `pytorch_train`'s Ray usage doesn't
submit through a `RayJob` the same way.

**Fix**: [`michelangelo-ai/michelangelo` PR #1573](https://github.com/michelangelo-ai/michelangelo/pull/1573)
raises the submitter backoff limit and tolerates the same transient
`HeadPodNotFound`/`ContainersNotReady` conditions as the fix above. Not yet
merged as of this writing — if you hit `BackoffLimitExceeded` before it
lands, the workaround is the same custom-image approach as the
`HeadPodNotFound` fix: build a controllermgr image from a commit with PR
#1573's changes and deploy it with `kubectl set image
deployment/michelangelo-controllermgr app=<image> -n default` (remember this
takes Helm field ownership — see the SSA conflict entry above if `ma sandbox
sync` fails afterward).

## `kuberay-historyserver` stuck in `ImagePullBackOff`

The `kuberay-historyserver:v0.1.0` image has no registry prefix in its
manifest, so containerd resolves it to `docker.io/library/...`, which
doesn't exist — it must be built locally and imported into the cluster
before the sandbox will come up cleanly. See kuberay's own build tooling
for a build-and-import script if your sandbox setup doesn't already handle
this.
