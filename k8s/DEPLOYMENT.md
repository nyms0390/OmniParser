# Deployment Checklist

## Fill in blanks in manifests

| File | Field | What to put |
|---|---|---|
| `omniparser.yaml` | `image:` | Your registry image, e.g. `myregistry.azurecr.io/omniparser-server:latest` |
| `paddleocr.yaml` | `image:` | Your PaddleOCR server image |
| `paddleocr.yaml` | `storage:` (×2) | e.g. `5Gi` |
| `agent.yaml` | `image:` | Your agent image |
| `agent.yaml` | `ANTHROPIC_API_KEY` (and others) | Real API keys |
| `agent.yaml` | `WINDOWS_HOST_URL` | External Windows host URL if used |
| `storageclass.yaml` | `storageAccount:` | Your Azure storage account name (or remove to use dynamic provisioning) |
| `gta1/omniparser/paddleocr .yaml` | `volumeName:` | Actual blob container names you create |

---

## Azure infrastructure (AKS)

1. **AKS cluster** with a GPU node pool (`agentpool: gpupool`, taint `sku=gpu:NoSchedule`)
2. **Enable Blob CSI driver**: `az aks update -n <cluster> -g <rg> --enable-blob-driver`
3. **Storage account + 3 blob containers**, pre-populated:
   - `gta1-models` ← `GTA1/models/GTA1-7B-2507/`
   - `omni-weights` ← `OmniParser/weights/`
   - `paddleocr-models` ← PaddleOCR model files
4. **NGINX ingress controller**: apply the URL in the comment at the top of `ingress.yaml`
5. **Uncomment** `nodeSelector`/`tolerations` in `gta1.yaml`, `omniparser.yaml`, `paddleocr.yaml`
6. **Switch storage**: for AKS keep Blob PVC active, hostPath blocks commented out

---

## Registry & secrets

1. Build and push images for all 4 services to your private registry
2. Create pull secret in the `omni` namespace:
   ```bash
   kubectl create secret docker-registry regcred \
     --namespace omni \
     --docker-server=<registry> \
     --docker-username=<user> \
     --docker-password=<token>
   ```

---

## Apply order

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/storageclass.yaml
kubectl apply -f k8s/gta1.yaml
kubectl apply -f k8s/paddleocr.yaml
kubectl apply -f k8s/omniparser.yaml
kubectl apply -f k8s/agent.yaml      # when ready
kubectl apply -f k8s/ingress.yaml
```

---

## Local testing (macOS, no GPU)

- Keep hostPath PV+PVC active (comment out Blob PVC, uncomment hostPath blocks)
- Leave `nodeSelector`/`tolerations` commented out
- GPU pods will pend indefinitely — expected; test only the agent or mock the GPU services
