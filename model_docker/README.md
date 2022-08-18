# _Model_
---
### Create Azure Container Registry
	
	Go to Services -> Container Registries
	Click on Add
	Subscription: <Subscription>
	Resource Group: <RG>
	Registry Name: paigesACR (NAME should be unique across Azure Cloud)
	Location: Central India
	SKU: Basic 
	Click on Review + Create
	Click on Create
 
### Docker Build
	
	docker build --no-cache -t paiges_pred:2.0.RELEASE .

### List Docker Images

	docker images
	docker images paiges_pred:2.0.RELEASE
	
### Enable Docker Login for ACR Repository
	
	Go to Services -> Container Registries -> paigesACR
	Go to Access Keys
	Click on Enable Admin User
	Make a note of Username and password


# Integrate ACR with AKS
	
	export ACR_REGISTRY=paigesacr.azurecr.io
	export ACR_NAMESPACE=app1
	export ACR_IMAGE_NAME=paiges_pred
	export ACR_IMAGE_TAG=2.0.RELEASE
	echo $ACR_REGISTRY, $ACR_NAMESPACE, $ACR_IMAGE_NAME, $ACR_IMAGE_TAG

### Login to ACR

	docker login $ACR_REGISTRY

### Tag

	docker tag paiges_pred:2.0.RELEASE  $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

#### It replaces as below
	docker tag paiges_pred:2.0.RELEASE paigesacr.azurecr.io/app1/paiges_pred:2.0.RELEASE

### List Docker Images to verify

	docker images paiges_pred:2.0.RELEASE
	docker images $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

### Push Docker Images

	docker push $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG



# Configure ACR integration with existing AKS Cluster

### Set ACR NAME

	export ACR_NAME=paigesACR
	echo $ACR_NAME

### set kube-config to connect to cluster in Azure

	az aks get-credentials --resource-group <resource-group> --name <AKS cluster>
	az aks get-credentials --resource-group RG-TAPP --name aks-paiges-pred


### Template

	az aks update -n myAKSCluster -g myResourceGroup --attach-acr <acr-name>

### Replace Cluster, Resource Group and ACR Repo Name

	az aks update -n aks-paiges-pred -g RG-TAPP --attach-acr $ACR_NAME

## Update Deployment Manifest with Image Name
	
	spec:
      containers:
        - name: paiges-pred-api
          image: paigesacr.azurecr.io/app1/paiges_pred:2.0.RELEASE
          imagePullPolicy: Always
          ports:
            - containerPort: 80
	
## Deploy to kubernetes
	kubectl apply -f kube-manifests/deployment_manual.yaml


## Deploy Metrics Server 

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```