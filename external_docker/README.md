---
# _Exernal_
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
	
	docker build --no-cache -t paiges_external:2.0.RELEASE .

### List Docker Images

	docker images
	docker images paiges_external:2.0.RELEASE
	
### Enable Docker Login for ACR Repository
	
	Go to Services -> Container Registries -> paigesACR
	Go to Access Keys
	Click on Enable Admin User
	Make a note of Username and password


### set kube-config to connect to cluster in Azure

	az aks get-credentials --resource-group <resource-group> --name <AKS cluster>
	az aks get-credentials --resource-group RG-TAPP --name aks-paiges-external

# Integrate ACR with AKS
	
	export ACR_REGISTRY=paigesacr.azurecr.io
	export ACR_NAMESPACE=app1
	export ACR_IMAGE_NAME=paiges_external
	export ACR_IMAGE_TAG=2.0.RELEASE
	echo $ACR_REGISTRY, $ACR_NAMESPACE, $ACR_IMAGE_NAME, $ACR_IMAGE_TAG

### Login to ACR

	docker login $ACR_REGISTRY

### Tag

	docker tag paiges_external:2.0.RELEASE  $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

#### It replaces as below
	docker tag paiges_external:2.0.RELEASE paigesacr.azurecr.io/app1/paiges_external:2.0.RELEASE
	
	
### List Docker Images to verify

	docker images paiges_external:2.0.RELEASE
	docker images $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

### Push Docker Images

	docker push $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG



# Configure ACR integration with existing AKS Cluster

### Set ACR NAME

	export ACR_NAME=paigesACR
	echo $ACR_NAME



### Template

	az aks update -n myAKSCluster -g myResourceGroup --attach-acr <acr-name>

### Replace Cluster, Resource Group and ACR Repo Name

	az aks update -n aks-paiges-external -g RG-TAPP --attach-acr $ACR_NAME

## Update Deployment Manifest with Image Name
	
	spec:
      containers:
        - name: paiges-auth-api
          image: paigesacr.azurecr.io/app1/paiges_external:2.0.RELEASE
          imagePullPolicy: Always
          ports:
            - containerPort: 80

## Create Kubernetes secret for pgbouncer-sidecar
	kubectl create secret generic azure-pgbouncer-config --from-file=pgbouncer.ini --from-file=userlist.txt
	
	- Note: The input for the secret is as configured in pgbouncer.ini and userlist.txt 
	
## Deploy to kubernetes
	kubectl apply -f kube-manifests/deployment_manual.yaml
	
## Deploy metrics server

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

---
