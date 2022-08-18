---
# _Model_
---
### Create Azure Container Registry
	
	Go to Services -> Container Registries
	Click on Add
	Subscription: <Subscription>
	Resource Group: <RG>
	Registry Name: paigesModelACR (NAME should be unique across Azure Cloud)
	Location: Central India
	SKU: Basic 
	Click on Review + Create
	Click on Create
 
### Docker Build
	
	docker build -t paiges_pred:1.0.RELEASE .

### List Docker Images

	docker images
	docker images paiges_pred:1.0.RELEASE
	
### Enable Docker Login for ACR Repository
	
	Go to Services -> Container Registries -> paigesExternalACR
	Go to Access Keys
	Click on Enable Admin User
	Make a note of Username and password


# Integrate ACR with AKS
	
	export ACR_REGISTRY=paigesmodelacr.azurecr.io
	export ACR_NAMESPACE=app1
	export ACR_IMAGE_NAME=paiges_pred
	export ACR_IMAGE_TAG=1.0.RELEASE
	echo $ACR_REGISTRY, $ACR_NAMESPACE, $ACR_IMAGE_NAME, $ACR_IMAGE_TAG

### Login to ACR

	docker login $ACR_REGISTRY

### Tag

	docker tag paiges_pred:1.0.RELEASE  $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

#### It replaces as below
	docker tag paiges_pred:1.0.RELEASE paigesmodelacr.azurecr.io/app1/paiges_pred:1.0.RELEASE

### List Docker Images to verify

	docker images paiges_pred:1.0.RELEASE
	docker images $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

### Push Docker Images

	docker push $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG



# Configure ACR integration with existing AKS Cluster

### Set ACR NAME

	export ACR_NAME=paigesModelACR
	echo $ACR_NAME

### set kube-config to connect to cluster in Azure

	az aks get-credentials --resource-group <resource-group> --name <AKS cluster>
	az aks get-credentials --resource-group RG-TAPP --name cluster-paiges-pred


### Template

	az aks update -n myAKSCluster -g myResourceGroup --attach-acr <acr-name>

### Replace Cluster, Resource Group and ACR Repo Name

	az aks update -n aks-paiges-pred -g RG-TAPP --attach-acr $ACR_NAME

## Update Deployment Manifest with Image Name
	
	spec:
      containers:
        - name: paiges-pred-api
          image: paigesmodelacr.azurecr.io/app1/paiges_pred:1.0.RELEASE
          imagePullPolicy: Always
          ports:
            - containerPort: 80

## Create Kubernetes secret for pgbouncer-sidecar
	kubectl create secret generic azure-pgbouncer-config --from-file=pgbouncer.ini --from-file=userlist.txt
	
	- Note: The input for the secret is as configured in pgbouncer.ini and userlist.txt 
	
## Deploy to kubernetes
	kubectl apply -f kube-manifests/deployment_manual.yaml
	

---
# _Extraction_
---

### Create Azure Container Registry
	
	Go to Services -> Container Registries
	Click on Add
	Subscription: <Subscription>
	Resource Group: <RG>
	Registry Name: paigesExtractionACR (NAME should be unique across Azure Cloud)
	Location: Central India
	SKU: Basic 
	Click on Review + Create
	Click on Create
 
### Docker Build
	
	docker build -t paiges_extract:1.0.RELEASE .

### List Docker Images

	docker images
	docker images paiges_extract:1.0.RELEASE
	
### Enable Docker Login for ACR Repository
	
	Go to Services -> Container Registries -> paigesExtractionACR
	Go to Access Keys
	Click on Enable Admin User
	Make a note of Username and password


### set kube-config to connect to cluster in Azure

	az aks get-credentials --resource-group <resource-group> --name <AKS cluster>
	az aks get-credentials --resource-group RG-TAPP --name cluster-paiges-extract

# Integrate ACR with AKS
	
	export ACR_REGISTRY=paigesextractionacr.azurecr.io
	export ACR_NAMESPACE=app1
	export ACR_IMAGE_NAME=paiges_extract
	export ACR_IMAGE_TAG=1.0.RELEASE
	echo $ACR_REGISTRY, $ACR_NAMESPACE, $ACR_IMAGE_NAME, $ACR_IMAGE_TAG

### Login to ACR

	docker login $ACR_REGISTRY

### Tag

	docker tag paiges_extract:1.0.RELEASE  $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

#### It replaces as below
	docker tag paiges_extract:1.0.RELEASE paigesextractionacr.azurecr.io/app1/paiges_extract:1.0.RELEASE
	
	
### List Docker Images to verify

	docker images paiges_extract:1.0.RELEASE
	docker images $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

### Push Docker Images

	docker push $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG



# Configure ACR integration with existing AKS Cluster

### Set ACR NAME

	export ACR_NAME=paigesExtractionACR
	echo $ACR_NAME



### Template

	az aks update -n myAKSCluster -g myResourceGroup --attach-acr <acr-name>

### Replace Cluster, Resource Group and ACR Repo Name

	az aks update -n cluster-paiges-extract -g RG-TAPP --attach-acr $ACR_NAME

## Update Deployment Manifest with Image Name
	
	spec:
      containers:
        - name: paiges-extract-api
          image: paigesextractionacr.azurecr.io/app1/paiges_extract:1.0.RELEASE
          imagePullPolicy: Always
          ports:
            - containerPort: 80

## Create Kubernetes secret for pgbouncer-sidecar
	kubectl create secret generic azure-pgbouncer-config --from-file=pgbouncer.ini --from-file=userlist.txt
	
	- Note: The input for the secret is as configured in pgbouncer.ini and userlist.txt 
	
## Deploy to kubernetes
	kubectl apply -f kube-manifests/deployment_manual.yaml

---
# _Exernal_
---

### Create Azure Container Registry
	
	Go to Services -> Container Registries
	Click on Add
	Subscription: <Subscription>
	Resource Group: <RG>
	Registry Name: paigesExternalACR (NAME should be unique across Azure Cloud)
	Location: Central India
	SKU: Basic 
	Click on Review + Create
	Click on Create
 
### Docker Build
	
	docker build --no-cache -t paiges_external:1.0.RELEASE .

### List Docker Images

	docker images
	docker images paiges_external:1.0.RELEASE
	
### Enable Docker Login for ACR Repository
	
	Go to Services -> Container Registries -> paigesExternalACR
	Go to Access Keys
	Click on Enable Admin User
	Make a note of Username and password


### set kube-config to connect to cluster in Azure

	az aks get-credentials --resource-group <resource-group> --name <AKS cluster>
	az aks get-credentials --resource-group RG-TAPP --name aks-paiges-external

# Integrate ACR with AKS
	
	export ACR_REGISTRY=paigesexternalacr.azurecr.io
	export ACR_NAMESPACE=app1
	export ACR_IMAGE_NAME=paiges_external
	export ACR_IMAGE_TAG=1.0.RELEASE
	echo $ACR_REGISTRY, $ACR_NAMESPACE, $ACR_IMAGE_NAME, $ACR_IMAGE_TAG

### Login to ACR

	docker login $ACR_REGISTRY

### Tag

	docker tag paiges_external:1.0.RELEASE  $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

#### It replaces as below
	docker tag paiges_external:1.0.RELEASE paigesextractionacr.azurecr.io/app1/paiges_external:1.0.RELEASE
	
	
### List Docker Images to verify

	docker images paiges_external:1.0.RELEASE
	docker images $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG

### Push Docker Images

	docker push $ACR_REGISTRY/$ACR_NAMESPACE/$ACR_IMAGE_NAME:$ACR_IMAGE_TAG



# Configure ACR integration with existing AKS Cluster

### Set ACR NAME

	export ACR_NAME=paigesExternalACR
	echo $ACR_NAME



### Template

	az aks update -n myAKSCluster -g myResourceGroup --attach-acr <acr-name>

### Replace Cluster, Resource Group and ACR Repo Name

	az aks update -n cluster-paiges-external -g RG-TAPP --attach-acr $ACR_NAME

## Update Deployment Manifest with Image Name
	
	spec:
      containers:
        - name: paiges-auth-api
          image: paigesexternalacr.azurecr.io/app1/paiges_external:1.0.RELEASE
          imagePullPolicy: Always
          ports:
            - containerPort: 80

## Create Kubernetes secret for pgbouncer-sidecar
	kubectl create secret generic azure-pgbouncer-config --from-file=pgbouncer.ini --from-file=userlist.txt
	
	- Note: The input for the secret is as configured in pgbouncer.ini and userlist.txt 
	
## Deploy to kubernetes
	kubectl apply -f kube-manifests/deployment_manual.yaml

---
