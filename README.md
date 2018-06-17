# Distributed Tensorflow on Kubernetes

Build a fully integrated pipeline to train your machine learning models with Tensorflow and Kubernetes.

This repo will guide you through:

1. setting up a local environment with python, pip and tensorflow
1. packaging up your models as Docker containers
1. creating and configuring a Kubernetes cluster
1. deploying models in your cluster
1. scaling your model using Distributed Tensorflow
1. serving your model
1. tuning your model using hyperparameter optimisation

## Prerequisites

You should have the following tools installed:

- minikube
- kubectl
- ksonnet
- python 2.7
- pip
- sed
- an account on Docker Hub
- an account on GCP
- Gcloud

## Recognising handwritten digits

MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:

![MNIST dataset](assets/MNIST.png)

It also includes labels for each image, telling us which digit it is. For example, the labels for the above images are 5, 0, 4, and 1.

In this tutorial, you're going to train a model to look at images and predict what digits they are.

## Writing scalable Tensorflow

If you plan to train your model using distributed Tensorflow you should be aware of:

- you should use the [Estimator API](https://medium.com/@yuu.ishikawa/serving-pre-modeled-and-custom-tensorflow-estimator-with-tensorflow-serving-12833b4be421) where possible.
- distribute Tensorflow works only with [tf.estimator.train_and_evaluate](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate). If you use the method [train](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train) and [evaluate](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train) it won't work.
- you should save your model with [export_savedmodel](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_savedmodel) so that Tensorflow serving can serve them
- you should use use [tf.estimator.RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) to read the configuration from the environment. The Tensorflow operator in Kubedflow automatically populated the environment variables that are consumed by that class.

## Setting up a local environment

You can create a virtual environment for python with:

```bash
virtualenv --system-site-packages --python /usr/bin/python src
```

> Please note that you may have to customise the path for your `python` binary.

You can activate the virtual environment with:

```bash
cd src
source bin/activate
```

You should install the dependencies with:

```bash
pip install -r requirements.txt
```

You can test that the script works as expected with:

```bash
python main.py
```

## Packaging up your models as Docker containers

You can package your application in a Docker image with:

```bash
cd src
docker build -t learnk8s/mnist:1.0.0 .
```

> Please note that you may want to customise the image to have the username of your Docker Hub account instead of _learnk8s_

You can test the Docker image with:

```bash
docker run -ti learnk8s/mnist:1.0.0
```

You can upload the Docker image to the Docker Hub registry with:

```bash
docker push learnk8s/mnist:1.0.0
```

## Creating and configuring a Kubernetes cluster

You can train your models in the cloud or locally.

### Minikube

You can create a local Kubernetes cluster with minikube:

```bash
minikube start --cpus 4 --memory 8096 --disk-size=40g
```

Once your cluster is ready, you can install [kubeflow](https://github.com/kubeflow/kubeflow).

#### Kubeflow

You can download the packages with:

```bash
ks init my-kubeflow
cd my-kubeflow
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/v0.1.2/kubeflow
ks pkg install kubeflow/core@v0.1.2
ks pkg install kubeflow/tf-serving@v0.1.2
ks pkg install kubeflow/tf-job@v0.1.2
```

You can generate a component from a Ksonnet prototype with:

```bash
ks generate core kubeflow-core --name=kubeflow-core
```

Create a separate namespace for kubeflow:

```bash
kubectl create namespace kubeflow
```

Make the environment the default environment for ksonnet with:

```bash
ks env set default --namespace kubeflow
```

Deploy kubeflow with:

```bash
ks apply default -c kubeflow-core
```

#### NFS

To use distributed Tensorflow, you have to share a filesystem between the master node and the parameter servers.

You can create an NFS server with:

```bash
kubectl create -f kube/nfs-minikube.yaml
```

Make a note of the IP of the service for the NFS server with:

```bash
kubectl get svc nfs-server
```

Replace `nfs-server.default.svc.cluster.local` with the ip address of the service in `kube/pvc-minikube.yaml`.

The change is necessary since kube-dns is not configured correctly in the VM and the kubelet can't resolve the domain name.

Create the volume with:

```bash
kubectl create -f kube/pvc-minikube.yaml
```

### GKE

Create a cluster on GKE with:

```bash
gcloud container clusters create distributed-tf --machine-type=n1-standard-8 --num-nodes=3
```

You can obtain the credentials for `kubectl` with:

```bash
gcloud container clusters get-credentials distributed-tf
```

Give yourself admin permission to install kubeflow:

```bash
kubectl create clusterrolebinding default-admin --clusterrole=cluster-admin --user=daniele.polencic@gmail.com
```

#### Kubeflow

You can download the packages with:

```bash
ks init my-kubeflow
cd my-kubeflow
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/v0.1.2/kubeflow
ks pkg install kubeflow/core@v0.1.2
ks pkg install kubeflow/tf-serving@v0.1.2
ks pkg install kubeflow/tf-job@v0.1.2
```

You can generate a component from a Ksonnet prototype with:

```bash
ks generate core kubeflow-core --name=kubeflow-core
```

Create a separate namespace for kubeflow:

```bash
kubectl create namespace kubeflow
```

Make the environment the default environment for ksonnet with:

```bash
ks env set default --namespace kubeflow
```

Configure kubeflow to run in the Google Cloud Platform:

```bash
ks param set kubeflow-core cloud gcp
```

Deploy kubeflow with:

```bash
ks apply default -c kubeflow-core
```

#### NFS

To use distributed Tensorflow, you have to share a filesystem between the master node and the parameter servers.

Create a Google Compute Engine persistent disk:

```bash
gcloud compute disks create --size=10GB gce-nfs-disk
```

You can create an NFS server with:

```bash
kubectl create -f kube/nfs-gke.yaml
```

Create an NFS volume with:

```bash
kubectl create -f kube/pvc-gke.yaml
```

## Deploying models in your cluster

You can submit a job to Kubernetes to run your Docker container with:

```bash
kubectl create -f kube/job.yaml
```

> Please note that you may want to customise the image for your container.

The job runs a single container and doesn't scale.

However, it is still more convenient than running it on your computer.

## Scaling your model using Distributed Tensorflow

You can run a distributed Tensorflow job on your NFS filesystem with:

```bash
kubectl create -f kube/tfjob.yaml
```

The results are stored in the NFS volume.

You can visualise the detail of your distributed tensorflow job with [Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).

You can deploy Tensorboard with:

```bash
kubectl create -f kube/tensorboard.yaml
```

Retrieve the name of the Tensorboard's Pod with:

```bash
kubectl get pods -l app=tensorboard
```

You can forward the traffic from the Pod on your cluster to your computer with:

```bash
kubectl port-forward tensorboard-XX-ID-XX 8080:6006
```

> Please note that you should probably use an Ingress manifest to expose your service to the public permanently.

You can visit the dashboard at [http://localhost:8080](http://localhost:8080).

## Serving your model

You can serve your model with [Tensorflow Serving](https://www.tensorflow.org/serving/).

You can create a Tensorflow Serving server with:

```bash
kubectl create -f kube/serving.yaml
```

Retrieve the name of the Tensorflow Serving's Pod with:

```bash
kubectl get pods -l app=tf-serving
```

You can forward the traffic from the Tensorboard's Pod on your cluster to your computer with:

```bash
kubectl port-forward tf-serving-XX-ID-XX 8080:9000
```

> Please note that you should probably use an Ingress manifest to expose your service to the public permanently.

You can query the model using the client:

```bash
cd src
python client.py --host localhost --port 8080 --image ../data/4.png --signature_name predict --model test
```

> Please make sure your virtualenv is still active.

The model should recognise the digit 4.

## Tuning your model using hyperparameter optimisation

The model can be tuned with the following parameters:

- the learning rate
- the number of hidden layers in the neural network

You could submit a set of jobs to investigate the different combinations of parameters.

The `templated` folder contains a `tf-templated.yaml` file with placeholders for the variables.

The `run.sh` script interpolated the values and submit the TFJobs to the cluster.

Before you run the jobs, make sure you have your Tensorboard running locally:

```bash
kubectl port-forward tensorboard-XX-ID-XX 8080:6006
```

You can run the test with:

```bash
cd templated
./run.sh
```

You can follow the progress of the training in real-time at [http://localhost:8080](http://localhost:8080).

## Final notes

You should probably expose your services such as Tensorboard and Tensorflow Serving with an ingress manifest rather than using the port forwarding functionality in `kube-proxy`.

The NFS volume is running on a single instance and isn't highly available. Having a single node for your storage may work if you run small workloads, but you should probably investigate [Ceph](http://docs.ceph.com/docs/mimic/cephfs/), [GlusterFS](https://www.gluster.org/) or [rook.io](https://rook.io) as a way to manage distributed storage.

You should consider using [Helm](https://helm.sh/) instead of crafting your own scripts to interpolate yaml files.
