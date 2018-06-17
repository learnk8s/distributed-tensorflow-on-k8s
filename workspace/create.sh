#!/bin/bash

declare -a adjectives=("aged" "ancient" "autumn" "billowing" "bitter" "black" "blue" "bold"
  "broad" "broken" "calm" "cold" "cool" "crimson" "curly" "damp"
  "dark" "dawn" "delicate" "divine" "dry" "empty" "falling" "fancy"
  "flat" "floral" "fragrant" "frosty" "gentle" "green" "hidden" "holy"
  "icy" "jolly" "late" "lingering" "little" "lively" "long" "lucky"
  "misty" "morning" "muddy" "mute" "nameless" "noisy" "odd" "old"
  "orange" "patient" "plain" "polished" "proud" "purple" "quiet" "rapid"
  "raspy" "red" "restless" "rough" "round" "royal" "shiny" "shrill"
  "shy" "silent" "small" "snowy" "soft" "solitary" "sparkling" "spring"
  "square" "steep" "still" "summer" "super" "sweet" "throbbing" "tight"
  "tiny" "twilight" "wandering" "weathered" "white" "wild" "winter" "wispy"
  "withered" "yellow" "young")
declare -a nouns=("art" "band" "bar" "base" "bird" "block" "boat" "bonus"
  "bread" "breeze" "brook" "bush" "butterfly" "cake" "cell" "cherry"
  "cloud" "credit" "darkness" "dawn" "dew" "disk" "dream" "dust"
  "feather" "field" "fire" "firefly" "flower" "fog" "forest" "frog"
  "frost" "glade" "glitter" "grass" "hall" "hat" "haze" "heart"
  "hill" "king" "lab" "lake" "leaf" "limit" "math" "meadow"
  "mode" "moon" "morning" "mountain" "mouse" "mud" "night" "paper"
  "pine" "poetry" "pond" "queen" "rain" "recipe" "resonance" "rice"
  "river" "salad" "scene" "sea" "shadow" "shape" "silence" "sky"
  "smoke" "snow" "snowflake" "sound" "star" "sun" "sun" "sunset"
  "surf" "term" "thunder" "tooth" "tree" "truth" "union" "unit"
  "violet" "voice" "water" "waterfall" "wave" "wildflower" "wind" "wood")

random_adjective_index=$[$RANDOM % ${#adjectives[@]}]
random_noun_index=$[$RANDOM % ${#nouns[@]}]

WORD1=${adjectives[$random_adjective_index]}
WORD2=${nouns[$random_noun_index]}

ID="${WORD1}-${WORD2}"

echo "Creating ${ID}"

echo "Creating gce-nfs-disk-$ID disk"
gcloud compute disks create --size=10GB gce-nfs-disk-$ID

echo "Creating account"
kubectl create namespace ${ID}
sleep .5
sed "s/%USERNAME%/${ID}/; s/%NAMESPACE%/${ID}/" account.yaml | kubectl create -f -
SECRET=$(kubectl get sa ${ID} -n ${ID} -o json | jq -r .secrets[].name)
TOKEN=$(kubectl get secret $SECRET -n ${ID} -o json | jq -r '.data["token"]' | base64 -D)
CERTIFICATE=$(kubectl get secret $secret -n ${ID} -o json | jq -r '.items[0].data["ca.crt"]')
current_context=`kubectl config current-context`
cluster_name=`kubectl config get-contexts $current_context | awk '{print $3}' | tail -n 1`
SERVER=`kubectl config view -o jsonpath="{.clusters[?(@.name == \"$cluster_name\")].cluster.server}"`

echo "Creating workspace"
sed "s/%USERNAME%/${ID}/; s/%NAMESPACE%/${ID}/" workspace.yaml | kubectl create -f -

echo "Saving kube config file"
cat > "${ID}.config" <<-END
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: ${CERTIFICATE}
    server: ${SERVER}
  name: kubeflow-demo
contexts:
- context:
    cluster: kubeflow-demo
    namespace: ${ID}
    user: ${ID}
  name: kubeflow-demo
current-context: kubeflow-demo
kind: Config
preferences: {}
users:
- name: ${ID}
  user:
    token: ${TOKEN}
END
