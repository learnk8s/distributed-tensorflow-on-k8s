#!/bin/bash

declare -a hidden_layers=(1 2 3)
declare -a learning_rates=("0.1" "0.01" "0.001")

for learning_rate_index in "${!learning_rates[@]}"
do
   for hidden_layers_index in "${!hidden_layers[@]}"
   do
      ID="rate${learning_rate_index}-layer${hidden_layers_index}"
      LEARNING_RATE=${learning_rates[$learning_rate_index]}
      HIDDEN_LAYERS_COUNT=${hidden_layers[$hidden_layers_index]}
      sed "\
         s/%ID%/${ID}/; \
         s/%LEARNING_RATE%/${LEARNING_RATE}/; \
         s/%HIDDEN_LAYERS_COUNT%/${HIDDEN_LAYERS_COUNT}/" \
         tfjob-templated.yaml | kubectl create -f -
      sleep .5
   done
done
