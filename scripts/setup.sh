#!/bin/bash

echo "Setting up SUMO and dependencies..."

sudo add-apt-repository ppa:sumo/stable -y
sudo apt-get update -y
sudo apt-get install sumo sumo-tools sumo-doc -y

export SUMO_HOME="/usr/share/sumo"

if ! grep -q 'SUMO_HOME="/usr/share/sumo"' ~/.bashrc; then
    echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
fi

echo "SUMO installation complete. SUMO is installed in $SUMO_HOME."

