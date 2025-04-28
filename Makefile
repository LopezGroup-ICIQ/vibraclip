# Copyright (c) 2025 - Institute of Chemical Research of Catalonia (ICIQ)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

wandb-sync:
	wandb sync --sync-all

clean-data:
	rm -rf ./data/processed

clean-pycache:
	find . -name "*pycache*" -type d -exec rm -rf {} +

clean-wandb:
	rm -rf wandb

clean-all:
	find . -name "*pycache*" -type d -exec rm -rf {} +
	rm -rf wandb
