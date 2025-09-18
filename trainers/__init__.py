"""FedETuning's trainers registry in trainer.__init__.py -- IMPORTANT!"""

from trainers.FedBaseTrainer import BaseTrainer
from run.fedavg.trainer import FedAvgTrainer

__all__ = [
    "BaseTrainer",
    "FedAvgTrainer"
]
