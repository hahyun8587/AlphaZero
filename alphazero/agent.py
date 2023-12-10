import numpy as np
import tensorflow as tf
from .environment import Environment
from .simulator import Simulator

class Agent():
    model: tf.keras.Model
    replay_mem: np.ndarray
    
    def __init__(self, model: tf.keras.Model):
        """_summary_

        Args:
            model (tf.keras.Model): _description_
            type (int): 
        """

    
    def train(self, sim: Simulator, steps: int) -> None:
        """_summary_

        Args:
            sim (Simulator): 
            steps (int): 
        """


    def apply(self, env: Environment, *args, **kwargs) -> None:
        """Applies this instance to the task.
        
        Args: 
            env (Environment): Environment of the task.
        """
        
        