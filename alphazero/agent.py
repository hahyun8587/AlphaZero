import numpy as np
import tensorflow as tf

from .environment import Environment
from .node import Node
from .replaymemory import ReplayMemory
from .simulator import Simulator

class Agent():
    """Alphazero agent that trains and applies to the task.
    
    Args: 
        model (tf.keras.Model): Neural network model.
        simulator (Simulator): Simulator.
        n_a (int): The number of actions.
    """
    
    #instance variables
    _model: tf.keras.Model
    _simulator: Simulator
    _n_a: int
    
    def __init__(self, model: tf.keras.Model, simulator: Simulator, 
                 n_a: int, type: int):
        self._model = model
        self._simulator = simulator
        self._n_a = n_a
        
        Node.configure(model, simulator, n_a, 1)

    
    def train(self, steps: int=700000, n_ep_train: int=10000, 
              batch_size: int=4096, n_train_save: int=1000, 
              n_replay_ep: int=50000) -> None:
        """Trains neural network of this instance per `n_ep_train` episodes 
        for `steps` steps.
        
        `n_replay_ep` numbers of episodes are saved in replay memory. 
    
        Args: 
            steps (int, optional): The number of steps to train. 
                Setted to 700000 by default.
            n_ep_train (int, opitional): The number of episode played 
                per training.
            batch_size(int, optional): Batch size of data 
                per one training step. Setted to 4096 by default.
            n_train_save(int, optional): The number of training per save.
                Setted to 1000 by default. 
            n_replay_ep (int, optional): The number of episodes to be saved 
                in replay memory. Setted to 50000 by default.
        """
        
        replay_mem = ReplayMemory(n_replay_ep)
        
        for i in range(1, steps  * n_ep_train + 1):
            ep_mem = []
            z = None
            
            root_s = self._simulator.gen_init_s()
            root_s = root_s.astype(np.float64)
            root_p, root_v = self._model(root_s, False)
            root_p = root_p.numpy().reshape(-1)
            root_v = root_v.numpy().astype(int).reshape(-1)
            root = Node(root_s, root_v, root_p, False) 
            
            while 1:
                pi = root.mcts()
                
                ep_mem.append([i, root.get_s(), pi])
                
                root = root.get_child(np.random.choice(self._n_a, p=pi))

                z = root.get_is_terminal()
                
                if z:             
                    break
    
            if root.get_s()[1][0][0] == 1:
                z = -z     
            
            for data in ep_mem:
                data.append(z)
                z = -z
            
            replay_mem.extend(ep_mem)
            
            if i % n_ep_train == 0:
                x, y = replay_mem.sample(batch_size)
                
                self._model.fit(x, y, batch_size=batch_size)       
                
            if i % n_train_save * n_ep_train == 0:
                self._model.save('bin/alphasix-{i}.tf')    

                                   

    def apply(self, env: Environment, *args, **kwargs) -> None:
        """Applies this instance to the task.
        
        Args: 
            env (Environment): Environment of the task.
        """
        
        