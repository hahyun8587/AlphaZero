import logging
import numpy as np
import tensorflow as tf
from environment import Environment
from node import Node
from replaymemory import ReplayMemory
from simulator import Simulator

class Agent():
    """Alphazero agent that trains and applies to the task.
    
    Args: 
        model (tf.keras.Model): Neural network model.
        simulator (Simulator): Simulator.
        n_action (int): The number of action that this agent can act.
        type(int): Type of agent. `0` indicates using single agented mcts 
            and `1` indicates using multi agented mcts.
    """
    
    def __init__(self, model: tf.keras.Model, simulator: Simulator, 
                 n_action: int, type: int):
        Node.configure(model, simulator, n_action, type)

    
    def train(self, steps: int = 700000, episodes: int = 100, 
              simulations: int = 800, replay_buffer_size: int = 50000,
              epochs: int = 1, examples: int = 4096, mini_batch_size: int = 32, 
              alpha: float = 1.0, epsilon: float = 0.25, 
              save_steps: int = 1000, 
              save_path: str = "bin/alphazero.keras") -> None:
        """Trains model of agent `epochs` epochs for `steps` steps. 
        The agent self-plays `episodes` episodes using mcts and trains 
        using generated episode data.
        
        Probabilities of root node is calculated as 
        `(1 - epsilon) * p_a + epsilon * eta_a`, where:
        * `p_a` are prior probabilities of the root node.
        * `eta_a` are dirichlet noise to the root node.
        
        Args: 
            steps (int, optional): The number of steps to train. 
                Setted to `700000` by default.
            episodes (int, opitional): The number of episodes 
                that the agent progresses before training in one step. 
                Setted to `100` by default.
            simulations (int, optional): The number of mcts simulations 
                conducted. Setted to `800` by default.
            replay_buffer_size (int, optional): The number of episodes to be 
                saved in replay buffer. Setted to `50000` by default.
            epochs (int, optional): The number of epochs to train in one step.
                Setted to `1` by default.
            examples (int, optional): The number of data sampled from the 
                replay buffer for training. Setted to `4096` by default.
            mini_batch_size (int, optional): Mini batch size of data 
                for training. Setted to `32` by default.
            alpha (float, optional): Dirichlet noise parameter. 
                If alpha is smaller than `1.0`, the distribution vectors 
                become near the basis vector. If alpha is `1.0`, 
                the distribution vectors are uniform. 
                If alpha is bigger than `1.0`, the distribution vectors
                become more-balanced. Alpha is preferred to be 
                an inverse proportion to the approximate number of 
                legal actions. Setted to `1.0` by default.
            epsilon (float, optional): Weight of dirichlet noise. 
                Setted to `0.25` by default.  
            save_steps (int, optional): The number of steps 
                for saving the model. Setted to `1000` by default. 
            save_path (str, optional): File path for saving model. 
                Setted to `bin/alphazero.tf` by default.
        """
        
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s %(levelname)s %(pathname)s:"
                                   "%(lineno)d] %(message)s") 
        model = Node.get_model()
        simulator = Node.get_simulator()
        replay_mem = ReplayMemory(replay_buffer_size)
        rng = np.random.default_rng()
        gamma = Node.get_gamma()
        type = Node.get_type()

        for i in range(1, steps + 1):
            logging.log(logging.INFO, "Train step %d", i)

            for j in range((i - 1) * episodes + 1, i * episodes + 1):
                logging.log(logging.INFO, "Generating episode %d", j) 

                episode_buffer = []
                episode_rewards = []
                
                s = simulator.gen_init_s()
                p, v = model(s[np.newaxis, :], False)
                p = ((1 - epsilon) * p + epsilon * rng.dirichlet(
                        [alpha for _ in range(p.shape[0])]))
                
                root = Node(s, p, False)
                
                k = 1

                while 1:
                    pi = root.mcts(simulations)
                    a = rng.choice(pi.shape[0], p=pi)

                    logging.log(logging.INFO, 
                                "Move %d of episode %d: %d", k, j, a) 

                    episode_buffer.append([j, root.get_s(), pi])
                    episode_rewards.append(root.get_reward(a))
                    
                    root = root.get_child(a)
                    
                    k += 1
                    
                    if root.get_is_terminal():
                        break
                
                traj_len = len(episode_buffer)
                
                episode_buffer[traj_len - 1].append(episode_rewards[traj_len - 1])
                
                for k in range(traj_len - 2, -1, -1):
                    v = episode_buffer[k + 1][3] if type == 0 \
                            else -episode_buffer[k + 1][3]
                    episode_buffer[k].append(episode_rewards[k] + gamma * v)
        
                replay_mem.extend(episode_buffer)

            logging.log(logging.INFO, "Training with samples at step %d", i)
            
            x, y = replay_mem.sample(examples)
            model.fit(x, y, epochs=epochs, batch_size=mini_batch_size)
            
            if i % save_steps == 0:
                logging.log(logging.INFO, "Saving model at step %d...", i)
                model.save(save_path)    
            
                                   
    def apply(self, env: Environment, *args, **kwargs) -> None:
        """Applies this instance to the task.
        
        Args: 
            env (Environment): Environment of the task.
        """
