import tensorflow as tf
from collections import deque

class ReplayMemory(deque):
    """Replay memory that saves data generated from each episode.
    
    Args:
        thresh (int): The number of episode to be saved.
    """    
    
    #instance variables
    _thresh: int
    _min_ep: int
    
    def __init__(self, thresh: int):
        super(ReplayMemory, self).__init__()

        self._thresh = thresh
        self._min_ep = 0
    
    
    def extend(self, data: list) -> None:
        """Saves `data` to this instance.
        
        All the data from least number of episode are discarded if the number of 
        episodes in this instance exceeds `thresh`.
        
        Args:
            data (list): The data to be saved. The data should be consists of
                list of [episode number, s, pi, z], where:
            * `s` indicates state
            * `pi` indicates policy from mcts
            * `z` indicates real result value
        """
    
        if data[0][0] > self._thresh:
            while 1:
                if len(self) == 0:
                    break
                
                x = self.popleft()
                
                if x[0] > self._min_ep:
                    self.appendleft(x)
                    
                    break
            
            self._thresh += 1
            self._min_ep += 1
        
        super(ReplayMemory, self).extend(data)
        
     
    def sample(self, n: int) -> tuple:
        """Samples random `n` data from this instance.

        Args:
            n (int): The number of samples
        
        Returns:
            tuple: Sampled data.
        """

        x = []
        y_pi = []
        y_z = []
        
        sampled_idx = tf.random.uniform([n], maxval=len(self), dtype=tf.int32)

        for i in sampled_idx:
            x.append(self[i][1])
            y_pi.append(self[i][2])
            y_z.append(self[i][3])
            
        return (tf.constant(x, dtype=tf.float64), 
                [tf.constant(y_pi), tf.constant(y_z, dtype=tf.float64)])         
    