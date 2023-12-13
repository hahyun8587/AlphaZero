import tensorflow as tf
import numpy as np
from math import sqrt
from .simulator import Simulator

class Node():
    """Node of mcts tree.
    Configuration of `Node` should be done 
    by using `Node.configure()` before the any instantiation.
    
    Args:
        s (np.ndarray): State of this instance.
        v_s (float): Value of this instance's state.
        p_s (np.ndarray): Prior possiblities of selecting action 
            at this instance's state.
      
    Attributes: 
        model (tf.keras.Model): Neural network model.
        simulator (Simulator): Simulator that uses state transition dynamics.
        n_a (int): Number of actions of this class.
        type (int): Type of `Node`. 0 if this class indicates single-agented, 
            1 if this class indicates double-agented.
    """
    
    #class variables
    _model: tf.keras.Model
    _simulator: Simulator
    _n_a: int
    _type: int 
    
    #instance variables
    _s: np.ndarray
    _v_s: float
    _p_s: np.ndarray
    _q_s: np.ndarray
    _w_s: np.ndarray
    _n_s: np.ndarray
    _children: np.ndarray
    _is_terminal: int
   
    def __init__(self, s: np.ndarray, v_s: float, p_s: np.ndarray, 
                 is_terminal: bool):
        self._s = s
        self._v_s = v_s
        self._p_s = p_s
        self._q_s = np.zeros(self._n_a, dtype=float)
        self._w_s = np.zeros(self._n_a, dtype=float)
        self._n_s = np.zeros(self._n_a, dtype=int)
        self._children = np.full(self._n_a, fill_value=None, dtype=Node)
        self._is_terminal = is_terminal
 
 
    @classmethod
    def configure(cls, model: tf.keras.Model, simulator: Simulator, 
                  n_a: int, type: int) -> None:
        """Configures Node with the given `model`, `simulator`, 
        `n_a`, and `type`.

        Args:
            model (tf.keras.Model): Neural network model.
            simulator (Simulator): Simulator. 
            n_a (int): Number of actions.
            type (int): Type of Node.
        """
        
        cls._model = model
        cls._simulator = simulator
        cls._n_a = n_a
        cls._type = type
    
    
    def mcts(self, n_sim: int=800, temp: float=1.0) -> np.ndarray:
        """Applies `n_sim` number of simulations of monte-carlo tree search 
        to this instance and calculates policy with the accumulated statistics.

        Simulates with `_expand_and_backup()` n_sim times 
        and calculates policy with `_calc_policy()`.
        
        Args: 
            n_sim (int, optional): Number of simulations to get policy 
                at state of this instance. It is setted to 800 by default.
            temp (float, optional): Temperature variable for calculating policy. 
                It should be in range of (0, 1]. 
                The larger temp gets, it flattens distribution of the policy 
                and leads more exploration.
                The smaller temp gets, it leads more exploitation.
                It is setted to 1.0 by default.
        
        Returns:
            np.ndarray: The policy of this instance.
        """
        
        for _ in range(n_sim):
            self._expand_and_backup()
        
        return self._calc_policy(temp)
        
        
    def _select(self, c_puct: float=4.0) -> int:
        """Selects action at the state of this instance accroding to puct   
        algorithm.
        
        The equation of puct algorithm is \\
        `a = argmax_a(Q(s, a) + U(s, a))`, where:
        * `U(s, a) = c_puct * P(s, a) * 
        sqrt(sum_b(N(s, b))) / (1 + N(s, a))`
        
        Args:
            c_puct (int, optional): puct constant. 
                Bigger c_puct value makes exploration to be more considered and 
                smaller c_puct value makes exploitation to be more considered.
                Approximately value of 4 is optimal.\n
                https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
        
        Returns:
            Integer that indicates the selected action.
        """

        return (self._q_s + c_puct * self._p_s 
                * sqrt(self._n_s.sum(axis=0)) / (self._n_s + 1)).argmax(axis=0)
       
        
    def _update(self, v_exp: float, i: int) -> None:
        """Updates statistics of this instance's `i + 1`th edge 
        with the given `v_exp` which is represented 
        in perspective of this instance.
        
        Statistics of `i + 1`th edge is updated as following: 
        * `W(s, a) = W(s, a) + V(s')`
        * `N(s, a) = N(s, a) + 1`
        * `Q(s, a) = W(s, a) / N(s, a)`, 
            
        where V(s') represents value of expanded node's state 
        in perspective of this instance.

        Args:
            v_exp (float): Value of expanded node's state.
            i (int): Index of this instance's edge to be updated.
        """
        
        self._w_s[i] += v_exp
        self._n_s[i] += 1
        self._q_s[i] = self._w_s[i] / self._n_s[i] 
        
          
    def _expand_and_backup(self) -> float:
        """Expands new node to leaf node of this instance 
        and backups all the children of this instance and itself
        with value of the new node's state.
        
        Returns:
            float: value of the expanded node's state 
                in perspective of this instance.
        """
        
        if self._is_terminal:
            return self._v_s
        
        a = self._select()
        
        if self._children[a] == None:
            s_exp = self._simulator.simulate(self._s, a)
            
            if s_exp is None:
                v_exp = 1.0
                self._children[a] = Node(None, v_exp, None, True)
            elif self._simulator.is_terminal(s_exp, a):
                v_exp = -1.0
                self._children[a] = Node(s_exp, v_exp, None, True)
            else:     
                p_exp, v_exp = self._model(s_exp[np.newaxis, :].astype(np.float64), 
                                           False)
                p_exp = p_exp.numpy().reshape(-1)
                v_exp = v_exp.numpy().astype(int).reshape(-1)
                self._children[a] = Node(s_exp, v_exp, p_exp, False)
        else: 
            v_exp = self._children[a]._expand_and_backup()        
        
        if self._type == 1:
            v_exp = -v_exp 
        
        self._update(v_exp, a)
            
        return v_exp
        
        
    def _calc_policy(self, temp: float) -> np.ndarray:
        """Calculates policy of this instance with the given `temp`.
    
        The equation of the policy is \\
        `pi = pow(N(s, .), 1 / tau) / sum_b(pow(N(s, b), 1 / tau))`, where:
        * `tau` indicates the temperature variable. 

        Args:
            temp (float): The temperature variable.  

        Returns:
            np.ndarray: The policy of this instance.
        """   

        n_pow = self._n_s ** (1 / temp) 

        return n_pow / n_pow.sum(axis=0)
                
         
    def get_s(self) -> np.ndarray:
        return self._s     
           
           
    def get_child(self, i: int) -> 'Node': 
        """Gets `i + 1`th child of this instance. 

        Args:
            i (int): Index of child to be obtained.
        
        Returns:
            Node: The `i + 1`th child.
        """     
        
        return self._children[i]


    def get_is_terminal(self) -> int:
        return self._is_terminal
    