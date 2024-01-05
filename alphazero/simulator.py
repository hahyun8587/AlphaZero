import numpy as np

class Simulator():
    """Interface of simulator for mcts and training.
    Class that implements this interface should implement 
    `gen_init_s()`, `simulate()`, and `is_terminal()`.
    """

    def gen_init_s(self) -> np.ndarray:
        """Generates initial state. 
        
        Returns:
            np.ndarray: The initial state.
            
        Raises: 
            NotImplementedError: Raises when class that implemented 
                this interface did not implement this method.
        """

        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method gen_init_s()')
      
        
    def simulate(self, s: np.ndarray, a: int) -> np.ndarray | None:
        """Simulates current state `s` with selected action `a`.

        Args:
            s (np.ndarray): The current state.
            a (int): The selected action.
                
        Returns:
            np.ndarray or None: Next state if `a` is a valid action, 
                `None` otherwise.

        Raises:
            NotImplementedError: Raises when class that implemented
                this interface did not implement this method.
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method simulate()')
        
        
    def is_terminal(self, s: np.ndarray) -> bool:
        """Checks whether state `s` is terminal state or not.

        Args:
            s (np.ndarray): The state to be checked.
            
        Returns:
            bool: `True` if the given state is terminal state, `False` otherwise. 
            
        Raises: 
            NotImplementedError: Raises when class that implemented 
                this interface did not implement this method.
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + "did not implement method is_terminal()")