import numpy as np

class Simulator():
    """Interface that simulates state with selected action.
    Class that implements this interface should implement 
    gen_init_s(), simulate(), and is_terminal().
    """

    def gen_init_s(self) -> np.ndarray:
        """Generates initial state. 

        This method is used for training model of agent.
        
        Raises: 
            NotImplementedError: Raises when class that implemented 
                this interface did not implement this method.
        
        Returns:
            np.ndarray: The initial state.
        """

        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method gen_init_s()')
      
        
    def simulate(self, s: np.ndarray, a: int) -> np.ndarray:
        """Simulates current state with selected action 
        by using the given s and a.

        Args:
            s (np.ndarray): Current state.
            a (int): Selected action.
        
        Raises:
            NotImplementedError: Raises when class that implemented
                this interface did not implement this method.
        
        Returns:
            np.ndarray: Next state.
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method simulate()')
        
        
    def is_terminal(self, s: np.ndarray) -> bool:
        """Checks whether the given state s is terminal state or not.

        Args:
            s (np.ndarray): The state to be checked.

        Returns:
            bool: true if the given state is terminal state, false otherwise.
        """
        
        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + "did not implement method is_terminal()")