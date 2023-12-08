import numpy as np

class Environment():
    """Interface that represents environment.
    Class that implements this interface should implement 
    gen_init_s() and response().
    """
    
    def get_init_s(self, *args, **kwargs):
        """Generates initial state. 

        This method is used for applying model of agent.
        
        Raises: 
            NotImplementedError: Raises when class that implemented 
                this interface did not implement this method.
        
        Returns:
            np.ndarray: The initial state.
        """

        raise NotImplementedError('class ' + self.__class__.__name__ 
                                  + ' did not implement method gen_init_s()')
    
    
    def response(self, s: np.ndarray, a: int):
        """Responses current state with selected action 
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
                                  + ' did not implement method response()')
    