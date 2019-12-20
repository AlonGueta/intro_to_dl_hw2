from network import *

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        #Create Workers and set jobs
        
        #Call the parent's fit 
        super().fit(training_data, validation_data)
        
        #Stop Workers
        
        raise NotImplementedError("To be implemented")
        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return batches created by workers
        '''
        raise NotImplementedError("To be implemented")

    
