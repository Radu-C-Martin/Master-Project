import casadi as cs
import numpy as np
import tensorflow as tf

from helpers import get_combined_evaluator

# Package the resulting regression model in a CasADi callback
class GPR(cs.Callback):
    def __init__(self, name, model, n_in, opts={}):
        cs.Callback.__init__(self)

        self.model = model
        self.combined_evaluator = get_combined_evaluator(model)
        self.n_in = n_in
        self.tf_var = tf.Variable(np.ones((1, self.n_in)), dtype = tf.float64)
        self.grads = None
        # Create a variable to keep all the gradient callback references
        self.refs = []

        self.construct(name, opts)
    
    # Update tf_evaluator
    def update_model(self, model):
        self.model = model
        self.combined_evaluator = get_combined_evaluator(model)

    def uses_output(self): return True
    
    # Number of inputs/outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    

    # Sparsity of the input/output
    def get_sparsity_in(self,i):
        return cs.Sparsity.dense(1,self.n_in)
    def get_sparsity_out(self,i):
        return cs.Sparsity.dense(1,1)


    def eval(self, arg):
        self.tf_var.assign(arg[0])
        preds, grads = self.combined_evaluator(self.tf_var)
        [mean, _] = preds
        self.grads = grads
        return [mean.numpy()]
    
    def has_reverse(self, nadj): return nadj==1
    def get_reverse(self, nadj, name, inames, onames, opts):
        grad_callback = GPR_grad(name, self.n_in, self.combined_evaluator)
        self.refs.append(grad_callback)
        
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return cs.Function(name, nominal_in+nominal_out+adj_seed, grad_callback.call(nominal_in), inames, onames)
        
class GPR_grad(cs.Callback):
    def __init__(self, name, n_in, combined_evaluator, opts={}):
        cs.Callback.__init__(self)
        
        self.combined_evaluator = combined_evaluator
        self.n_in = n_in
        self.tf_var = tf.Variable(np.ones((1, self.n_in)), dtype = tf.float64)

        self.construct(name, opts)

    
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    
    def get_sparsity_in(self,i):
        return cs.Sparsity.dense(1,self.n_in)
    def get_sparsity_out(self,i):
        return cs.Sparsity.dense(1,self.n_in)
    

    def eval(self, arg):
        self.tf_var.assign(arg[0])
        _, grads = self.combined_evaluator(self.tf_var)

        return [grads.numpy()]
