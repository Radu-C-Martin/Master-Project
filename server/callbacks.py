import casadi as cs
import numpy as np
import tensorflow as tf

# Package the resulting regression model in a CasADi callback
class GPR(cs.Callback):
    def __init__(self, name, model, opts={}):
        cs.Callback.__init__(self)

        self.model = model
        self.n_in = model.data[0].shape[1]
        # Create a variable to keep all the gradient callback references
        self.refs = []

        self.construct(name, opts)
    
    # Number of inputs/outputs
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    

    # Sparsity of the input/output
    def get_sparsity_in(self,i):
        return cs.Sparsity.dense(1,self.n_in)
    def get_sparsity_out(self,i):
        return cs.Sparsity.dense(1,1)


    def eval(self, arg):
        inp = np.array(arg[0])
        inp = tf.Variable(inp, dtype=tf.float64)
        [mean, _] = self.model.predict_f(inp)
        return [mean.numpy()]
    
    def has_reverse(self, nadj): return nadj==1
    def get_reverse(self, nadj, name, inames, onames, opts):
        grad_callback = GPR_grad(name, self.model)
        self.refs.append(grad_callback)
        
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return cs.Function(name, nominal_in+nominal_out+adj_seed, grad_callback.call(nominal_in), inames, onames)
        
class GPR_grad(cs.Callback):
    def __init__(self, name, model, opts={}):
        cs.Callback.__init__(self)  
        self.model = model
        self.n_in = model.data[0].shape[1]

        self.construct(name, opts)

    
    def get_n_in(self): return 1
    def get_n_out(self): return 1
    
    def get_sparsity_in(self,i):
        return cs.Sparsity.dense(1,self.n_in)
    def get_sparsity_out(self,i):
        return cs.Sparsity.dense(1,self.n_in)


    def eval(self, arg):
        inp = np.array(arg[0])
        inp = tf.Variable(inp, dtype=tf.float64)
        
        with tf.GradientTape() as tape:
            preds = self.model.predict_f(inp)

        grads = tape.gradient(preds, inp)
        return [grads.numpy()]
