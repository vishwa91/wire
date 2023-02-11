#!/usr/bin/env python

# Somewhat hacky way of importing
from . import gauss
from . import mfn
from . import relu
from . import siren
from . import wire
from . import wire2d

model_dict = {'gauss': gauss,
              'mfn': mfn,
              'relu': relu,
              'siren': siren,
              'wire': wire,
              'wire2d': wire2d}

def get_INR(nonlin, in_features, hidden_features, hidden_layers,
            out_features, outermost_linear=True, first_omega_0=30,
            hidden_omega_0=30, scale=10, pos_encode=False,
            sidelength=512, fn_samples=None, use_nyquist=True):
    '''
        Function to get a class instance for a given type of
        implicit neural representation
        
        Inputs:
            nonlin: One of 'gauss', 'mfn', 'posenc', 'siren',
                'wire', 'wire2d'
            in_features: Number of input features. 2 for image,
                3 for volume and so on.
            hidden_features: Number of features per hidden layer
            hidden_layers: Number of hidden layers
            out_features; Number of outputs features. 3 for color
                image, 1 for grayscale or volume and so on
            outermost_linear (True): If True, do not apply nonlin
                just before output
            first_omega0 (30): For siren and wire only: Omega
                for first layer
            hidden_omega0 (30): For siren and wire only: Omega
                for hidden layers
            scale (10): For wire and gauss only: Scale for
                Gaussian window
            pos_encode (False): If True apply positional encoding
            sidelength (512): if pos_encode is true, use this 
                for side length parameter   
            fn_samples (None): Redundant parameter
            use_nyquist (True): if True, use nyquist sampling for 
                positional encoding
        Output: An INR class instance
    '''

    inr_mod = model_dict[nonlin]
    model = inr_mod.INR(in_features,
                        hidden_features,
                        hidden_layers,
                        out_features,
                        outermost_linear,
                        first_omega_0,
                        hidden_omega_0,
                        scale,
                        pos_encode,
                        sidelength,
                        fn_samples,
                        use_nyquist)
    
    return model