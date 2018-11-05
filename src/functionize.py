import ipdb
import torch
from torch import nn
from torch.nn import functional as F
from .allennlp_mods.elmo_text_field_embedder import ElmoTextFieldEmbedder, ElmoTokenEmbedderWrapper

class Scope(object):
    pass

import sys
from collections import OrderedDict

PY2 = sys.version_info[0] == 2
_internal_attrs = {'_backend', '_parameters', '_buffers', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_modules', 'forward'}


class Scope(object):
    def __init__(self):
        self._modules = OrderedDict()


def _make_functional(module, params_box, params_offset, trg_func='forward', parent=None):
    ''' TODO(Alex): something like create a functional version of module
    1) Create a Scope object
    2) "Copy" the module by setting Scope's attributes to module attributes
        a) Recursively apply _make_functional to named children
    3) Create functional fmodule and return it.
        fmodule
    '''
    self = Scope()
    num_params = len(module._parameters)
    param_names = list(module._parameters.keys())
    if trg_func == 'forward':
        func = type(module).forward.__func__ if PY2 else type(module).forward

        # copy internal attributes
        for name, attr in module.__dict__.items():
            if name in _internal_attrs:
                continue
            if isinstance(attr, ElmoTextFieldEmbedder):
                ipdb.set_trace()
            setattr(self, name, attr)

        child_params_offset = params_offset + num_params

        # copy class attributes (?) and methods
        module_cls = type(module)
        for name, attr in module_cls.__dict__.items():
            if name.startswith("__") or name in _internal_attrs: # NOTE: 'forward' added to _internal_attrs
                continue
            if callable(attr):
                _, func_ffunc, _ = _make_functional(module, params_box, child_params_offset, name, self)
                setattr(self, name, func_ffunc)
            else:
                setattr(self, name, attr)

        # also need to walk the class ancestor tree...
        # we do hacky shit instead
        if isinstance(module, ElmoTextFieldEmbedder):
            print(" ### In an elmotextfieldembedder!")
            pass #ipdb.set_trace()
        if issubclass(module_cls, nn.Module):
            for name, attr in nn.Module.__dict__.items():
                if name.startswith("__") or name in _internal_attrs:
                    continue
                if callable(attr):
                    _, func_ffunc, scope_internal = _make_functional(module, params_box, child_params_offset, name, self)
                    setattr(self, name, func_ffunc)
                else:
                    setattr(self, name, attr)
                if 'named_children' in name:
                    print("!!! Added %s to %s" % (name, module_cls.__name__))
                    #ipdb.set_trace()

        # Jiant specific: copy each forward function
        if hasattr(module, '_forward_funcs'):
            for name, ffunc in module._forward_funcs.items():
                # want to create functional versions of these methods and set them as the attribute
                _, func_ffunc, scope_internal = _make_functional(module, params_box, child_params_offset, name, self)
                setattr(self, name, func_ffunc)

        child_params_offset = params_offset + num_params
        for name, child in module.named_children():
            if isinstance(child, ElmoTextFieldEmbedder):
                ipdb.set_trace()
            child_params_offset, fchild, scope_internal = _make_functional(child, params_box, child_params_offset)
            self._modules[name] = fchild

            setattr(self, name, fchild)

    else: # when we want to copy *not* module.forward
        child_params_offset = params_offset
        func = getattr(type(module), trg_func)

    def fmodule(*args, **kwargs):
        ''' Create a closure using param_names, params_offset, num_params, forward.
        I think because params_box is a list, it persists in memory and
        the parameters will be there when it is modified by fmodule
        when the parameters are popped and set.  '''
        for name, param in zip(param_names, params_box[0][params_offset:params_offset + num_params]):
            setattr(self, name, param)

        #kwargs.setdefault('name', 'forward')
        #fname = kwargs['name']
        #del kwargs['name']
        #return getattr(self, func)(self, *args, **kwargs)
        #try:
        if parent is not None:
            return func(parent, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
        #except Exception as e:
        #    ipdb.set_trace()

    #if hasattr(module, '_pair_sentence_forward'):
    #    ipdb.set_trace()
    return child_params_offset, fmodule, self


def make_functional(module):
    ''' Given an nn.Module, return a functional version, fmodule, of that module.
    The functional expects the input expected by module.forward, but also
    accepts an additional keyword `params`, which is used ...
    '''

    # params_box holds parameters
    # when constructing fmodule, params_box = [None]
    params_box = [None]
    _, fmodule_internal, scope_internal = _make_functional(module, params_box, 0)

    def fmodule(*args, **kwargs):
        ''' Functional that actually gets returned
        Will get called as forward(x, params=theta) '''
        params_box[0] = kwargs.pop('params')
        return fmodule_internal(*args, **kwargs)

    return fmodule


