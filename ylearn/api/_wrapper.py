from ylearn.effect_interpreter.policy_interpreter import PolicyInterpreter


def _transform(fn):
    def _exec(obj, data, *args, **kwargs):
        assert isinstance(obj, WrappedPolicyInterpreter)

        transformer = getattr(obj, 'transformer', None)
        if transformer is not None and data is not None:
            data = transformer.transform(data)

        return fn(obj, data, *args, **kwargs)

    return _exec


class WrappedPolicyInterpreter(PolicyInterpreter):
    @_transform
    def decide(self, data):
        return super().decide(data)

    @_transform
    def predict(self, data):
        return super().predict(data)

    @_transform
    def interpret(self, data=None):
        return super().interpret(data)
