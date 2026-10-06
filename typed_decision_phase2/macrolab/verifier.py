"""Only support feedback leaves this boundary. Hidden evaluator is a separate command."""
from interpreter import run
from oracle import target
class SupportVerifier:
    def __init__(self,program,support_inputs):self._program=program;self._inputs=support_inputs;self.calls=0
    def verify(self,candidate):
        self.calls+=1
        if self.calls>4:raise ValueError('verifier budget exceeded')
        n=0;counter=None
        for x in self._inputs:
            expected=target(self._program,x);actual=run(candidate,x)
            if actual==expected:n+=1
            elif counter is None:counter={'input':x,'expected':expected,'actual':actual}
        return {'passed':n,'total':5,'counterexample':counter}
