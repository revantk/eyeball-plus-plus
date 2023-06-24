import pytest
from unittest.mock import Mock, patch
from eyeball_pp.recorders import EvalRecorder, MemoryRecorder
with patch.dict('sys.modules', {'openai': Mock()}):

class TestEvaluator:

    def setup_method(self):
        with patch('eyeball_pp.eval.comparators', autospec=True):
            self.evaluator = Evaluator()
            self.evaluator.recorder = Mock(spec=EvalRecorder)