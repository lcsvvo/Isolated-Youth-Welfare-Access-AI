import unittest
from core.models import LLM

class Tensor:
    shape = (1, 3)

class Inputs(dict):
    def to(self, device):
        self.device = device
        return self

class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.options = kwargs
        return Inputs(input_ids=Tensor(), attention_mask=Tensor())
    def decode(self, tokens, **kwargs):
        return str(tokens)

class Model:
    device = "cuda:0"
    def generate(self, **kwargs):
        self.received = kwargs
        return [[1, 2, 3, 9, 8]]

class RealModelAdapterTests(unittest.TestCase):
    def test_generate_gets_input_ids_and_attention_mask(self):
        llm = object.__new__(LLM)
        llm.tok, llm.model = Tokenizer(), Model()
        self.assertEqual(llm.chat("system", "user"), "[9, 8]")
        self.assertTrue(llm.tok.options["return_dict"])
        self.assertIn("attention_mask", llm.model.received)
        self.assertEqual(llm.model.received["max_new_tokens"], 512)
        self.assertFalse(llm.model.received["do_sample"])

if __name__ == "__main__":
    unittest.main()
