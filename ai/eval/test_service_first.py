import copy
import unittest
from pathlib import Path
from core.mock import MockLLM
from pipeline import Pipeline
from confirm.questions import auto_finish

class SpyLLM(MockLLM):
    def __init__(self): self.calls=[]
    def chat_json(self, system, user, task=None, retries=1):
        self.calls.append(task)
        return super().chat_json(system, user, task, retries)

class ServiceFirstTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe=Pipeline(mock=True,services_path=Path(__file__).resolve().parents[1]/"data/sample_services.csv")
    def setUp(self):
        self.pipe.llm=SpyLLM()
        self.session=auto_finish(self.pipe.prepare("월세와 생활비가 걱정돼요"))
        self.pipe.llm.calls.clear()
    def test_fast_results_skip_generation_and_preserve_order(self):
        fast=self.pipe.search(self.session,explain_mode="source_first")
        self.assertEqual(self.pipe.llm.calls,[])
        self.assertTrue(all(c["explanation_pending"] for c in fast["cards"]))
        normal=self.pipe.search(self.session)
        self.assertEqual([c["servId"] for c in fast["cards"]],[c["servId"] for c in normal["cards"]])
        self.assertEqual(self.pipe.llm.calls.count("explain"),len(normal["cards"]))
        self.assertIn("explanation_ms",fast["timings_ms"])
    def test_explanation_only_on_request_and_reuses_completed_card(self):
        out=self.pipe.search(self.session,explain_mode="source_first")
        sid=out["cards"][0]["servId"]
        card=self.pipe.explain_service(self.session,out,sid)
        self.assertEqual(self.pipe.llm.calls,["explain"])
        self.assertFalse(card["explanation_pending"])
        out["cards"][0]=card
        self.assertEqual(self.pipe.explain_service(self.session,out,sid),card)
        self.assertEqual(self.pipe.llm.calls,["explain"])
    def test_crisis_stale_conditions_and_unknown_service_are_rejected(self):
        out=self.pipe.search(self.session,explain_mode="source_first")
        sid=out["cards"][0]["servId"]
        crisis={**self.session,"status":"crisis_stop"}
        with self.assertRaises(ValueError): self.pipe.explain_service(crisis,out,sid)
        changed=copy.deepcopy(self.session);changed["confirmed_profile"]["age"]=88
        with self.assertRaises(ValueError): self.pipe.explain_service(changed,out,sid)
        with self.assertRaises(ValueError): self.pipe.explain_service(self.session,out,"not-returned")
        self.assertEqual(self.pipe.llm.calls,[])

if __name__=="__main__": unittest.main()
