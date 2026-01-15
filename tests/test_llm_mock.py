import unittest

from addm.llm import LLMService


class TestLLMMock(unittest.IsolatedAsyncioTestCase):
    async def test_batch_call(self):
        llm = LLMService()
        llm.configure(provider="mock", max_concurrent=2)
        llm.set_mock_responder(lambda messages: "ok")
        batch = [
            [{"role": "user", "content": "hi"}],
            [{"role": "user", "content": "hello"}],
        ]
        outputs = await llm.batch_call(batch)
        self.assertEqual(outputs, ["ok", "ok"])
