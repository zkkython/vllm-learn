import asyncio
from dataclasses import dataclass


@dataclass
class SimpleRequest:
    request_id: str
    prompt: str


@dataclass
class SimpleOutput:
    request_id: str
    text: str
    finished: bool


class SimpleOutputCollector:
    def __init__(self):
        self._queue = asyncio.Queue()

    def put(self, output):
        self._queue.put_nowait(output)

    async def get(self):
        return await self._queue.get()

    def get_nowait(self):
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


class SimpleAsyncLLM:
    def __init__(self):
        self.request_states = {}

    async def generate(self, prompt: str, request_id: str):
        collector = SimpleOutputCollector()
        request = SimpleRequest(request_id=request_id, prompt=prompt)
        self.request_states[request_id] = collector

        print(f"[AsyncLLM] 发送请求 {request.request_id} -> EngineCore")

        loop = asyncio.get_running_loop()
        loop.call_later(0.1, collector.put, SimpleOutput(request_id, "Hello", False))
        loop.call_later(0.2, collector.put, SimpleOutput(request_id, "Hello World!", True))

        finished = False
        while not finished:
            out = collector.get_nowait() or await collector.get()
            finished = out.finished
            yield out


async def main():
    llm = SimpleAsyncLLM()
    async for output in llm.generate("Say hello", "req_001"):
        print(f"[Output] {output.text} (finished={output.finished})")


asyncio.run(main())
