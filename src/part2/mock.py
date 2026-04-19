import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class MockEngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    finished: bool


@dataclass
class MockEngineCoreOutputs:
    outputs: list[MockEngineCoreOutput]
    timestamp: float = field(default_factory=time.time)


@dataclass
class MockRequestOutput:
    request_id: str
    text: str
    finished: bool


class MockOutputCollector:
    def __init__(self):
        self.output = None
        self.ready = asyncio.Event()

    def put(self, output):
        self.output = output
        self.ready.set()

    async def get(self):
        while self.output is None:
            await self.ready.wait()
        out = self.output
        self.output = None
        self.ready.clear()
        return out

    def get_nowait(self):
        out = self.output
        if out is not None:
            self.output = None
            self.ready.clear()
        return out


class MockEngineCore:
    def __init__(self):
        self._output_queue = asyncio.Queue()
        self._requests = {}

    def add_request(self, request_id: str, prompt: str):
        self._requests[request_id] = {"prompt": prompt, "tokens_generated": 0}
        asyncio.get_running_loop().call_later(0.05, self._step)

    def _step(self):
        outputs = []
        for req_id, state in list(self._requests.items()):
            state["tokens_generated"] += 1
            finished = state["tokens_generated"] >= 3
            outputs.append(
                MockEngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=[100 + state["tokens_generated"]],
                    finished=finished,
                )
            )
            if finished:
                del self._requests[req_id]

        if outputs:
            self._output_queue.put_nowait(MockEngineCoreOutputs(outputs=outputs))
        if self._requests:
            asyncio.get_running_loop().call_later(0.05, self._step)

    async def get_output(self):
        return await self._output_queue.get()


class MockAsyncLLM:
    def __init__(self):
        self.engine_core = MockEngineCore()
        self.request_states = {}
        self._handler_started = False
        self.vocab = {101: "Hello", 102: " World", 103: "!"}

    def _start_output_handler(self):
        if self._handler_started:
            return
        self._handler_started = True
        asyncio.create_task(self._output_handler())

    async def _output_handler(self):
        while True:
            outputs = await self.engine_core.get_output()
            for output in outputs.outputs:
                collector = self.request_states.get(output.request_id)
                if collector is None:
                    continue
                collector.put(
                    MockRequestOutput(
                        request_id=output.request_id,
                        text=self.vocab.get(output.new_token_ids[0], "?"),
                        finished=output.finished,
                    )
                )
                if output.finished:
                    del self.request_states[output.request_id]

    async def generate(self, prompt: str, request_id: str):
        self._start_output_handler()
        collector = MockOutputCollector()
        self.request_states[request_id] = collector
        self.engine_core.add_request(request_id, prompt)

        finished = False
        while not finished:
            out = collector.get_nowait() or await collector.get()
            finished = out.finished
            yield out


async def main():
    llm = MockAsyncLLM()
    async for output in llm.generate("Say hello", "req_001"):
        print(output.text, output.finished)


asyncio.run(main())
