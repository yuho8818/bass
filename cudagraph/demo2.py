import torch
import torch.nn as nn
import os
import nvtx
import time
import gc
from torch.profiler import profile, record_function, ProfilerActivity

D_in = 32
D_out = 32
torch.manual_seed(1)


class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}

    def capture(self, x, y, z):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        self.cuda_graph.enable_debug_mode()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x, y, z)
        torch.cuda.synchronize()
        self.cuda_graph.debug_dump("graph.dot")

        # 定义 graph 输入 placeholder
        self.graph_input['x'] = x
        self.graph_input['y'] = y
        self.graph_input['z'] = z
        # 定义 graph 输出 placeholder
        self.graph_output['output'] = out

    def forward(self, x, y, z):
        self.graph_input['x'].copy_(x)
        self.graph_input['y'].copy_(y)
        self.graph_input['z'].copy_(z)
        self.cuda_graph.replay()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# 创建模型和输入数据
class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        num_layer = 10000
        self.blocks = torch.nn.ModuleList([nn.Linear(D_in, D_out) for _ in range(num_layer)])

    def forward(self, x, y, z):
        a = torch.matmul(x, y)
        b = torch.matmul(x, z)
        c = torch.add(a, b)
        for block in self.blocks:
            c = block(c)
        return c


def timed(fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    repeat = 10
    start.record()
    for _ in range(repeat):
        result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / repeat


model = simpel_model().cuda()
inp = torch.randn(32, D_in).cuda()
model.eval()
model(x=inp, y=inp, z=inp)  # warm up, 触发一些 gpu 资源的初始化
graph_runner = CUDAGraphRunner(model)
inputs = {"x": inp, "y": inp, "z": inp}
graph_runner.capture(**inputs)
graph_runner(**inputs)  # cuda_graph_runner warm up

input = torch.randn(32, D_in).cuda()
output, cuda_graph_elasped_time = timed(graph_runner, **inputs)
output_ref, ori_infernce_elasped_time = timed(model.forward, **inputs)

torch.cuda.synchronize()
torch.testing.assert_close(output_ref, output, rtol=1e-03, atol=1e-03)
print(
    f"cuda_graph_elasped_time: {cuda_graph_elasped_time} ms, ori_infernce_elasped_time: {ori_infernce_elasped_time} ms")


