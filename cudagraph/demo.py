import torch
import torch.nn as nn

D_in = 32
D_out = 32
torch.manual_seed(1)


class CUDAGraphRunner():
    def __init__(self, model):
        self.model = model
        self.cuda_graph = None
        self.graph_input = {}
        self.graph_output = {}

    def capture(self, x, condition):
        assert self.cuda_graph is None

        self.cuda_graph = torch.cuda.CUDAGraph()
        self.cuda_graph.enable_debug_mode()
        with torch.cuda.graph(self.cuda_graph):
            out = self.model(x, condition)
        torch.cuda.synchronize()
        self.cuda_graph.debug_dump("graph.dot")

        # 定义 graph 输入 placeholder
        self.graph_input['x'] = x
        # 定义 graph 输出 placeholder
        self.graph_output['output'] = out

    def forward(self, x, condition):
        self.graph_input['x'].copy_(x)
        self.cuda_graph.replay()
        return self.graph_output['output']

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# 创建模型和输入数据
class simpel_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_in, D_out)

    def forward(self, x, condition):
        if condition:
            out = torch.add(x, 1)
        else:
            out = self.proj(x)
        return out


model = simpel_model().cuda()
model.eval()

inp = torch.randn(32, D_in).cuda()
model(inp, condition=False)  # warm up, 触发一些 gpu 资源的初始化

graph_runner = CUDAGraphRunner(model)
graph_runner.capture(inp, condition=False)

output = graph_runner(inp, condition=True)  # cuda_graph_runner run
output_ref = model(inp, condition=True)

torch.cuda.synchronize()
torch.testing.assert_close(output_ref, output, rtol=1e-03, atol=1e-03)