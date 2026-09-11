"""Native WebGPU compute, tested on AMD/Vulkan; no browser or CPU fallback.

Route planning is on CPU. Gather, both expert projections, and merge run in
custom WGSL shaders. FP32 CPU tensors form the public input/output boundary.
"""

from functools import lru_cache
from importlib.resources import files

import numpy as np
import torch

from .routing import _compile_routes
from .runtime import ExpertWeights, validate_inputs


def _wgpu():
    try:
        import wgpu
    except ImportError as exc:
        raise RuntimeError("Install the AMD GPU extra: python -m pip install -e '.[amd]'") from exc
    return wgpu


def list_adapters() -> list[dict]:
    return [dict(adapter.info) for adapter in _wgpu().gpu.enumerate_adapters_sync()]


def _select_adapter(adapters, name=None):
    hardware = [a for a in adapters if a.info['adapter_type'] in ('DiscreteGPU', 'IntegratedGPU')]
    if name:
        hardware = [a for a in hardware if name.lower() in a.info['device'].lower()]
    if not hardware:
        raise RuntimeError(f"No hardware GPU matches {name!r}; software adapters are not accepted")
    # Prefer the discrete card and Vulkan; allow other real GPU backends.
    return min(hardware, key=lambda a: (
        a.info['adapter_type'] != 'DiscreteGPU', a.info['backend_type'] != 'Vulkan'
    ))


@lru_cache(maxsize=4)
def get_context(adapter_name=None):
    return GPUContext(adapter_name)


class GPUContext:
    def __init__(self, adapter_name=None):
        wgpu = _wgpu()
        self.adapter = _select_adapter(wgpu.gpu.enumerate_adapters_sync(), adapter_name)
        self.info = dict(self.adapter.info)
        self.device = self.adapter.request_device_sync(label="RouteKernel compute")
        self.pipelines = {}
        for name in ('gather', 'grouped_gemm', 'merge'):
            source = files('routekernel').joinpath('shaders', name + '.wgsl').read_text(encoding='utf-8')
            module = self.device.create_shader_module(label=name, code=source)
            self.pipelines[name] = self.device.create_compute_pipeline(
                label=name, layout='auto', compute={'module': module, 'entry_point': 'main'}
            )

    def upload(self, values, label, uniform=False):
        wgpu = _wgpu()
        values = np.ascontiguousarray(values)
        # WebGPU disallows zero-sized bindings. Empty workloads never dispatch.
        if not values.nbytes:
            values = np.zeros(1, dtype=values.dtype)
        self.check_buffer_size(values.nbytes, uniform)
        usage = wgpu.BufferUsage.UNIFORM if uniform else wgpu.BufferUsage.STORAGE
        return self.device.create_buffer_with_data(label=label, data=values, usage=usage | wgpu.BufferUsage.COPY_DST)

    def check_buffer_size(self, size, uniform=False):
        kind = 'max-uniform-buffer-binding-size' if uniform else 'max-storage-buffer-binding-size'
        if size > min(self.device.limits[kind], self.device.limits['max-buffer-size']):
            raise ValueError(f"Buffer of {size} bytes exceeds this GPU's binding limits")

    def allocate(self, size, label):
        wgpu = _wgpu()
        size = max(size, 4)
        self.check_buffer_size(size)
        return self.device.create_buffer(label=label, size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    def bindings(self, name, buffers):
        return self.device.create_bind_group(
            label=name, layout=self.pipelines[name].get_bind_group_layout(0),
            entries=[{'binding': i, 'resource': {'buffer': buffer}} for i, buffer in enumerate(buffers)],
        )


class GPUWorkload:
    """Prepared input buffers and four dispatches; weights belong to the engine."""
    def __init__(self, engine, tokens, ids, weights):
        self.engine = engine
        self.context = engine.context
        self.buffers = []
        self.dispatches = []
        self.shape = tuple(tokens.shape)
        self.closed = False
        try:
            self._prepare(engine, tokens, ids, weights)
        except Exception:
            self.close()
            raise

    def _prepare(self, engine, tokens, ids, weights):
        ctx = self.context
        t, d = tokens.shape
        e, _, h = engine.experts.up.shape
        k = ids.shape[1]
        routes = _compile_routes(ids, weights, e)
        count = t * k
        offsets = routes.expert_offsets.tolist()
        tile_experts, tile_rows = [], []
        for expert in range(e):
            for start in range(offsets[expert], offsets[expert + 1], 16):
                tile_experts.append(expert)
                tile_rows.append(start)
        # Every shader address is u32; reject inputs that would overflow it.
        if max(count * d, count * h, e * d * h, t * d) >= 2**32:
            raise ValueError('Workload exceeds the WGSL u32 indexing range')
        grids = [((count * d + 255) // 256, 1, 1), ((h + 15) // 16, len(tile_rows), 1),
                 ((d + 15) // 16, len(tile_rows), 1), ((t * d + 255) // 256, 1, 1)]
        if any(max(grid) > ctx.device.limits['max-compute-workgroups-per-dimension'] for grid in grids):
            raise ValueError('Workload exceeds GPU dispatch dimensions; use a smaller batch')

        def upload(value, label, uniform=False):
            buffer = ctx.upload(value, label, uniform)
            self.buffers.append(buffer)
            return buffer

        def allocate(elements, label):
            buffer = ctx.allocate(elements * 4, label)
            self.buffers.append(buffer)
            return buffer

        x = upload(tokens.detach().numpy(), 'tokens')
        token_ids = upload(routes.token_ids.numpy().astype(np.uint32), 'token IDs')
        boundaries = upload(np.array(offsets, dtype=np.uint32), 'expert offsets')
        schedule = upload(np.array(tile_experts, dtype=np.uint32), 'tile experts')
        starts = upload(np.array(tile_rows, dtype=np.uint32), 'tile row starts')
        routing_weights = upload(routes.routing_weights.detach().numpy().astype(np.float32), 'routing weights')
        inverse = upload(routes.inverse_permutation.numpy().astype(np.uint32), 'inverse permutation')
        grouped = allocate(count * d, 'grouped tokens')
        hidden = allocate(count * h, 'hidden activations')
        values = allocate(count * d, 'expert outputs')
        self.output = allocate(t * d, 'merged output')
        gather_shape = upload(np.array([count, d, k, 0], dtype=np.uint32), 'gather shape', True)
        up_shape = upload(np.array([count, h, d, 1], dtype=np.uint32), 'up shape', True)
        down_shape = upload(np.array([count, d, h, 0], dtype=np.uint32), 'down shape', True)
        merge_shape = upload(np.array([t, d, k, 0], dtype=np.uint32), 'merge shape', True)
        groups = [
            ('gather', [x, token_ids, grouped, gather_shape]),
            ('grouped_gemm', [grouped, engine.up, hidden, boundaries, schedule, starts, up_shape]),
            ('grouped_gemm', [hidden, engine.down, values, boundaries, schedule, starts, down_shape]),
            ('merge', [values, routing_weights, inverse, self.output, merge_shape]),
        ]
        self.dispatches = [(name, ctx.bindings(name, buffers), grid) for (name, buffers), grid in zip(groups, grids)]

    def submit(self):
        if self.closed:
            raise RuntimeError('GPU workload is closed')
        if self.engine.closed:
            raise RuntimeError('The GPU engine must stay open while its workloads execute')
        if self.shape[0] == 0:
            return
        encoder = self.context.device.create_command_encoder(label='RouteKernel forward')
        compute = encoder.begin_compute_pass(label='gather / experts / merge')
        for name, bindings, grid in self.dispatches:
            compute.set_pipeline(self.context.pipelines[name])
            compute.set_bind_group(0, bindings)
            compute.dispatch_workgroups(*grid)
        compute.end()
        self.context.device.queue.submit([encoder.finish()])

    def synchronize(self):
        if self.closed:
            raise RuntimeError('GPU workload is closed')
        if self.shape[0]:
            # wgpu 0.32's Windows queue-completion callback has an ABI mismatch.
            # A blocking four-byte read is a public-API completion fence. Its
            # copy/map overhead is included in resident benchmark timings.
            self.context.device.queue.read_buffer(self.output, size=4)

    def read(self):
        if self.closed:
            raise RuntimeError('GPU workload is closed')
        if self.shape[0] == 0:
            return torch.empty(self.shape, dtype=torch.float32)
        # read_buffer blocks until the GPU copy/map completes.
        data = self.context.device.queue.read_buffer(self.output, size=int(np.prod(self.shape)) * 4)
        return torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy().reshape(self.shape))

    def run(self):
        self.submit()
        return self.read()

    def close(self):
        if not self.closed:
            for buffer in self.buffers:
                buffer.destroy()
            self.dispatches.clear()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class WgpuEngine:
    """An inference session with expert parameters uploaded once to the GPU.

    Inputs and parameters must be CPU FP32 tensors. Parameters are a snapshot:
    create a new engine after changing them. prepare() validates by default.
    """
    def __init__(self, experts: ExpertWeights, *, adapter_name=None):
        if experts.up.ndim != 3 or experts.down.ndim != 3:
            raise ValueError('Expected expert matrices [E,D,H] and [E,H,D]')
        e, d, h = experts.up.shape
        if min(e, d, h) < 1 or experts.down.shape != (e, h, d):
            raise ValueError('Invalid expert matrix shapes')
        if any(x.dtype != torch.float32 or x.device.type != 'cpu' for x in (experts.up, experts.down)):
            raise ValueError('wgpu parameters must be CPU float32 tensors')
        self.context = get_context(adapter_name)
        self.experts = experts
        self.closed = False
        self.up = self.context.upload(experts.up.detach().numpy(), 'expert up weights')
        try:
            self.down = self.context.upload(experts.down.detach().numpy(), 'expert down weights')
        except Exception:
            self.up.destroy()
            raise

    def prepare(self, tokens, ids, weights, *, validate=True):
        if self.closed:
            raise RuntimeError('GPU engine is closed')
        if tokens.dtype != torch.float32 or tokens.device.type != 'cpu':
            raise ValueError('wgpu currently accepts CPU float32 input tensors')
        if validate:
            validate_inputs(tokens, ids, weights, self.experts)
        return GPUWorkload(self, tokens, ids, weights)

    def forward(self, tokens, ids, weights):
        with self.prepare(tokens, ids, weights) as workload:
            return workload.run()

    def close(self):
        if not self.closed:
            self.up.destroy()
            self.down.destroy()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
