"""Stream and device safety of FlashTP's uvu_TP, checked against e3nn.

Each test runs in its own process, so a crash in one does not hide the others, and prints one line:
  eager       default stream, no graph (control)
  sidestream  the same call under `with torch.cuda.stream(s)`, after ~0.2 s of GPU work queued on s
  graph1      one layer captured in a CUDA graph, replayed on new inputs (output and d(out)/dx)
  graph2      two layers with the same shapes but different tables, captured in one graph and replayed
  compile     torch.compile(mode="reduce-overhead"), which uses CUDA graphs internally
  graph_f64   graph2 in float64, with d(out)/dx of each layer, replayed twice on new inputs
  graph_dd    second derivatives (through a force-like loss) in a CUDA graph, float32 and float64, replayed twice

The layers use l <= 2, so they run the sptp_exp_opt_large forward, backward and double-backward kernels,
in float32 and float64.

usage: python tests/test_stream_safety.py [all|eager|sidestream|graph1|graph2|compile|graph_f64|graph_dd]
       pytest tests/test_stream_safety.py
Exit code 1 if a test fails. Needs e3nn and an installed flashTP_e3nn (rebuild it after changing the CUDA code).
"""
import json, os, random, re, subprocess, sys

TESTS = ['eager', 'sidestream', 'graph1', 'graph2', 'compile', 'graph_f64', 'graph_dd']
TOL = 1e-5                         # max error relative to the largest reference value (float32 rounding: ~3e-7)
N_NODE, K_NEIGH = 256, 32          # 8,192 edges grouped by destination, as in example/flashTP_test.py
SLEEP_CYCLES = 500_000_000         # torch.cuda._sleep: about 0.2 s


def setup(dev, dtype=None):
    import torch, e3nn.o3 as o3, flashTP_e3nn
    dtype = dtype or torch.float32
    torch.manual_seed(0); random.seed(0)

    def layer(in2):
        irr1, irr2 = o3.Irreps('32x0e+32x1o'), o3.Irreps(in2)
        out, ins = [], []
        for i1, (m1, ir1) in enumerate(irr1):
            for i2, (_, ir2) in enumerate(irr2):
                for ir in ir1 * ir2:
                    if ir.l <= 2:
                        out.append((m1, ir)); ins.append((i1, i2, len(out) - 1, 'uvu', True))
        irro = o3.Irreps(out)
        ref = o3.TensorProduct(irr1, irr2, irro, ins, shared_weights=False, internal_weights=False).to(dev, dtype)
        ftp = flashTP_e3nn.uvu_TP(irr1, irr2, irro, ins, device=dev, dtype=dtype)
        return dict(ref=ref, ftp=ftp, d1=irr1.dim, d2=irr2.dim, do=irro.dim, nw=ref.weight_numel, dtype=dtype)

    # same shapes, in2 irreps in a different order: different path and CG tables
    A, B = layer('1x0e+1x1o+1x2e'), layer('1x2e+1x1o+1x0e')
    src, dst = [], []
    for i in range(N_NODE):
        for _ in range(K_NEIGH):
            j = i
            while j == i:
                j = random.randint(0, N_NODE - 1)
            src.append(j); dst.append(i)
    src = torch.tensor(src, dtype=torch.int32, device=dev)
    dst = torch.tensor(dst, dtype=torch.int32, device=dev)
    return torch, A, B, src, dst


def inputs(torch, L, n_edge, seed, dev):
    g, dt = torch.Generator(device=dev).manual_seed(seed), L['dtype']
    return (torch.randn(N_NODE, L['d1'], generator=g, device=dev, dtype=dt),
            torch.randn(n_edge, L['d2'], generator=g, device=dev, dtype=dt),
            torch.randn(n_edge, L['nw'], generator=g, device=dev, dtype=dt))


def reference(torch, L, x, sh, w, src, dst):
    y = L['ref'](x[src.long()], sh, w)
    return torch.zeros(N_NODE, L['do'], device=x.device, dtype=x.dtype).index_add_(0, dst.long(), y)


def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def capture(torch, step):
    """Warm up on a side stream, as the PyTorch docs prescribe, then capture step() in a CUDA graph."""
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = step()
    return g, outs


def replay(torch, g, static, new):
    """Copy new inputs into the graph's static tensors and replay."""
    with torch.no_grad():
        for a, b in zip(static, new):
            a.copy_(b)
    g.replay(); torch.cuda.synchronize()


def run(test):
    import torch
    dev = 'cuda'
    if test == 'graph_f64':
        _, A, B, src, dst = setup(dev, torch.float64)
        E = src.numel()
        xg, ss, ws = (t.clone() for t in inputs(torch, A, E, 1, dev))
        xg.requires_grad_()
        c = torch.randn(N_NODE, A['do'], device=dev, dtype=torch.float64)

        def out_dx(f, x, sh, w):
            o = f(x, sh, w)
            return o.detach(), torch.autograd.grad((o * c).sum(), x)[0]

        g, outs = capture(torch, lambda: out_dx(lambda *a: A['ftp'](*a, src, dst), xg, ss, ws) +
                          out_dx(lambda *a: B['ftp'](*a, src, dst), xg, ss, ws))
        r = {}
        for seed in (2, 3):                       # two replays on different inputs
            x, sh, w = inputs(torch, A, E, seed, dev)
            replay(torch, g, (xg, ss, ws), (x, sh, w))
            refs = [t for L in (A, B) for t in out_dx(lambda *a, L=L: reference(torch, L, *a, src, dst),
                                                      x.clone().requires_grad_(), sh, w)]
            for k, o, ref in zip(('err_A', 'err_A_dx', 'err_B', 'err_B_dx'), outs, refs):
                r[k] = max(r.get(k, 0.0), rel(o, ref))
            r['layers_differ'] = rel(refs[0], refs[2])
        return r

    if test == 'graph_dd':
        r = {}
        for name, dt in (('f32', torch.float32), ('f64', torch.float64)):
            _, A, _, src, dst = setup(dev, dt)
            E = src.numel()
            static = [t.clone().requires_grad_() for t in inputs(torch, A, E, 1, dev)]
            c = torch.randn(N_NODE, A['do'], device=dev, dtype=dt)

            def second(f, x, sh, w):              # force-like loss |d/dx|^2 + |d/dsh|^2, then its gradient
                gx, gs = torch.autograd.grad((f(x, sh, w) * c).sum(), (x, sh), create_graph=True)
                return torch.autograd.grad((gx ** 2).sum() + (gs ** 2).sum(), (x, sh, w))

            g, outs = capture(torch, lambda: second(lambda *a: A['ftp'](*a, src, dst), *static))
            for seed in (2, 3):
                new = inputs(torch, A, E, seed, dev)
                replay(torch, g, static, new)
                refs = second(lambda *a: reference(torch, A, *a, src, dst), *(t.clone().requires_grad_() for t in new))
                for lab, o, ref in zip(('d2x', 'd2sh', 'd2w'), outs, refs):
                    k = f'err_{name}_{lab}'
                    r[k] = max(r.get(k, 0.0), rel(o, ref))
        return r

    _, A, B, src, dst = setup(dev)
    E = src.numel()
    fA = lambda x, sh, w: A['ftp'](x, sh, w, src, dst)
    fB = lambda x, sh, w: B['ftp'](x, sh, w, src, dst)
    x1, s1, w1 = inputs(torch, A, E, 1, dev)
    x2, s2, w2 = inputs(torch, A, E, 2, dev)
    refA2, refB2 = reference(torch, A, x2, s2, w2, src, dst), reference(torch, B, x2, s2, w2, src, dst)
    for _ in range(3):
        fA(x1, s1, w1); fB(x1, s1, w1)
    torch.cuda.synchronize()

    if test == 'eager':
        return {'err_A': rel(fA(x2, s2, w2), refA2), 'err_B': rel(fB(x2, s2, w2), refB2)}

    if test == 'sidestream':
        xs = torch.zeros_like(x2); torch.cuda.synchronize()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            torch.cuda._sleep(SLEEP_CYCLES)       # the caller's stream is busy ...
            xs.copy_(x2)                          # ... then writes the input ...
            out = fA(xs, s2, w2)                  # ... then calls FlashTP on the same stream
        torch.cuda.synchronize()
        return {'err': rel(out, refA2)}

    if test in ('graph1', 'graph2'):
        xs, ss, ws = x1.clone(), s1.clone(), w1.clone()
        c = torch.randn(N_NODE, A['do'], device=dev)
        xg = xs.clone().requires_grad_() if test == 'graph1' else None

        def step():
            if test == 'graph1':
                o = fA(xg, ss, ws)
                return o.detach(), torch.autograd.grad((o * c).sum(), xg)[0]
            return fA(xs, ss, ws), fB(xs, ss, ws)

        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):             # warm-up on a side stream, as the PyTorch docs prescribe
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o1, o2 = step()
        xs.copy_(x2); ss.copy_(s2); ws.copy_(w2)  # new inputs, then replay
        if xg is not None:
            with torch.no_grad():
                xg.copy_(x2)
        g.replay(); torch.cuda.synchronize()
        if test == 'graph1':
            xr = x2.clone().requires_grad_()
            gref = torch.autograd.grad((reference(torch, A, xr, s2, w2, src, dst) * c).sum(), xr)[0]
            return {'err_out': rel(o1, refA2), 'err_dx': rel(o2, gref)}
        # the case only means something if the two layers really compute different things
        return {'err_A': rel(o1, refA2), 'err_B': rel(o2, refB2), 'layers_differ': rel(refA2, refB2)}

    if test == 'compile':
        f = torch.compile(lambda x, sh, w: A['ftp'](x, sh, w, src, dst), mode='reduce-overhead')
        for _ in range(3):
            f(x1, s1, w1)
        o = f(x2, s2, w2).clone(); torch.cuda.synchronize()
        return {'err': rel(o, refA2)}


def check(r):
    """(passed, text) for one result dict."""
    if 'skipped' in r:
        return True, 'skip  ' + r['skipped']
    errs = {k: v for k, v in r.items() if k.startswith('err')}
    ok = all(v < TOL for v in errs.values())     # NaN fails
    if 'layers_differ' in r:
        ok = ok and r['layers_differ'] > 0.1
    extra = {k: v for k, v in r.items() if not k.startswith('err')}
    return ok, ('ok    ' if ok else 'FAIL  ') + ', '.join(f'{k} {v:.1e}' for k, v in errs.items()) + \
        (f'  {extra}' if extra else '')


def run_in_subprocess(test):
    p = subprocess.run([sys.executable, os.path.abspath(__file__), test], capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
    if not lines:
        lines = p.stderr.strip().splitlines()
        # the last exception line is the innermost error, e.g. the CUDA error under a TorchScript traceback
        errs = [l.strip() for l in lines if re.match(r'\s*[\w.]*(Error|Exception)\b', l)] or lines[-1:]
        return False, f'FAIL  crashed (exit {p.returncode}): {errs[-1][:300] if errs else ""}'
    return check(json.loads(lines[-1][7:]))


def main():
    test = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if test not in TESTS + ['all']:
        print(__doc__); return 2
    import torch
    if not torch.cuda.is_available():
        print('no CUDA device: nothing to test'); return 0
    if test != 'all':
        r = run(test)
        print('RESULT ' + json.dumps(r), flush=True)
        return 0 if check(r)[0] else 1
    print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}, '
          f'{torch.cuda.device_count()} GPU(s)')
    failed = 0
    for t in TESTS:
        ok, text = run_in_subprocess(t)
        print(f'{t:11s} {text}', flush=True)
        failed += not ok
    return 1 if failed else 0


def test_stream_safety():
    """pytest entry point: every case in its own process."""
    import pytest
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA')
    failures = [f'{t}: {text}' for t in TESTS for ok, text in [run_in_subprocess(t)] if not ok]
    assert not failures, '\n'.join(failures)


if __name__ == '__main__':
    sys.exit(main())
