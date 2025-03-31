    # filename: _benchmark_util.py
    import torch
    import itertools
    import triton
    import matplotlib.pyplot as plt
    import numpy as np

    def _generate_configs(dimensions):
        # Simplified config generation
        if not isinstance(dimensions, list) or not all(isinstance(d, list) for d in dimensions):
             raise ValueError("Dimensions must be a list of lists")
        for config_values in itertools.product(*dimensions):
             yield config_values # Assuming config values directly correspond to inputs

    def _check_outputs_match(config, correctness_report, output1, output2, dtype, atol, rtol, label=""):
        correctness_report["total"] += 1
        try:
            # Using torch.allclose for comparison
            is_close = torch.allclose(output1, output2, atol=atol, rtol=rtol)
            if is_close:
                correctness_report["passed"] += 1
                status = "PASS"
            else:
                correctness_report["failed"] += 1
                status = "FAIL"
            print(f"Config {config} - {label}: {status}")
        except Exception as e:
             correctness_report["failed"] += 1
             status = f"ERROR ({e})"
             print(f"Config {config} - {label}: {status}")

        correctness_report["results"].append({
            "config": config,
            "label": label,
            "status": status,
            # Add more details if needed, e.g., max difference
        })


    def _copy_parameters(source_module, target_module):
        # Simple parameter copy by name
        target_state_dict = target_module.state_dict()
        for name, param in source_module.named_parameters():
            if name in target_state_dict:
                target_state_dict[name].copy_(param.data)

    def _backward_pass(module, inputs, dy):
        # Assumes inputs is a tuple or a single tensor
        if isinstance(inputs, tuple):
            outputs = module(*inputs)
        else:
            outputs = module(inputs)

        # Retain grad for inputs that require it
        if isinstance(inputs, tuple):
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.requires_grad:
                     inp.retain_grad()
        elif isinstance(inputs, torch.Tensor) and inputs.requires_grad:
            inputs.retain_grad()

        # Assuming output is a single tensor for simplicity
        outputs.backward(gradient=dy, retain_graph=True)

        # Collect input gradients
        dx = []
        if isinstance(inputs, tuple):
            for inp in inputs:
                 if isinstance(inp, torch.Tensor) and inp.grad is not None:
                     dx.append(inp.grad.clone())
                 else:
                     dx.append(None) # Placeholder for non-tensor or non-req-grad inputs
            dx = tuple(dx) if len(dx) > 1 else dx[0] # Return tuple if multiple grads
        elif isinstance(inputs, torch.Tensor) and inputs.grad is not None:
            dx = inputs.grad.clone()
        else:
            dx = None # No input gradients

        # Collect parameter gradients
        dw = []
        for param in module.parameters():
             if param.grad is not None:
                 dw.append(param.grad.clone())
             else:
                 dw.append(None) # Placeholder if no gradient
        dw = tuple(dw) if len(dw) > 1 else (dw[0] if dw else None) # Return tuple or single tensor

        # Zero gradients for next iteration if necessary (might be handled elsewhere)
        module.zero_grad(set_to_none=True)
        if isinstance(inputs, tuple):
            for inp in inputs:
                if isinstance(inp, torch.Tensor) and inp.grad is not None:
                    inp.grad = None
        elif isinstance(inputs, torch.Tensor) and inputs.grad is not None:
            inputs.grad = None


        return dx, dw # Return input grads and weight grads

    def _benchmark_implementations(
        config,
        func_triton,
        func_pytorch,
        inputs,
        timings_triton,
        timings_pytorch,
        speedups,
        performance_report,
        num_warmup=25,
        num_repeats=100
    ):
        # Benchmark function execution time
        try:
             t_triton = triton.testing.do_bench(lambda: func_triton(inputs), warmup=num_warmup, rep=num_repeats)
             t_pytorch = triton.testing.do_bench(lambda: func_pytorch(inputs), warmup=num_warmup, rep=num_repeats)
             speedup = t_pytorch / t_triton
             timings_triton.append(t_triton)
             timings_pytorch.append(t_pytorch)
             speedups.append(speedup)
             status = f"Triton: {t_triton:.4f} ms, PyTorch: {t_pytorch:.4f} ms, Speedup: {speedup:.2f}x"
             print(f"Config {config}: {status}")
             performance_report["results"].append({
                 "config": config,
                 "triton_ms": t_triton,
                 "pytorch_ms": t_pytorch,
                 "speedup": speedup,
                 "status": "Success"
             })
        except Exception as e:
            status = f"ERROR ({e})"
            print(f"Config {config}: {status}")
            # Append placeholder values or handle error as appropriate
            timings_triton.append(float('inf'))
            timings_pytorch.append(float('inf'))
            speedups.append(0)
            performance_report["results"].append({
                 "config": config,
                 "triton_ms": None,
                 "pytorch_ms": None,
                 "speedup": None,
                 "status": status
            })

    def _plot_benchmark_results(configs, timings_triton, timings_pytorch, speedups, label=""):
        # Basic plotting function
        plt.figure(figsize=(12, 6))

        # Assuming configs are tuples, create string labels
        config_labels = [str(c) for c in configs]
        x = np.arange(len(configs))

        plt.subplot(1, 2, 1)
        plt.plot(x, timings_triton, 'o-', label='Triton')
        plt.plot(x, timings_pytorch, 'o-', label='PyTorch (Compiled)')
        plt.xticks(x, config_labels, rotation=90)
        plt.xlabel("Configuration")
        plt.ylabel("Time (ms)")
        plt.title(f"{label.capitalize()} Performance")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(x, speedups, 'o-')
        plt.xticks(x, config_labels, rotation=90)
        plt.xlabel("Configuration")
        plt.ylabel("Speedup (PyTorch / Triton)")
        plt.title(f"{label.capitalize()} Speedup")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"code.performance.{label}.png")
        print(f"\nPerformance graph saved to code.performance.{label}.png")
        # plt.show() # Optionally display the plot
    