PROJECT_ROOT\.env\lib\site-packages\transformers\utils\deprecation.py:172: UserWarning: The following named arguments are not valid for `SegformerImageProcessor.__init__` and were ignored: 'feature_extractor_type'
  return func(*args, **kwargs)
Traceback (most recent call last):
  File "PYTHON_PATH\Python310\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "PYTHON_PATH\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "PROJECT_ROOT\experiments\run.py", line 82, in <module>
    main()
  File "PROJECT_ROOT\experiments\run.py", line 70, in main
    trainer = HuggingFaceModelTrainer(
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\utils\deprecation.py", line 172, in wrapped_func
    return func(*args, **kwargs)
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 701, in __init__
    raise ValueError(
ValueError: The train_dataset does not implement __len__, max_steps has to be specified. The number of steps needs to be known in advance for the learning rate scheduler.




# Next Fucking Problem
```powershell
  File "PROJECT_ROOT\experiments\run.py", line 91, in <module>
    main()
  File "PROJECT_ROOT\experiments\run.py", line 87, in main
    trainer.train()
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 2245, in train
    return inner_training_loop(
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 2514, in _inner_training_loop
    batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 5243, in get_batch_samples
    batch_samples.append(next(epoch_iterator))
  File "PROJECT_ROOT\.env\lib\site-packages\accelerate\data_loader.py", line 858, in __iter__
    next_batch, next_batch_info = self._fetch_batches(main_iterator)
  File "PROJECT_ROOT\.env\lib\site-packages\accelerate\data_loader.py", line 812, in _fetch_batches
    batches.append(next(iterator))
  File "PROJECT_ROOT\.env\lib\site-packages\torch\utils\data\dataloader.py", line 708, in __next__
    data = self._next_data()
  File "PROJECT_ROOT\.env\lib\site-packages\torch\utils\data\dataloader.py", line 764, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "PROJECT_ROOT\.env\lib\site-packages\torch\utils\data\_utils\fetch.py", line 33, in fetch
    data.append(next(self.dataset_iter))
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 2266, in __iter__
    for key, example in ex_iterable:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1866, in __iter__
    for key, example in self.ex_iterable:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1722, in __iter__
    for key_example in islice(self.ex_iterable, self.n - ex_iterable_num_taken):
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1535, in __iter__
    for x in self.ex_iterable:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1084, in __iter__
    yield from self._iter()
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1263, in _iter
    for key, transformed_example in outputs:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1244, in iter_outputs
    for i, key_example in inputs_iterator:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1133, in iter_inputs
    for key, example in iterator:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1866, in __iter__
    for key, example in self.ex_iterable:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1084, in __iter__
    yield from self._iter()
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1263, in _iter
    for key, transformed_example in outputs:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1244, in iter_outputs
    for i, key_example in inputs_iterator:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1133, in iter_inputs
    for key, example in iterator:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1869, in __iter__
    example = _apply_feature_types_on_example(
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\iterable_dataset.py", line 1781, in _apply_feature_types_on_example
    decoded_example = features.decode_example(encoded_example, token_per_repo_id=token_per_repo_id)
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\features\features.py", line 2100, in decode_example
    return {
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\features\features.py", line 2101, in <dictcomp>
    column_name: decode_nested_example(feature, value, token_per_repo_id=token_per_repo_id)
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\features\features.py", line 1414, in decode_nested_example
    return schema.decode_example(obj, token_per_repo_id=token_per_repo_id) if obj is not None else None
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\features\image.py", line 182, in decode_example
    with xopen(path, "rb", download_config=download_config) as f:
  File "PROJECT_ROOT\.env\lib\site-packages\datasets\utils\file_utils.py", line 949, in xopen
    file_obj = fsspec.open(file, mode=mode, *args, **kwargs).open()
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\core.py", line 147, in open
    return self.__enter__()
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\core.py", line 105, in __enter__
    f = self.fs.open(self.path, mode=mode)
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\spec.py", line 1303, in open
    f = self._open(
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\implementations\zip.py", line 129, in _open
    out = self.zip.open(path, mode.strip("b"), force_zip64=self.force_zip_64)
  File "PYTHON_PATH\Python310\lib\zipfile.py", line 1530, in open
    fheader = zef_file.read(sizeFileHeader)
  File "PYTHON_PATH\Python310\lib\zipfile.py", line 745, in read
    data = self._file.read(n)
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\implementations\http.py", line 598, in read
    return super().read(length)
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\spec.py", line 1941, in read
    out = self.cache._fetch(self.loc, self.loc + length)
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\caching.py", line 491, in _fetch
    self.cache = self.fetcher(start, bend)
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\asyn.py", line 118, in wrapper
    return sync(self.loop, func, *args, **kwargs)
  File "PROJECT_ROOT\.env\lib\site-packages\fsspec\asyn.py", line 91, in sync
    if event.wait(1):
  File "PYTHON_PATH\Python310\lib\threading.py", line 607, in wait
    signaled = self._cond.wait(timeout)
  File "PYTHON_PATH\Python310\lib\threading.py", line 324, in wait
    gotit = waiter.acquire(True, timeout)
```


# ANOTHER PROBLEM
```powershell
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [747,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [748,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [749,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [750,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [751,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [752,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [753,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [754,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [755,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [756,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [757,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [758,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [235,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [236,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [237,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [238,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [239,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [240,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [241,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [242,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [243,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [244,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [245,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [246,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [1,0,0], thread: [247,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [747,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [748,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [749,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [750,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [751,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [752,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [753,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [754,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [755,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [756,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [757,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [758,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [235,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [236,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [237,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [238,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [239,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [240,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [241,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [242,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [243,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [244,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [245,0,0] Assertion `t >= 0 && t < n_classes` failed.
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\NLLLoss2d.cu:106: block: [0,0,0], thread: [246,0,0] Assertion `t >= 0 && t < n_classes` failed.
Traceback (most recent call last):
  File "PYTHON_PATH\Python310\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "PYTHON_PATH\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "PROJECT_ROOT\experiments\run.py", line 91, in <module>
    main()
  File "PROJECT_ROOT\experiments\run.py", line 87, in main
    trainer.train()
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 2245, in train
    return inner_training_loop(
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 2560, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "PROJECT_ROOT\.env\lib\site-packages\transformers\trainer.py", line 3782, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "PROJECT_ROOT\.env\lib\site-packages\accelerate\accelerator.py", line 2454, in backward
    loss.backward(**kwargs)
  File "PROJECT_ROOT\.env\lib\site-packages\torch\_tensor.py", line 626, in backward
    torch.autograd.backward(
  File "PROJECT_ROOT\.env\lib\site-packages\torch\autograd\__init__.py", line 347, in backward
    _engine_run_backward(
  File "PROJECT_ROOT\.env\lib\site-packages\torch\autograd\graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```




