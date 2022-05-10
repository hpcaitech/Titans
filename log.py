============================= test session starts ==============================
platform linux -- Python 3.8.13, pytest-7.1.2, pluggy-1.0.0
rootdir: /home/lcwbx/Titans
collected 5 items

tests/test_layer/test_embedding/test_detr_embedding.py FFFFF             [100%]

=================================== FAILURES ===================================
____________________ test_detr_embedding[parallel_config0] _____________________

args = (), kwargs = {'parallel_config': (4, '1d')}, try_count = 1
error_lines = ['', '', '-- Process 2 terminated with the following error:', 'Traceback (most recent call last):', '  File "/home/lcw...vs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap', '    fn(i, *args)', ...]

    def _run_until_success(*args, **kwargs):
        try_count = 0
        assert max_try is None or isinstance(max_try, int), \
            f'Expected max_try to be None or int, but got {type(max_try)}'
    
        while max_try is None or try_count < max_try:
            try:
                try_count += 1
                ret = func(*args, **kwargs)
                return ret
            except exception_type as e:
                error_lines = str(e).split('\n')
                if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                    print('Exception is caught, retrying...')
                    # when pattern is not specified, we always skip the exception
                    # when pattern is specified, we only skip when pattern is matched
                    continue
                else:
                    print('Maximum number of attempts is reached or pattern is not matched, no more retrying...')
>                   raise e

../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:138: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:127: in _run_until_success
    ret = func(*args, **kwargs)
tests/test_layer/test_embedding/test_detr_embedding.py:42: in test_detr_embedding
    run_with_parallel_config(*parallel_config, run_func=run_dist)
tests/utils/dist_test.py:24: in run_with_parallel_config
    mp.spawn(run_func, nprocs=world_size)
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:240: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:198: in start_processes
    while not context.join():
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <torch.multiprocessing.spawn.ProcessContext object at 0x7f7e668f43d0>
timeout = None

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.
    
        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.
    
        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True
    
        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
    
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
    
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
    
        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
    
        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )
    
        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
>       raise ProcessRaisedException(msg, error_index, failed_process.pid)
E       torch.multiprocessing.spawn.ProcessRaisedException: 
E       
E       -- Process 2 terminated with the following error:
E       Traceback (most recent call last):
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
E           fn(i, *args)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 36, in run_dist
E           run_detr_embed(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 23, in run_detr_embed
E           out = model(data)
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
E           return forward_call(*input, **kwargs)
E         File "/home/lcwbx/Titans/titans/layer/embedding/detr_embedding.py", line 28, in forward
E           x = tensor_list.tensors
E       AttributeError: 'Tensor' object has no attribute 'tensors'

../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:160: ProcessRaisedException
----------------------------- Captured stdout call -----------------------------
[05/13/22 16:11:59] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 3                                        
[05/13/22 16:11:59] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 2                                        
[05/13/22 16:11:59] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
[05/13/22 16:11:59] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
[05/13/22 16:12:00] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 2 is  
                             bound to device 2                                  
[05/13/22 16:12:00] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 3 is  
                             bound to device 3                                  
[05/13/22 16:12:00] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
[05/13/22 16:12:00] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 0 is  
                             bound to device 0                                  
                    INFO     colossalai - colossalai - INFO: process rank 1 is  
                             bound to device 1                                  
[05/13/22 16:12:03] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:03] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:03] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:03] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 2, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1026,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 0, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1024,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 1, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1025,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 3, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1027,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/initialize.py:117 launch                       
                    INFO     colossalai - colossalai - INFO: Distributed        
                             environment is initialized, data parallel size: 1, 
                             pipeline parallel size: 1, tensor parallel size: 4 
Maximum number of attempts is reached or pattern is not matched, no more retrying...
____________________ test_detr_embedding[parallel_config1] _____________________

args = (), kwargs = {'parallel_config': (4, '2d')}, try_count = 1
error_lines = ['', '', '-- Process 2 terminated with the following error:', 'Traceback (most recent call last):', '  File "/home/lcw...vs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap', '    fn(i, *args)', ...]

    def _run_until_success(*args, **kwargs):
        try_count = 0
        assert max_try is None or isinstance(max_try, int), \
            f'Expected max_try to be None or int, but got {type(max_try)}'
    
        while max_try is None or try_count < max_try:
            try:
                try_count += 1
                ret = func(*args, **kwargs)
                return ret
            except exception_type as e:
                error_lines = str(e).split('\n')
                if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                    print('Exception is caught, retrying...')
                    # when pattern is not specified, we always skip the exception
                    # when pattern is specified, we only skip when pattern is matched
                    continue
                else:
                    print('Maximum number of attempts is reached or pattern is not matched, no more retrying...')
>                   raise e

../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:138: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:127: in _run_until_success
    ret = func(*args, **kwargs)
tests/test_layer/test_embedding/test_detr_embedding.py:42: in test_detr_embedding
    run_with_parallel_config(*parallel_config, run_func=run_dist)
tests/utils/dist_test.py:24: in run_with_parallel_config
    mp.spawn(run_func, nprocs=world_size)
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:240: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:198: in start_processes
    while not context.join():
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <torch.multiprocessing.spawn.ProcessContext object at 0x7f808b2bea00>
timeout = None

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.
    
        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.
    
        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True
    
        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
    
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
    
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
    
        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
    
        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )
    
        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
>       raise ProcessRaisedException(msg, error_index, failed_process.pid)
E       torch.multiprocessing.spawn.ProcessRaisedException: 
E       
E       -- Process 2 terminated with the following error:
E       Traceback (most recent call last):
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
E           fn(i, *args)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 36, in run_dist
E           run_detr_embed(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 23, in run_detr_embed
E           out = model(data)
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
E           return forward_call(*input, **kwargs)
E         File "/home/lcwbx/Titans/titans/layer/embedding/detr_embedding.py", line 28, in forward
E           x = tensor_list.tensors
E       AttributeError: 'Tensor' object has no attribute 'tensors'

../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:160: ProcessRaisedException
----------------------------- Captured stdout call -----------------------------
[05/13/22 16:12:06] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 1                                        
[05/13/22 16:12:07] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
[05/13/22 16:12:07] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
[05/13/22 16:12:07] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
[05/13/22 16:12:07] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 1 is  
                             bound to device 1                                  
                    INFO     colossalai - colossalai - INFO: process rank 3 is  
                             bound to device 3                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 0 is  
                             bound to device 0                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 2 is  
                             bound to device 2                                  
[05/13/22 16:12:11] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:11] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:11] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 3, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1027,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 2, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1026,the default parallel seed is                  
                             ParallelMode.DATA.                                 
[05/13/22 16:12:11] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 1, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1025,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 0, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1024,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/initialize.py:117 launch                       
                    INFO     colossalai - colossalai - INFO: Distributed        
                             environment is initialized, data parallel size: 1, 
                             pipeline parallel size: 1, tensor parallel size: 4 
Maximum number of attempts is reached or pattern is not matched, no more retrying...
____________________ test_detr_embedding[parallel_config2] _____________________

args = (), kwargs = {'parallel_config': (4, '2.5d')}, try_count = 1
error_lines = ['', '', '-- Process 3 terminated with the following error:', 'Traceback (most recent call last):', '  File "/home/lcw...vs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap', '    fn(i, *args)', ...]

    def _run_until_success(*args, **kwargs):
        try_count = 0
        assert max_try is None or isinstance(max_try, int), \
            f'Expected max_try to be None or int, but got {type(max_try)}'
    
        while max_try is None or try_count < max_try:
            try:
                try_count += 1
                ret = func(*args, **kwargs)
                return ret
            except exception_type as e:
                error_lines = str(e).split('\n')
                if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                    print('Exception is caught, retrying...')
                    # when pattern is not specified, we always skip the exception
                    # when pattern is specified, we only skip when pattern is matched
                    continue
                else:
                    print('Maximum number of attempts is reached or pattern is not matched, no more retrying...')
>                   raise e

../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:138: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:127: in _run_until_success
    ret = func(*args, **kwargs)
tests/test_layer/test_embedding/test_detr_embedding.py:42: in test_detr_embedding
    run_with_parallel_config(*parallel_config, run_func=run_dist)
tests/utils/dist_test.py:24: in run_with_parallel_config
    mp.spawn(run_func, nprocs=world_size)
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:240: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:198: in start_processes
    while not context.join():
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <torch.multiprocessing.spawn.ProcessContext object at 0x7f808b20c310>
timeout = None

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.
    
        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.
    
        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True
    
        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
    
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
    
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
    
        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
    
        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )
    
        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
>       raise ProcessRaisedException(msg, error_index, failed_process.pid)
E       torch.multiprocessing.spawn.ProcessRaisedException: 
E       
E       -- Process 3 terminated with the following error:
E       Traceback (most recent call last):
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
E           fn(i, *args)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 36, in run_dist
E           run_detr_embed(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 23, in run_detr_embed
E           out = model(data)
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
E           return forward_call(*input, **kwargs)
E         File "/home/lcwbx/Titans/titans/layer/embedding/detr_embedding.py", line 28, in forward
E           x = tensor_list.tensors
E       AttributeError: 'Tensor' object has no attribute 'tensors'

../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:160: ProcessRaisedException
----------------------------- Captured stdout call -----------------------------
[05/13/22 16:12:14] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 1                                        
[05/13/22 16:12:15] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
[05/13/22 16:12:15] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 3                                        
[05/13/22 16:12:15] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
[05/13/22 16:12:15] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 4 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 4 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 4 nodes.       
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 1 is  
                             bound to device 1                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 3 is  
                             bound to device 3                                  
                    INFO     colossalai - colossalai - INFO: process rank 0 is  
                             bound to device 0                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 2 is  
                             bound to device 2                                  
[05/13/22 16:12:19] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:19] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:19] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 1, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1025,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 0, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1024,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 3, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1027,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/initialize.py:117 launch                       
[05/13/22 16:12:19] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: Distributed        
                             environment is initialized, data parallel size: 1, 
                             pipeline parallel size: 1, tensor parallel size: 4 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 2, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1026,the default parallel seed is                  
                             ParallelMode.DATA.                                 
Maximum number of attempts is reached or pattern is not matched, no more retrying...
____________________ test_detr_embedding[parallel_config3] _____________________

args = (), kwargs = {'parallel_config': (8, '2.5d')}, try_count = 1
error_lines = ['', '', '-- Process 2 terminated with the following error:', 'Traceback (most recent call last):', '  File "/home/lcw...vs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap', '    fn(i, *args)', ...]

    def _run_until_success(*args, **kwargs):
        try_count = 0
        assert max_try is None or isinstance(max_try, int), \
            f'Expected max_try to be None or int, but got {type(max_try)}'
    
        while max_try is None or try_count < max_try:
            try:
                try_count += 1
                ret = func(*args, **kwargs)
                return ret
            except exception_type as e:
                error_lines = str(e).split('\n')
                if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                    print('Exception is caught, retrying...')
                    # when pattern is not specified, we always skip the exception
                    # when pattern is specified, we only skip when pattern is matched
                    continue
                else:
                    print('Maximum number of attempts is reached or pattern is not matched, no more retrying...')
>                   raise e

../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:138: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:127: in _run_until_success
    ret = func(*args, **kwargs)
tests/test_layer/test_embedding/test_detr_embedding.py:42: in test_detr_embedding
    run_with_parallel_config(*parallel_config, run_func=run_dist)
tests/utils/dist_test.py:24: in run_with_parallel_config
    mp.spawn(run_func, nprocs=world_size)
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:240: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:198: in start_processes
    while not context.join():
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <torch.multiprocessing.spawn.ProcessContext object at 0x7f808b23b4f0>
timeout = None

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.
    
        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.
    
        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True
    
        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
    
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
    
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
    
        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
    
        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )
    
        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
>       raise ProcessRaisedException(msg, error_index, failed_process.pid)
E       torch.multiprocessing.spawn.ProcessRaisedException: 
E       
E       -- Process 2 terminated with the following error:
E       Traceback (most recent call last):
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
E           fn(i, *args)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 36, in run_dist
E           run_detr_embed(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 23, in run_detr_embed
E           out = model(data)
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
E           return forward_call(*input, **kwargs)
E         File "/home/lcwbx/Titans/titans/layer/embedding/detr_embedding.py", line 28, in forward
E           x = tensor_list.tensors
E       AttributeError: 'Tensor' object has no attribute 'tensors'

../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:160: ProcessRaisedException
----------------------------- Captured stdout call -----------------------------
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 2                                        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 3                                        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 4                                        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 1                                        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 5                                        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 7                                        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
[05/13/22 16:12:23] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 5                                        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 0                                        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
[05/13/22 16:12:24] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:47 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:47 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:48 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:48 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:49 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:49 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:50 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:50 with 8 nodes.       
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 1 is  
                             bound to device 1                                  
                    INFO     colossalai - colossalai - INFO: process rank 3 is  
                             bound to device 3                                  
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 0 is  
                             bound to device 0                                  
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 6 is  
                             bound to device 6                                  
                    INFO     colossalai - colossalai - INFO: process rank 2 is  
                             bound to device 2                                  
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 5 is  
                             bound to device 5                                  
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 4 is  
                             bound to device 4                                  
[05/13/22 16:12:25] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 7 is  
                             bound to device 7                                  
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 7, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1031,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 6, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1030,the default parallel seed is                  
                             ParallelMode.DATA.                                 
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 1, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1025,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 2, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1026,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 3, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1027,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 0, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1024,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 4, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1028,the default parallel seed is                  
                             ParallelMode.DATA.                                 
[05/13/22 16:12:31] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 5, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1029,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/initialize.py:117 launch                       
                    INFO     colossalai - colossalai - INFO: Distributed        
                             environment is initialized, data parallel size: 1, 
                             pipeline parallel size: 1, tensor parallel size: 8 
Maximum number of attempts is reached or pattern is not matched, no more retrying...
____________________ test_detr_embedding[parallel_config4] _____________________

args = (), kwargs = {'parallel_config': (8, '3d')}, try_count = 1
error_lines = ['', '', '-- Process 1 terminated with the following error:', 'Traceback (most recent call last):', '  File "/home/lcw...vs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap', '    fn(i, *args)', ...]

    def _run_until_success(*args, **kwargs):
        try_count = 0
        assert max_try is None or isinstance(max_try, int), \
            f'Expected max_try to be None or int, but got {type(max_try)}'
    
        while max_try is None or try_count < max_try:
            try:
                try_count += 1
                ret = func(*args, **kwargs)
                return ret
            except exception_type as e:
                error_lines = str(e).split('\n')
                if try_count < max_try and (pattern is None or _match_lines(error_lines, pattern)):
                    print('Exception is caught, retrying...')
                    # when pattern is not specified, we always skip the exception
                    # when pattern is specified, we only skip when pattern is matched
                    continue
                else:
                    print('Maximum number of attempts is reached or pattern is not matched, no more retrying...')
>                   raise e

../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:138: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/example3.8/lib/python3.8/site-packages/colossalai/testing/utils.py:127: in _run_until_success
    ret = func(*args, **kwargs)
tests/test_layer/test_embedding/test_detr_embedding.py:42: in test_detr_embedding
    run_with_parallel_config(*parallel_config, run_func=run_dist)
tests/utils/dist_test.py:24: in run_with_parallel_config
    mp.spawn(run_func, nprocs=world_size)
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:240: in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:198: in start_processes
    while not context.join():
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <torch.multiprocessing.spawn.ProcessContext object at 0x7f808b20f460>
timeout = None

    def join(self, timeout=None):
        r"""
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.
    
        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.
    
        Args:
            timeout (float): Wait this long before giving up on waiting.
        """
        # Ensure this function can be called even when we're done.
        if len(self.sentinels) == 0:
            return True
    
        # Wait for any process to fail or all of them to succeed.
        ready = multiprocessing.connection.wait(
            self.sentinels.keys(),
            timeout=timeout,
        )
    
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
    
        # Return if there was no error.
        if error_index is None:
            # Return whether or not all processes have been joined.
            return len(self.sentinels) == 0
    
        # Assume failure. Terminate processes that are still alive.
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
    
        # There won't be an error on the queue if the process crashed.
        failed_process = self.processes[error_index]
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise ProcessExitedException(
                    "process %d terminated with signal %s" %
                    (error_index, name),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode,
                    signal_name=name
                )
            else:
                raise ProcessExitedException(
                    "process %d terminated with exit code %d" %
                    (error_index, exitcode),
                    error_index=error_index,
                    error_pid=failed_process.pid,
                    exit_code=exitcode
                )
    
        original_trace = self.error_queues[error_index].get()
        msg = "\n\n-- Process %d terminated with the following error:\n" % error_index
        msg += original_trace
>       raise ProcessRaisedException(msg, error_index, failed_process.pid)
E       torch.multiprocessing.spawn.ProcessRaisedException: 
E       
E       -- Process 1 terminated with the following error:
E       Traceback (most recent call last):
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 69, in _wrap
E           fn(i, *args)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 36, in run_dist
E           run_detr_embed(data, IMAGE_SIZE, PATCH_SIZE, IN_CHANS, HIDDEN_SIZE)
E         File "/home/lcwbx/Titans/tests/test_layer/test_embedding/test_detr_embedding.py", line 23, in run_detr_embed
E           out = model(data)
E         File "/home/lcwbx/.conda/envs/example3.8/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
E           return forward_call(*input, **kwargs)
E         File "/home/lcwbx/Titans/titans/layer/embedding/detr_embedding.py", line 28, in forward
E           x = tensor_list.tensors
E       AttributeError: 'Tensor' object has no attribute 'tensors'

../.conda/envs/example3.8/lib/python3.8/site-packages/torch/multiprocessing/spawn.py:160: ProcessRaisedException
----------------------------- Captured stdout call -----------------------------
[05/13/22 16:12:35] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 6                                        
[05/13/22 16:12:35] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 3                                        
[05/13/22 16:12:35] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 7                                        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 4                                        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 5                                        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 1                                        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:1 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
[05/13/22 16:12:36] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:1 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:2 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:2 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:3 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:3 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:4 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:4 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:5 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:5 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:6 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:6 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:7 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:7 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:8 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 4                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 6                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 3                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 5                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 0                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 1                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:8 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 2                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:9 to store
                             for rank: 7                                        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:9 with 8 nodes.        
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:10 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:10 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:11 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:11 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:12 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:12 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:13 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:13 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:14 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:14 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:15 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:15 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:16 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:16 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:17 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:17 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:18 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:18 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:19 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:19 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:20 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:20 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:21 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:21 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 1                                  
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 2                                  
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 3                                  
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 6                                  
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 5                                  
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 4                                  
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:22 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
[05/13/22 16:12:37] INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:22 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:23 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:23 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:24 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:24 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:25 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:25 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:26 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:26 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:27 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:27 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:28 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:28 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:29 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:29 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:30 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:30 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:31 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:31 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:32 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:32 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:33 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:33 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:34 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:34 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:35 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:35 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:36 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:36 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:37 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:37 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:38 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:38 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:39 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:39 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:40 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:40 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:41 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:41 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:42 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:42 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:43 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:43 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:44 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:44 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:45 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 0                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 4                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 1                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 2                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 5                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:45 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 6                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 3                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 3: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 6: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Added key: store_based_barrier_key:46 to     
                             store for rank: 7                                  
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 7: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 0: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 4: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 1: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 2: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - torch.distributed.distributed_c10d -  
                             INFO: Rank 5: Completed store-based barrier for    
                             key:store_based_barrier_key:46 with 8 nodes.       
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 3 is  
                             bound to device 3                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 2 is  
                             bound to device 2                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 1 is  
                             bound to device 1                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 5 is  
                             bound to device 5                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 7 is  
                             bound to device 7                                  
                    INFO     colossalai - colossalai - INFO: process rank 0 is  
                             bound to device 0                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 6 is  
                             bound to device 6                                  
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:521 set_device     
                    INFO     colossalai - colossalai - INFO: process rank 4 is  
                             bound to device 4                                  
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 1, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1025,the default parallel seed is                  
                             ParallelMode.DATA.                                 
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
[05/13/22 16:12:43] INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/context/parallel_context.py:557 set_seed       
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 2, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1026,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 3, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1027,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 0, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1024,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 7, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1031,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 5, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1029,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 6, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1030,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: initialized seed on
                             rank 4, numpy: 1024, python random: 1024,          
                             ParallelMode.DATA: 1024, ParallelMode.TENSOR:      
                             1028,the default parallel seed is                  
                             ParallelMode.DATA.                                 
                    INFO     colossalai - colossalai - INFO: /home/lcwbx/.conda/
                             envs/example3.8/lib/python3.8/site-packages/colossa
                             lai/initialize.py:117 launch                       
                    INFO     colossalai - colossalai - INFO: Distributed        
                             environment is initialized, data parallel size: 1, 
                             pipeline parallel size: 1, tensor parallel size: 8 
Maximum number of attempts is reached or pattern is not matched, no more retrying...
=========================== short test summary info ============================
FAILED tests/test_layer/test_embedding/test_detr_embedding.py::test_detr_embedding[parallel_config0]
FAILED tests/test_layer/test_embedding/test_detr_embedding.py::test_detr_embedding[parallel_config1]
FAILED tests/test_layer/test_embedding/test_detr_embedding.py::test_detr_embedding[parallel_config2]
FAILED tests/test_layer/test_embedding/test_detr_embedding.py::test_detr_embedding[parallel_config3]
FAILED tests/test_layer/test_embedding/test_detr_embedding.py::test_detr_embedding[parallel_config4]
============================== 5 failed in 49.78s ==============================
