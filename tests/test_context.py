import asyncio

import pytest

from inspect_mi.core.context import MIContext, bind_context, get_current_context

# @pytest.mark.asyncio
# async def test_async_context_isolation():
#     """Verifies that concurrent asyncio tasks running through the same process maintain isolated MIContexts."""
#     results = {}

#     async def worker(task_id: str, value: int):
#         # Create a sample context unique to this worker
#         ctx = MIContext(sample_id=task_id)
#         ctx.buffer["value"] = value

#         with bind_context(ctx):
#             # Yield execution back to the event loop to simulate interleaving token generations
#             await asyncio.sleep(0.01)

#             # Read active context back from ContextVars
#             current = get_current_context()
#             assert current is not None
#             assert current.sample_id == task_id
#             results[task_id] = current.buffer["value"]

#     # Run 10 concurrent tasks simultaneously
#     tasks = [worker(f"sample_{i}", i * 10) for i in range(10)]
#     await asyncio.gather(*tasks)

#     # Verify no state was overwritten across concurrent executions
#     for i in range(10):
#         assert results[f"sample_{i}"] == i * 10




def test_sync_context_binding():
    """Verifies synchronous context binding with `with bind_context(ctx)`."""
    assert get_current_context() is None

    ctx = MIContext(sample_id="sync_sample")
    with bind_context(ctx):
        current = get_current_context()
        assert current is not None
        assert current.sample_id == "sync_sample"

    # Verify context resets back to None on exit
    assert get_current_context() is None


@pytest.mark.asyncio
async def test_async_context_binding():
    """Verifies asynchronous context binding with `async with bind_context(ctx)`."""
    assert get_current_context() is None

    ctx = MIContext(sample_id="async_sample")
    async with bind_context(ctx):
        current = get_current_context()
        assert current is not None
        assert current.sample_id == "async_sample"

    assert get_current_context() is None


@pytest.mark.asyncio
async def test_async_context_isolation():
    """Verifies that concurrent asyncio tasks maintain isolated MIContexts across yields."""
    results = {}

    async def worker(task_id: str, value: int):
        ctx = MIContext(sample_id=task_id)
        ctx.buffer["value"] = value

        async with bind_context(ctx):
            # Yield execution to simulate interleaving token generations
            await asyncio.sleep(0.01)

            current = get_current_context()
            assert current is not None
            assert current.sample_id == task_id
            results[task_id] = current.buffer["value"]

    tasks = [worker(f"sample_{i}", i * 10) for i in range(10)]
    await asyncio.gather(*tasks)

    for i in range(10):
        assert results[f"sample_{i}"] == i * 10


def test_unbound_context_returns_none():
    """Verifies that accessing get_current_context outside a bound context safely returns None."""
    assert get_current_context() is None
