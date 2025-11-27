"""Scheduling abstractions on PyTorch Inductor WrapperCodeGen level.

Attributes:
    DEFAULT_STREAM: Name of the default CUDA Stream on the final generated Python code.
    DEFAULT_STREAM_IDX: Index number of the default CUDA Stream in `torchsched` internal passes.
    STREAM_NAME_TEMPLATE: Python string template to generate stream names. Can be used as:

            idx: int = ...
            stream = STREAM_NAME_TEMPLATE.format(stream_idx=idx)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from torch._inductor.codegen.wrapper import EnterDeviceContextManagerLine
from torch._inductor.codegen.wrapper import ExitDeviceContextManagerLine
from torch._inductor.codegen.wrapper import IndentedBuffer
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen
from torch._inductor.codegen.wrapper import WrapperLine
from torch._inductor.virtualized import V

import apex.contrib.torchsched.config as config
from apex.contrib.torchsched.inductor._utils import DEFAULT_STREAM
from apex.contrib.torchsched.inductor._utils import ENTRANCE_EVENT
from apex.contrib.torchsched.inductor._utils import STREAM_NAME_TEMPLATE
from apex.contrib.torchsched.inductor._utils import get_stream_name

if TYPE_CHECKING:
    from torch._inductor.graph import GraphLowering
    from torch._inductor.ir import GraphPartitionSignature

    from apex.contrib.torchsched.inductor.event import CudaEventSym


@dataclasses.dataclass
class EnterDeviceContextManagerWithStreamInfoLine(EnterDeviceContextManagerLine):
    """Enter a CUDA device context and allocate required side streams.

    Note:
        - The number of allocated streams is controlled by :attr:`torchsched.config.num_streams`;
    """

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate context switching and stream allocation code."""
        if V.graph.cpp_wrapper:
            super().codegen(code)
        else:
            super().codegen(code)
            code.writeline(f"{DEFAULT_STREAM} = torch.cuda.current_stream()")
            code.writeline(f"{ENTRANCE_EVENT} = {DEFAULT_STREAM}.record_event()")

            code.writeline(
                "from apex.contrib.torchsched.inductor._utils import get_cuda_stream_pool"
            )
            code.writeline(
                f"cuda_stream_pool = get_cuda_stream_pool(device={self.device_idx}, "
                f"pool_size={config.num_streams})",
            )

            for i in range(1, config.num_streams):
                code.writeline(
                    f"{STREAM_NAME_TEMPLATE.format(stream_idx=i)} = cuda_stream_pool.acquire()",
                )


@dataclasses.dataclass
class ExitDeviceContextManagerWithStreamInfoLine(ExitDeviceContextManagerLine):
    """Exit a CUDA device context and release allocated streams."""

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate context switching and stream release code."""
        for i in range(1, config.num_streams):
            code.writeline(
                f"cuda_stream_pool.release({STREAM_NAME_TEMPLATE.format(stream_idx=i)})",
            )
        if not V.graph.cpp_wrapper:
            code.do_unindent()


@dataclasses.dataclass
class EnterCudaStreamContextLine(WrapperLine):
    """Enter a context executed by respective CUDA Stream and insert necessary syncs.

    Attributes:
        wrapper: The code-gen wrapper of the current compilation phase.
        stream_idx: The index number corresponds to the entering CUDA Stream context.
        upstream_events: Names of CUDA Events that the current stream should be waiting for before
            the stream switching.
        buffers_from_other_streams: Name of buffers produced by other CUDA Streams. Those buffers
            should be recorded to the current stream to avoid accidental memory free.
        buffers_requiring_device_check: Name of buffers that might not be on CUDA devices and
            require runtime device checking before recording stream to them.
    """

    stream_idx: int

    def __post_init__(self) -> None:
        """Track buffers have been recorded on this stream to reduce duplicate recording."""
        self.buffers_recorded_on_this_stream: set[str] = set()

    def codegen(self, code: IndentedBuffer) -> None:
        """Generate stream switching and buffer recording code."""
        code.writeline(f"with torch.cuda.stream({get_stream_name(self.stream_idx)}):")
        code.do_indent()

        # [NOTE] The 3-indent-level assertion
        #
        #     Indent level 1: Inductor wrapper call indent
        #         Indent level 2: Device guard context indent
        #             Indent level 3: CUDA Stream context indent
        #
        # Over or under indenting usually means that :meth:`MultiCudaStreamScheduler.codegen`
        # introduced bugs on stream context switching. This check also applies to stream context
        # exiting, as in :meth:`ExitCudaStreamContextLine.codegen`.
        assert code._indent == 3


@dataclasses.dataclass
class ExitCudaStreamContextLine(WrapperLine):
    """Generate code to exit the current stream context.

    Note:
        Most attributes and checking logics of this class have been moved to
        :meth:`MultiStreamWrapperCodeGen.codegen_cuda_stream_exit`. We preserve this data structure
        because the checking and unindent should be generated in the latter phase of code-gen.
    """

    def codegen(self, code: IndentedBuffer) -> None:
        """Check indentation level and exit the current stream context."""
        assert code._indent == 3  # See :note:`The 3-indent-level assertion` above.
        code.do_unindent()


class MultiStreamWrapperCodegen(PythonWrapperCodegen):
    """Wrapper code generator for graph scheduling."""

    def __init__(self) -> None:
        """Construct a code-gen wrapper and disable raw stream caching.

        Note:
            The :meth:`write_get_raw_stream` method processed in this constructor is invoked from
            literally everywhere throughout the Inductor stack, but the current
            :meth:`PythonWrapperCodegen.write_get_raw_stream` is LRU-cached and always returns a
            const raw stream name. This is not what we wanted in a multi-stream environment. Thus
            we need to re-patch this function in instance initialization.
        """
        super().__init__()
        self.write_get_raw_stream = self._write_get_raw_stream

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str,
        parent_wrapper: MultiStreamWrapperCodegen,
        partition_signatures: GraphPartitionSignature | None = None,
    ) -> MultiStreamWrapperCodegen | SubgraphPythonWrapperCodegen:
        """Instantiate a wrapper codegen for an Inductor graph or a subgraph."""
        if is_subgraph:
            assert subgraph_name is not None
            assert parent_wrapper is not None
            return SubgraphPythonWrapperCodegen(
                subgraph_name,
                parent_wrapper,
                partition_signatures,
            )
        return MultiStreamWrapperCodegen()

    def _write_get_raw_stream(self, device_idx: int, graph: GraphLowering | None = None) -> str:
        self.write_triton_header_once()
        if (current_stream_name := V.graph.scheduler.current_stream_name) is not None:
            name = f"{current_stream_name}_raw"
            self.writeline(f"{name} = {current_stream_name}.cuda_stream")
        else:
            name = f"stream{device_idx}"
            self.writeline(f"{name} = get_raw_stream({device_idx})")
        return name

    def codegen_graph_nvtx_range_push(self, post_grad_graph_id: int) -> None:
        """Generate NVTX range push for graph."""
        self.writeline(f"torch.cuda.nvtx.range_push('graph {post_grad_graph_id}')")

    def codegen_graph_nvtx_range_pop(self) -> None:
        """Generate NVTX range pop for graph."""
        self.writeline("torch.cuda.nvtx.range_pop()")

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        """Generate data structure for device guard context.

        Note:
            Refer to :class:`EnterDeviceContextManagerWithStreamInfoLine` doc for more details.
        """
        self.writeline(
            EnterDeviceContextManagerWithStreamInfoLine(
                device_idx,
                self.last_seen_device_guard_index,
            ),
        )
        self.last_seen_device_guard_index: int = device_idx

    def codegen_device_guard_exit(self) -> None:
        """Generate data structure for exiting device guard context."""
        self.writeline(ExitDeviceContextManagerWithStreamInfoLine())

    def codegen_cuda_stream_enter(
        self,
        stream_idx: int,
        upstream_events: set[CudaEventSym],
        buffers_from_other_streams: set[str],
        buffers_requiring_device_check: set[str] | None = None,
    ) -> EnterCudaStreamContextLine:
        """Generate data structure for entering a CUDA Stream context.

        Args:
            stream_idx: The index number of the entering CUDA Stream context.
            upstream_events: Names of CUDA Events that the current stream should be waiting for
                before the stream switching. This is usually the events that are generated by the
                previous stream context.
            buffers_from_other_streams: Name of buffers produced by other CUDA Streams. Those
                buffers should be recorded to the current stream to avoid accidental memory free.
            buffers_requiring_device_check: Name of buffers that might not be on CUDA devices and
                require runtime device checking before recording stream to them.

        Note:
            - Refer to :class:`EnterCudaStreamContextLine` for argument specifications;
            - Once entered a context, the stream associated with this context will also be recorded
              such that kernels in subsequent code-gen can get the correct stream index.

        Raises:
            ValueError: If this function is called while the previous stream context isn't exited.
        """
        if (current_stream_name := V.graph.scheduler.current_stream_name) is not None:
            raise ValueError(
                f"Nested stream context switching: {current_stream_name} -> "
                f"{get_stream_name(stream_idx)}",
            )
        ctx_entrance = EnterCudaStreamContextLine(stream_idx=stream_idx)
        self.writeline(ctx_entrance)
        self.codegen_buffers_record_stream(
            buffers=buffers_from_other_streams,
            stream_idx=stream_idx,
            buffers_requiring_device_check=buffers_requiring_device_check,
        )
        ctx_entrance.buffers_recorded_on_this_stream |= buffers_from_other_streams
        self.codegen_events_wait_stream(
            events=upstream_events,
            stream_idx=stream_idx,
        )
        return ctx_entrance

    def codegen_cuda_stream_exit(self) -> None:
        """Generate data structure for exiting a CUDA Stream context."""
        self.writeline(ExitCudaStreamContextLine())

    def codegen_events_wait_stream(self, events: set[CudaEventSym], stream_idx: int) -> None:
        """Generate data structure for syncing hanging CUDA Events with certain stream.

        Args:
            events: Symbols of the events that need to be synchronized with the given stream.
            stream_idx: Index of the CUDA stream to synchronize the events with.
        """
        for event in events:
            self.writeline(event.wait(stream_idx))

    def codegen_buffers_record_stream(
        self,
        buffers: set[str],
        stream_idx: int,
        buffers_requiring_device_check: set[str] | None = None,
    ) -> None:
        """Generate data structure for recording steam on return tensors before program exit.

        Args:
            buffers: Names of buffers that need to be recorded on the given stream.
            stream_idx: Index of the CUDA stream to record the buffers to.
            buffers_requiring_device_check: Name of buffers that might not be on CUDA devices and
                require runtime device checking before recording stream to them. If not provided,
                buffers will be recorded to the given stream without runtime device checking.
        """
        for buff in buffers:
            prefix = (
                f"if {buff}.is_cuda: "
                if buffers_requiring_device_check and buff in buffers_requiring_device_check
                else ""
            )
            self.writeline(f"{prefix}{buff}.record_stream({get_stream_name(stream_idx)})")
