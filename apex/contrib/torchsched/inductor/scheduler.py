"""Scheduling abstractions on PyTorch Inductor Scheduler level."""

from __future__ import annotations

import collections
import copy
import itertools
import re
from typing import TYPE_CHECKING
from typing import cast

import torch
import torch._inductor.config as inductor_config
from torch._inductor import ir
from torch._inductor.dependencies import WeakDep
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.scheduler import ExternKernelSchedulerNode  # noqa: F401
from torch._inductor.scheduler import ForeachKernelSchedulerNode  # noqa: F401
from torch._inductor.scheduler import FusedSchedulerNode
from torch._inductor.scheduler import NopKernelSchedulerNode
from torch._inductor.scheduler import Scheduler
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import device_need_guard
from torch._inductor.virtualized import V

import apex.contrib.torchsched.config as config
from apex.contrib.torchsched.inductor.event import CudaEventFactory
from apex.contrib.torchsched.inductor.event import CudaEventSym
from apex.contrib.torchsched.inductor.wrapper import DEFAULT_STREAM_IDX
from apex.contrib.torchsched.inductor.wrapper import EnterCudaStreamContextLine

if TYPE_CHECKING:
    from apex.contrib.torchsched.inductor.wrapper import MultiStreamWrapperCodegen  # noqa: F401


schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")


class MultiCudaStreamScheduler(Scheduler):
    """Scheduling post-fusion graph with multi-stream awareness.

    This class introduced a new optimization pass on top of the Inductor :class:`Scheduler`. I.e.,
    it firstly searches for the longest critical path in the given compute graph, currently using
    the path depth as a proxy of execution cost. Then it executes the non-critical computations in
    parallel with the critical path computations by launching them to side CUDA Streams, with the
    goal of scheduling critical path computations back-to-back while saturating GPU resources at
    runtime.

    Args:
        operations: A list of Inductor IR nodes representing fused computations.
    """

    def __init__(self, operations: list[ir.Operation]) -> None:
        """Construct a scheduler object from a list of Inductor IR nodes.

        Refer to :class:`MultiCudaStreamScheduler` doc for argument specification.
        """
        super().__init__(operations)
        self.event_factory = CudaEventFactory()
        self.buff_to_event: dict[str, CudaEventSym] = collections.defaultdict(
            lambda: self.event_factory.get_sym_event(),
        )
        self.unjoined_events: dict[int, set[CudaEventSym]] = collections.defaultdict(set)
        self.buffers_requiring_device_check: set[str] = set()
        self.current_stream: int | None = None
        self.current_ctx_entrance: EnterCudaStreamContextLine | None = None
        self.current_ctx_ending_event: CudaEventSym | None = None
        self.schedule_multi_cuda_streams()

    def _schedule_execution(self) -> None:
        # A linear representation of parallelizable execution schedule.
        #
        #    schedule := List[producer, consumer]
        #    producer := List | Set | Node
        #    consumer := Set | Node
        #    List     := Sequential execution
        #    Set      := Emit parallel primitives on set opening, sync parallel results on set
        #                close
        #
        # NOTE: Greedy construction of this data structure is suboptimal, very likely to miss
        # parallel opportunities, thus this design is kept here but not used.
        producer_graph = None
        consumer_graph: set[BaseSchedulerNode] = set()
        produced_dependency: set[str] = set()
        consuming_dependency: set[str] = set()

        for node in self.nodes:
            consuming_dependency |= {d.name for d in node.unmet_dependencies}
            if consuming_dependency - produced_dependency:
                # Turn current consumers to producers.
                if producer_graph is None:
                    producer_graph = copy.copy(consumer_graph)
                else:
                    producer_graph = [copy.copy(producer_graph), copy.copy(consumer_graph)]
                produced_dependency |= set().union(*[n.get_buffer_names() for n in consumer_graph])
                consumer_graph.clear()
                consuming_dependency.clear()
            consumer_graph.add(node)
            consuming_dependency = set().union(*[n.get_buffer_names() for n in consumer_graph])

        if consumer_graph:
            execution_schedule = [copy.copy(producer_graph), copy.copy(consumer_graph)]
        else:
            execution_schedule = [copy.copy(producer_graph)]

        self.execution_schedule = execution_schedule

    def debug_str_short(self, node: BaseSchedulerNode) -> str:
        """Generate short string representing scheduler node's calling function or indices."""
        if node.is_extern() and isinstance(node.node, ir.MultiOutput):
            kernel_str = node.node.codegen_list_tuple_access(
                basename="getitem",
                indices=node.node.indices,
            )
            return f"{node.get_name()} ({kernel_str})"
        elif node.is_extern():
            kernel_name = node.node.get_kernel_name() or str(node.node.op_overload)
            return f"{node.get_name()} ({kernel_name})"
        else:
            return node.get_name()

    def get_last_event(self, events: set[CudaEventSym]) -> CudaEventSym:
        """Identify the latest generated CUDA event among all given events."""
        return sorted(events, reverse=True)[0]  # CudaEventSym is total-ordering.

    def schedule_multi_cuda_streams(self) -> None:
        """Assign each fused Inductor IR nodes with the CUDA Stream to be launched to."""
        if not self.nodes:
            # Empty graphs are sent to compiler in very rare circumstances. Just Skip scheduling.
            return

        buf_originate: dict[str, BaseSchedulerNode] = {}
        node_users: dict[BaseSchedulerNode, set[BaseSchedulerNode]] = collections.defaultdict(set)
        for node in self.nodes:
            for n in node.get_buffer_names():
                buf_originate[n] = node
        for node in self.nodes:
            for d in node.unmet_dependencies:
                assert d.name in buf_originate
                node_users[buf_originate[d.name]].add(node)

        critical_path_per_depth: dict[int, set[BaseSchedulerNode]] = collections.defaultdict(set)
        node_depth: dict[BaseSchedulerNode, int] = collections.defaultdict(lambda: -1)

        def visit(node: BaseSchedulerNode, depth: int, prev: set[BaseSchedulerNode]) -> None:
            if node_depth[node] < depth:
                node_depth[node] = depth
                path = prev | {node}
                if len(critical_path_per_depth[depth]) < len(path):
                    critical_path_per_depth[depth] = path
                for user in node_users[node]:
                    visit(user, depth + 1, path)

        graph_entries = [n for n in self.nodes if not n.unmet_dependencies]
        for entry in graph_entries:
            visit(entry, depth=1, prev=set())

        max_depth, longest_critical_path = sorted(critical_path_per_depth.items(), reverse=True)[0]

        # Allocate CUDA Streams for each fused node:
        # - Critical path nodes go to the default stream
        # - Nodes without GPU operations (currently only covered getitem nodes) go to their
        #   producer's stream
        # - Other nodes go to a set of pre-defined number of side-streams in a round-robin manner
        num_streams = config.num_streams
        if num_streams == 1:
            node_to_stream = {node: DEFAULT_STREAM_IDX for node in self.nodes}
        else:
            node_to_stream = {}
            side_stream_indices = itertools.cycle(range(1, num_streams))
            for node in self.nodes:
                if node in longest_critical_path:
                    node_to_stream[node] = DEFAULT_STREAM_IDX
                elif node.is_extern() and isinstance(node.node, ir.MultiOutput):
                    assert len(node.unmet_dependencies) == 1
                    producer = buf_originate[next(iter(node.unmet_dependencies)).name]
                    node_to_stream[node] = node_to_stream[producer]
                else:
                    node_to_stream[node] = next(side_stream_indices)
        self.node_to_stream = node_to_stream

        # Also remember buffer originate streams.
        buff_to_stream = {}
        for node, stream_idx in node_to_stream.items():
            for buf_name in node.get_buffer_names():
                buff_to_stream[buf_name] = stream_idx
        self.buff_to_stream = buff_to_stream

        schedule_log.debug(f"{' Multi-CUDA-Stream scheduling results ':=^79}")
        schedule_log.debug("Post-fusion graph depth: %d", max_depth)
        schedule_log.debug("Total number of allocated CUDA Streams: %d", num_streams)
        schedule_log.debug(f"{' Critical path ':-^79}")
        for node in self.nodes:
            if node in longest_critical_path:
                schedule_log.debug("- %s", self.debug_str_short(node))
        schedule_log.debug(f"{' Stream assignments of other nodes ':-^79}")
        for node, stream_idx in node_to_stream.items():
            if node not in longest_critical_path:
                schedule_log.debug("- %s -> Stream %d", self.debug_str_short(node), stream_idx)

    def prune_unjoined_events(self, synced_upstream_events: dict[int, set[CudaEventSym]]) -> None:
        """Remove synced CUDA Events on specific CUDA Streams.

        Args:
            sync_upstream_events: A dict that maps event names to stream indices. Each key-value
                pair represents a event on the corresponding stream has been properly synced so it
                do not need to be synced again at the end of the program.

        Raises:
            ValueError: If any of the event name was not registered to its corresponding stream.
        """
        for stream_idx, events in synced_upstream_events.items():
            if stream_idx != DEFAULT_STREAM_IDX:
                self.unjoined_events[stream_idx] -= events

    def get_final_events_to_sync(self) -> set[CudaEventSym]:
        """Return the CUDA Events that need to be synced at the end of the program.

        Raises:
            ValueError: If there is hanging event on the default stream. This usually means the
                user didn't properly use :meth:`add_unjointed_event` to register hanging events.
        """
        if self.unjoined_events.get(DEFAULT_STREAM_IDX):
            raise ValueError(
                f"Unexpected {self.unjoined_events[DEFAULT_STREAM_IDX]=} on default stream",
            )
        events_to_sync = set()
        for stream, events in self.unjoined_events.items():
            if len(events) == 0:
                schedule_log.debug(f"All events on stream{stream} have been consumed")
                continue
            last_event = self.get_last_event(events)
            if 1 < len(events):
                schedule_log.debug(
                    f"Seeing multiple hanging {events=} on stream{stream}, scheduling the "
                    f"{last_event=} to sync",
                )
            else:
                schedule_log.debug(
                    f"Scheduling the {last_event=} on stream{stream} to sync",
                )
            events_to_sync.add(last_event)
        return events_to_sync

    def clear_unjoined_events(self) -> None:
        """Clear handing event syncs registered by :meth:`add_unjointed_event`."""
        self.unjoined_events.clear()

    def register_downstream_event(
        self,
        node: BaseSchedulerNode,
    ) -> CudaEventSym:
        """Register one CUDA event indicating node execution complete.

        For ordinary Inductor IR nodes, the completion event is newly created using an internal
        event counter. For Inductor no-op nodes, the last event corresponding to its inputs will be
        used instead.

        Args:
            node: The Inductor IR node to generate completion event for.

        Returns:
            The name of the completion event.

        Raises:
            ValueError: If this function is called out side of any stream context.
        """
        if isinstance(node, NopKernelSchedulerNode) and node.unmet_dependencies:
            upstream_events = set()
            for dep in node.unmet_dependencies:
                assert dep.name in self.buff_to_event
                upstream_events.add(self.buff_to_event[dep.name])
            assert 1 <= len(upstream_events)
            downstream_event = self.get_last_event(upstream_events)
            for buff in node.get_buffer_names():
                self.buff_to_event[buff] = downstream_event
        else:
            for i, buff in enumerate(sorted(node.get_buffer_names())):
                if i == 0:
                    downstream_event = self.buff_to_event[buff]
                else:
                    self.buff_to_event[buff] = downstream_event
            if (node_stream := self.node_to_stream[node]) != DEFAULT_STREAM_IDX:
                self.unjoined_events[node_stream].add(downstream_event)
            V.graph.wrapper_code.writeline(downstream_event.record(node_stream))
        return downstream_event

    def get_cross_stream_dependencies(
        self,
        node: BaseSchedulerNode,
        prune_unjointed_events: bool = False,
    ) -> tuple[set[CudaEventSym], set[str]]:
        """Get CUDA Event and buffer dependencies of an IR node.

        Args:
            node: The Inductor IR node to generate code for.
            prune_unjointed_events: If set to True, the returned CUDA Events will be considered
                synced and no need to be synced again before program exiting. (Default: True)

        Returns:
            upstream_events: A set of CUDA Event symbols, these events need to be synced before
                executing `node`'s code.
            buffer_from_other_streams: A set of buffer names, these buffers need to be recorded on
                the CUDA Stream that `node` is running on.
        """
        assert node in self.node_to_stream

        # Process cross-cuda-stream dependencies.
        node_stream = self.node_to_stream[node]
        events_on_stream: dict[int, set[CudaEventSym]] = collections.defaultdict(set)
        buffers_from_other_streams = set()
        if not node.unmet_dependencies and node_stream != DEFAULT_STREAM_IDX:
            # Graph entries on side streams should wait upon the main stream entrance.
            entrance_event = self.event_factory.get_entrance_event()
            events_on_stream[DEFAULT_STREAM_IDX].add(entrance_event)
        for dep in node.read_writes.reads:
            buff = dep.name  # To track stream number and cuda events.
            buff_real = self.mutation_real_name.get(buff, buff)  # The real name in code.
            if dep not in node.unmet_dependencies and not isinstance(dep, WeakDep):
                # Materialized dependencies should be recorded on this stream.
                buffers_from_other_streams.add(buff_real)
                # The scalar tensor argument `dropout_p` of SDPA backward kernels might be on CUDA
                # or CPU devices depending on execution scenario. To ensure program correctness we
                # add a runtime check for it.
                #
                # TODO (@davidli): Remove this ad-hoc checking once PyTorch fix SDPA and
                # MultiOutputLayout issues.
                if node.is_extern() and re.match(
                    r"aten._scaled_dot_product_.*_attention_backward",
                    str(node.node.op_overload),
                ):
                    self.buffers_requiring_device_check.add(buff_real)
                continue
            elif isinstance(dep, WeakDep):
                # Skip unmaterialized dependencies.
                continue
            assert buff in self.buff_to_event
            assert buff in self.buff_to_stream
            buff_event = self.buff_to_event[buff]
            buff_stream = self.buff_to_stream[buff]
            events_on_stream[buff_stream].add(buff_event)
            if buff_stream != node_stream:
                if node.is_extern() and isinstance(node.node, ir.MultiOutput):
                    assert len(node.read_writes.reads) == 1
                    buff_real = node.node.codegen_list_tuple_access(
                        basename=buff_real,
                        indices=node.node.indices,
                    )
                    self.buffers_requiring_device_check |= {buff_real, node.node.get_name()}
                buffers_from_other_streams.add(buff_real)

        # Should only wait for the latest event from each stream.
        upstream_events = set()
        for stream, events in events_on_stream.items():
            if stream != node_stream:
                last_event = self.get_last_event(events)
                upstream_events.add(last_event)

        # Side-effect: Populated cross-stream events get pruned optionally.
        if prune_unjointed_events:
            self.prune_unjoined_events(events_on_stream)

        return upstream_events, buffers_from_other_streams

    def generate_stream_ctx_enter(self, node: BaseSchedulerNode) -> None:
        """Code-gen to enter the Stream context assigned to node."""
        assert not isinstance(node, NopKernelSchedulerNode)
        wrapper_code = cast("MultiStreamWrapperCodegen", V.graph.wrapper_code)
        upstream_events, buffers_from_other_streams = self.get_cross_stream_dependencies(node)
        node_stream = self.node_to_stream[node]
        self.current_ctx_entrance = wrapper_code.codegen_cuda_stream_enter(
            stream_idx=node_stream,
            upstream_events=upstream_events,
            buffers_from_other_streams=buffers_from_other_streams,
            buffers_requiring_device_check=self.buffers_requiring_device_check,
        )

    def generate_stream_ctx_exit(self) -> None:
        """Code-gen to exit from the current Stream context."""
        assert self.current_stream is not None
        wrapper_code = cast("MultiStreamWrapperCodegen", V.graph.wrapper_code)
        wrapper_code.codegen_cuda_stream_exit(
            stream_idx=self.current_stream,
        )
        self.current_ctx_entrance = None

    def propagate_cross_stream_dependencies(self, node: BaseSchedulerNode) -> None:
        """Move input node's dependencies to the entrance of current CUDA Stream context.

        If node is scheduled in the middle of a stream context, its dependencies should be properly
        synced before entering this context. This function extracts `node`'s dependencies and move
        them to the data structure that represents the entrance of current stream context.

        Args:
            node: The Inductor IR node to generate code for. This node must have an assigned stream
                in :meth:`schedule_multi_cuda_streams`.
        """
        assert self.current_ctx_entrance is not None
        upstream_events, buffers_from_other_streams = self.get_cross_stream_dependencies(node)
        self.current_ctx_entrance.upstream_events |= upstream_events
        self.current_ctx_entrance.buffers_from_other_streams |= buffers_from_other_streams

    def generate_stream_ctx_switching(self, node: BaseSchedulerNode) -> None:
        """Generate stream entering and exiting to properly run node in a multi-stream scenario.

        Stream context switching is only generated if `node`'s assigned stream is different from
        the previous node's stream. If the node is a no-op, its code will be generated in the same
        context of previous node.
        """
        assert node in self.node_to_stream
        stream = None if isinstance(node, NopKernelSchedulerNode) else self.node_to_stream[node]
        if stream == self.current_stream:
            if stream is not None:
                self.propagate_cross_stream_dependencies(node)
            return
        elif self.current_stream is not None and stream is None:
            # Don't generate ctx switching. Memory plaining code (e.g., delete buffers) on current
            # node goes to previous stream ctx.
            return
        elif self.current_stream is None and stream is not None:
            # Enter new ctx, update current stream status.
            self.generate_stream_ctx_enter(node)
            self.current_stream = stream
        else:
            # Switching from previous stream ctx to the new stream ctx.
            self.generate_stream_ctx_exit()
            self.generate_stream_ctx_enter(node)
            self.current_stream = stream

    def codegen(self) -> None:
        """Generate Python code for each of the Scheduler IR nodes.

        Note:
            The overall `torch.compile` code-gen is a multi-pass process, which means that this
            method doesn't necessarily generate final program strings for every IR nodes. For
            certain types of IRs, e.g., those involve memory allocation/deletion and CUDA Stream
            switching, this method only generates respective data structures, and the final
            code-gen is delegated to :meth:`WrapperCodeGen.codegen` using information form these
            data structures.

        Raises:
            AssertionError: If any of the conditions met
                * A node need to switch device context but it didn't include device information;
                * A node contains at least one non-weak dependence that was not seen in the
                  :meth:`schedule_multi_cuda_streams` pass;
                * A node contains at least one non-weak cross-stream dependence that the
                  corresponding event was not generated before that point;
                * The fused compute graph contains :class:`ForeachKernelSchedulerNode` but the
                  target backend doesn't support SIMD scheme.
        """
        wrapper_code = cast("MultiStreamWrapperCodegen", V.graph.wrapper_code)
        for node in self.nodes:
            try:
                schedule_log.debug(
                    "Generating code for node %s with estimated runtime %f",
                    node.get_name(),
                    node.get_estimated_runtime(),
                )
            except Exception:
                schedule_log.debug(
                    "Generating code for node %s with estimated runtime 0.0",
                    node.get_name(),
                )

            self.enter_context(node)

            if not isinstance(node, NopKernelSchedulerNode) and (device := node.get_device()):
                if device != self.current_device or node.is_extern() or node.is_template():
                    self.flush()
                if device != self.current_device:
                    if self.current_device and device_need_guard(
                        self.current_device.type,
                    ):
                        wrapper_code.codegen_device_guard_exit()
                    if device_need_guard(device.type):
                        assert device.index is not None, "device should have an index"
                        wrapper_code.codegen_device_guard_enter(device.index)
                    self.current_device: torch.device | None = device

            self.generate_stream_ctx_switching(node)
            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                node, *epilogue = node.get_nodes()
                self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                node = cast("ExternKernelSchedulerNode", node)
                self.codegen_extern_call(node)
            elif node.is_foreach():
                node = cast("ForeachKernelSchedulerNode", node)
                backend_ = self.get_backend(device)
                from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
                from torch._inductor.codegen.simd import SIMDScheduling

                if isinstance(backend_, (SIMDScheduling, CUDACombinedScheduling)):
                    backend = backend_
                else:
                    raise AssertionError(f"{type(self)=}")
                backend.codegen_combo_kernel(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_node(node)
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.mark_run()

            if inductor_config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_buffer_names())
            self.completed_operations.update(node.get_operation_names())
            self.current_ctx_ending_event = self.register_downstream_event(node)

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if device is not None and self.get_backend(device).ready_to_flush():
                    self.flush()

        if self.current_device and device_need_guard(self.current_device.type):
            # Exit the last stream context.
            if self.current_ctx_entrance:
                self.generate_stream_ctx_exit()
            # Record the default stream on buffers from other streams.
            side_stream_buffers = set()
            for output in V.graph.get_output_names():
                if self.buff_to_stream.get(output, DEFAULT_STREAM_IDX) != DEFAULT_STREAM_IDX:
                    side_stream_buffers.add(output)
            wrapper_code.codegen_record_buff_from_side_streams(
                buffers=side_stream_buffers,
                buffers_requiring_device_check=self.buffers_requiring_device_check,
            )
            # Sync hanging events from other streams.
            if events_to_sync := self.get_final_events_to_sync():
                wrapper_code.codegen_sync_unjoined_cuda_events(events_to_sync)
            # exit the outermost CUDA device guard. this is
            # important for nested indentation codegen-ing.
            wrapper_code.codegen_device_guard_exit()

        self.flush()
