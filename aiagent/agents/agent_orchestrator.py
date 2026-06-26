import json
import logging
import typing as tp

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aiagent.agents.agent_state import AgentState
from aiagent.agents.prompt_builder import (
    build_intent_classification_prompt,
    build_langchain_messages,
    build_planning_system_message,
    build_response_synthesis_prompt,
)
from aiagent.constants import DEFAULT_INTENT, IntentType, MessageRole
from aiagent.schemas.agent_io import (
    ConversationMessage,
    PlannedToolCall,
    ToolExecutionReport,
    ToolExecutionResult,
)
from aiagent.tools.tool_registry import TOOL_NAME_RUN_MODEL_INFERENCE

logger = logging.getLogger(__name__)


DICT_SCENARIO_TOOL_NAMES: tp.Dict[IntentType, tp.List[str]] = dict(
    {
        IntentType.MODEL_INFERENCE: list([TOOL_NAME_RUN_MODEL_INFERENCE]),
        IntentType.GENERAL_QA: list(),
    }
)


class AgentOrchestrator:
    """
    사용자 의도를 파악하고 의도에 맞는 도구 시나리오를 실행하는 통합 에이전트.
    외부 인터페이스(run)를 유지하기 위해 내부 흐름은 캡슐화되어 있다.
    """

    def __init__(self, model_llm: ChatOpenAI, list_tools: tp.List[BaseTool]) -> None:
        self.model_llm: ChatOpenAI = model_llm
        self.dict_tool_by_name: tp.Dict[str, BaseTool] = dict(
            (model_tool.name, model_tool) for model_tool in list_tools
        )
        self.graph_workflow: CompiledStateGraph = self._build_workflow()

    def _resolve_scenario_tools(self, model_intent: IntentType) -> tp.List[BaseTool]:
        """시나리오 도구 이름을 주입된 도구 구현으로 변환한다."""
        list_names: tp.List[str] = DICT_SCENARIO_TOOL_NAMES.get(model_intent, list())
        return list(
            self.dict_tool_by_name[str_name]
            for str_name in list_names
            if str_name in self.dict_tool_by_name
        )

    def _build_workflow(self) -> CompiledStateGraph:
        """LangGraph 워크플로우 정의."""
        graph_builder: StateGraph = StateGraph(AgentState)

        graph_builder.add_node("analyze_intent", self._node_analyze_intent)
        graph_builder.add_node("plan_tools", self._node_plan_tools)
        graph_builder.add_node("execute_tools", self._node_execute_tools)
        graph_builder.add_node("format_response", self._node_format_response)

        graph_builder.add_edge(START, "analyze_intent")
        graph_builder.add_edge("analyze_intent", "plan_tools")
        graph_builder.add_edge("plan_tools", "execute_tools")
        graph_builder.add_edge("execute_tools", "format_response")
        graph_builder.add_edge("format_response", END)

        return graph_builder.compile()

    async def _node_analyze_intent(self, state: AgentState) -> tp.Dict[str, tp.Any]:
        """사용자 의도 및 필요한 파라미터 추출 (대화 이력 참고)."""
        list_invoke_args: tp.List[BaseMessage] = build_langchain_messages(state.list_messages)
        str_prompt: str = build_intent_classification_prompt(state.str_user_input)
        list_invoke_args.append(HumanMessage(content=str_prompt))

        model_response = await self.model_llm.ainvoke(list_invoke_args)
        model_intent, dict_params = self._parse_intent_response(str(model_response.content))

        logger.info("intent classified: %s", model_intent.value)
        return dict(model_intent=model_intent, dict_extracted_params=dict_params)

    def _parse_intent_response(
        self, str_content: str
    ) -> tp.Tuple[IntentType, tp.Dict[str, tp.Any]]:
        """LLM 응답 본문에서 의도와 파라미터를 파싱. 실패 시 기본 의도로 폴백."""
        str_payload: str = self._strip_code_fence(str_content)
        try:
            dict_parsed: tp.Dict[str, tp.Any] = json.loads(str_payload)
        except json.JSONDecodeError:
            logger.warning("intent payload is not valid JSON, falling back to default intent")
            return DEFAULT_INTENT, dict()

        str_raw_intent: str = str(dict_parsed.get("intent", DEFAULT_INTENT.value))
        dict_params: tp.Dict[str, tp.Any] = dict(dict_parsed.get("params", dict()))
        return self._to_intent_type(str_raw_intent), dict_params

    @staticmethod
    def _strip_code_fence(str_content: str) -> str:
        """LLM이 코드펜스로 감싼 JSON을 정리."""
        str_stripped: str = str_content.strip()
        if str_stripped.startswith("```"):
            str_stripped = str_stripped.strip("`")
            if str_stripped.startswith("json"):
                str_stripped = str_stripped[len("json"):]
        return str_stripped.strip()

    @staticmethod
    def _to_intent_type(str_raw_intent: str) -> IntentType:
        """문자열 의도를 IntentType으로 매핑. 미지원 값은 기본 의도로 폴백."""
        try:
            return IntentType(str_raw_intent)
        except ValueError:
            logger.warning("unknown intent value '%s', falling back to default", str_raw_intent)
            return DEFAULT_INTENT

    async def _node_plan_tools(self, state: AgentState) -> tp.Dict[str, tp.Any]:
        """의도에 맞는 Tool을 바인딩하고 어떤 도구를 호출할지 계획."""
        model_intent: IntentType = state.model_intent or DEFAULT_INTENT
        list_tools: tp.List[BaseTool] = self._resolve_scenario_tools(model_intent)

        if not list_tools:
            logger.info("intent %s has no tools, answering directly", model_intent.value)
            return dict(list_planned_tool_calls=list())

        model_with_tools = self.model_llm.bind_tools(list_tools)

        list_invoke_args: tp.List[BaseMessage] = list()
        list_invoke_args.append(
            SystemMessage(content=build_planning_system_message(model_intent))
        )
        list_invoke_args.extend(build_langchain_messages(state.list_messages))
        list_invoke_args.append(HumanMessage(content=state.str_user_input))

        model_response = await model_with_tools.ainvoke(list_invoke_args)
        list_raw_tool_calls: tp.List[tp.Dict[str, tp.Any]] = list(model_response.tool_calls)

        if not list_raw_tool_calls:
            logger.info("no tool calls planned for intent %s", model_intent.value)
            return dict(
                list_planned_tool_calls=list(),
                str_direct_answer=str(model_response.content),
            )

        list_planned: tp.List[PlannedToolCall] = list(
            PlannedToolCall(
                str_name=str(dict_call.get("name")),
                dict_args=dict(dict_call.get("args", dict())),
                str_id=dict_call.get("id"),
            )
            for dict_call in list_raw_tool_calls
        )
        logger.info("planned %d tool call(s)", len(list_planned))
        return dict(list_planned_tool_calls=list_planned)

    async def _node_execute_tools(self, state: AgentState) -> tp.Dict[str, tp.Any]:
        """계획된 Tool을 실제로 실행하고 결과를 수집."""
        if not state.list_planned_tool_calls:
            return dict()

        model_intent: IntentType = state.model_intent or DEFAULT_INTENT
        list_tools: tp.List[BaseTool] = self._resolve_scenario_tools(model_intent)
        dict_tool_by_name: tp.Dict[str, BaseTool] = dict(
            (model_tool.name, model_tool) for model_tool in list_tools
        )

        list_execution_results: tp.List[ToolExecutionResult] = list()
        for model_call in state.list_planned_tool_calls:
            model_tool: tp.Optional[BaseTool] = dict_tool_by_name.get(model_call.str_name)
            if model_tool is None:
                logger.warning(
                    "planned tool '%s' is not available for intent %s",
                    model_call.str_name,
                    model_intent.value,
                )
                continue
            list_execution_results.append(self._invoke_single_tool(model_tool, model_call))

        model_report: ToolExecutionReport = ToolExecutionReport(
            list_results=list_execution_results,
            str_message=f"{len(list_execution_results)}개의 도구가 실행되었습니다.",
        )
        return dict(model_tool_report=model_report)

    def _invoke_single_tool(
        self, model_tool: BaseTool, model_call: PlannedToolCall
    ) -> ToolExecutionResult:
        """단일 도구를 실행하고 성공/실패를 결과 객체로 변환."""
        try:
            dict_output: tp.Dict[str, tp.Any] = model_tool.invoke(model_call.dict_args)
            return ToolExecutionResult(
                str_tool_name=model_call.str_name,
                dict_tool_input=model_call.dict_args,
                dict_tool_output=dict_output,
            )
        except NotImplementedError:
            logger.warning("tool '%s' is not implemented yet", model_call.str_name)
            return ToolExecutionResult(
                str_tool_name=model_call.str_name,
                dict_tool_input=model_call.dict_args,
                str_error="도구가 아직 구현되지 않았습니다.",
            )
        except (ValueError, KeyError, RuntimeError) as model_error:
            logger.exception("tool '%s' execution failed", model_call.str_name)
            return ToolExecutionResult(
                str_tool_name=model_call.str_name,
                dict_tool_input=model_call.dict_args,
                str_error=str(model_error),
            )

    async def _node_format_response(self, state: AgentState) -> tp.Dict[str, tp.Any]:
        """최종 결과를 자연어로 변환하고 이력에 저장."""
        model_intent: IntentType = state.model_intent or DEFAULT_INTENT
        str_evidence: str = self._build_evidence_text(state)

        str_prompt: str = build_response_synthesis_prompt(model_intent, str_evidence)
        model_response = await self.model_llm.ainvoke([HumanMessage(content=str_prompt)])
        str_answer: str = str(model_response.content)

        list_updated_messages: tp.List[ConversationMessage] = list(state.list_messages)
        list_updated_messages.append(
            ConversationMessage(model_role=MessageRole.USER, str_content=state.str_user_input)
        )
        list_updated_messages.append(
            ConversationMessage(model_role=MessageRole.ASSISTANT, str_content=str_answer)
        )

        return dict(str_final_answer=str_answer, list_messages=list_updated_messages)

    @staticmethod
    def _build_evidence_text(state: AgentState) -> str:
        """답변 생성 프롬프트에 넣을 근거 텍스트 구성."""
        if state.model_tool_report is not None:
            return state.model_tool_report.model_dump_json()
        if state.str_direct_answer is not None:
            return state.str_direct_answer
        return "도구 실행 결과 없음"

    async def run(
        self,
        str_input: str,
        list_history: tp.Optional[tp.List[ConversationMessage]] = None,
    ) -> AgentState:
        """에이전트 실행 진입점 (이력 포함)."""
        model_initial_state: AgentState = AgentState(
            str_user_input=str_input,
            list_messages=list(list_history) if list_history is not None else list(),
        )

        dict_final_state: tp.Dict[str, tp.Any] = await self.graph_workflow.ainvoke(
            model_initial_state
        )
        return AgentState(**dict(dict_final_state))
