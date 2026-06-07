"""LangGraph state machine for the arXiv research agent."""

from __future__ import annotations

import json
import sys
import time
from typing import Annotated, TypedDict

from anthropic import APIStatusError, RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from arxiv_agent.agent.prompts import get_system_prompt
from arxiv_agent.agent.retry import with_llm_retry
from arxiv_agent.agent.schemas import AgentAnswer
from arxiv_agent.agent.tools import ALL_TOOLS
from arxiv_agent.config import (
    ANTHROPIC_API_KEY,
    MAX_AGENT_ITERATIONS,
    PRIMARY_MODEL,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """The state passed through the graph.

    `messages` accumulates as the agent reasons. The `add_messages` reducer
    appends new messages to the list rather than replacing it.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    iterations: int


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------


def _build_llm():
    """Build the Anthropic LLM client bound to our tools."""
    llm = ChatAnthropic(
        model=PRIMARY_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=4096,
        temperature=0,
    )
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current message history.

    The LLM either:
      - Returns a final answer (no tool calls) → graph ends
      - Returns one or more tool calls → graph routes to tool node
    """
    llm = _build_llm()
    response = with_llm_retry(llm.invoke)(state["messages"])

    # Print a brief status to stderr so the user sees progress
    if response.tool_calls:
        names = ", ".join(tc["name"] for tc in response.tool_calls)
        print(
            f"[agent] iteration {state['iterations'] + 1}: calling tool(s) {names}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[agent] iteration {state['iterations'] + 1}: producing final answer",
            file=sys.stderr,
            flush=True,
        )

    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
    }


# ToolNode is a LangGraph helper that executes any tool calls in the latest
# AIMessage and appends the results as ToolMessages.
tool_node = ToolNode(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def should_continue(state: AgentState) -> str:
    """Decide whether to call tools or end.

    Continue (call tools) when:
      - The latest message is an AIMessage with tool_calls
      - We haven't hit the iteration limit

    End otherwise.
    """
    if state["iterations"] >= MAX_AGENT_ITERATIONS:
        print(
            f"[agent] hit max iterations ({MAX_AGENT_ITERATIONS}); forcing end",
            file=sys.stderr,
            flush=True,
        )
        return END

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Assemble the agent's state machine."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


def _extract_final_answer(messages: list[AnyMessage]) -> str:
    """Grab the last AI message's text content."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            # AIMessage.content can be a string or a list of content blocks
            if isinstance(msg.content, str):
                return msg.content
            # Block-style content; concatenate text blocks
            parts = []
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
    return ""


def _parse_agent_answer(raw_text: str, question: str, iterations: int) -> AgentAnswer:
    """Parse the LLM's final response into an AgentAnswer.

    Tries multiple strategies in order:
      1. Whole response is JSON
      2. JSON inside ```json ... ``` fence
      3. JSON inside generic ``` ... ``` fence
      4. First {...} object found in the response
    """
    candidates: list[str] = []

    # Strategy 1: whole response
    candidates.append(raw_text.strip())

    # Strategy 2 & 3: code fences
    if "```json" in raw_text:
        start = raw_text.find("```json") + len("```json")
        end = raw_text.find("```", start)
        if end > start:
            candidates.append(raw_text[start:end].strip())
    if "```" in raw_text:
        start = raw_text.find("```") + 3
        end = raw_text.find("```", start)
        if end > start:
            candidates.append(raw_text[start:end].strip())

    # Strategy 4: first balanced {...} block
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(raw_text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                continue
            answer = AgentAnswer.model_validate(data)
            answer.question = question
            answer.iterations_used = iterations
            return answer
        except (json.JSONDecodeError, ValueError):
            continue

    # Fallback: agent failed to produce structured output
    print(
        "[agent] WARNING: could not parse structured answer; using fallback",
        file=sys.stderr,
        flush=True,
    )
    return AgentAnswer(
        question=question,
        question_type="cannot_answer",
        answer=(
            "The agent failed to produce a structured answer. "
            "Raw response below:\n\n" + raw_text
        ),
        confidence="low",
        confidence_reason="Output did not match the required schema.",
        iterations_used=iterations,
    )


# Cache the compiled graph since rebuilding it is expensive
_compiled_graph = None


def get_graph():
    """Return the compiled graph, building it on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(question: str) -> AgentAnswer:
    """Run the agent end-to-end on a single question.

    Args:
        question: A natural-language research question.

    Returns:
        A validated AgentAnswer.
    """
    from arxiv_agent.agent.tracing import flush, get_callback_handler

    graph = get_graph()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=get_system_prompt()),
            HumanMessage(content=question),
        ],
        "iterations": 0,
    }

    config: dict = {"recursion_limit": MAX_AGENT_ITERATIONS * 3}

    handler = get_callback_handler()
    if handler is not None:
        config["callbacks"] = [handler]
        # Tag traces so they're easy to find in the Langfuse UI
        config["metadata"] = {
            "agent": "arxiv-research-agent",
            "version": "v1",
        }
        config["tags"] = ["arxiv-agent", "v1"]
        config["run_name"] = f"arxiv-agent: {question[:60]}"

    max_retries = 5
    base_wait = 15.0
    last_exc = None
    try:
        for attempt in range(max_retries):
            try:
                final_state = graph.invoke(initial_state, config=config)
                break
            except (RateLimitError, APIStatusError) as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code != 429 and not isinstance(exc, RateLimitError):
                    raise
                last_exc = exc
                wait = min(base_wait * (2**attempt), 120.0)
                print(
                    f"[agent] Rate limited (attempt {attempt + 1}/{max_retries}); "
                    f"waiting {wait:.0f}s before retry",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        else:
            # Exhausted all retries
            raise RuntimeError(
                f"Exhausted {max_retries} retries on rate-limit errors"
            ) from last_exc
    finally:
        flush()  # ensure traces get sent even if invoke raises

    raw_answer = _extract_final_answer(final_state["messages"])
    return _parse_agent_answer(
        raw_answer,
        question=question,
        iterations=final_state["iterations"],
    )
