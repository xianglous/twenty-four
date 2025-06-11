from typing import Annotated, Literal, Optional

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic

from twenty_four.basic_solver import Solver
from twenty_four.game import Game


class AgentState(AgentStatePydantic):
    """Pydantic agent state containing the game instance."""

    game: Game


class AgentSolver(Solver):
    """
    AgentSolver uses LangGraph ReAct agent to solve the 24 game.
    """

    def __init__(self, model: str = "o4-mini", **kwargs):
        # Initialize LLM
        model = ChatOpenAI(model=model, **kwargs)

        # Tools definition
        @tool
        def perform_op(
            idx1: int,
            idx2: int,
            op: Literal["+", "-", "*", "/"],
            game: Annotated[Game, InjectedState("game")],
        ) -> str:
            """Perform an operation on two numbers at the given indices.

            Args:
                idx1: Index of first number
                idx2: Index of second number
                op: Operation to perform (+, -, *, /)
            """
            op_func = game.get_op_func(op)
            if op_func(idx1, idx2):
                return f"Performed {op} on {idx1} and {idx2}. Current state: {game.get_cur_state()}"
            else:
                return f"Failed to perform {op} on {idx1} and {idx2}"

        @tool
        def undo_op(game: Annotated[Game, InjectedState("game")]) -> str:
            """Undo the last operation."""
            game.undo()
            return f"Undone the last operation. Current state: {game.get_cur_state()}"

        @tool
        def direct_solve(game: Annotated[Game, InjectedState("game")]) -> str:
            """Directly solve the game when there are only two numbers in the current state."""
            if len(game.get_cur_state()) > 2:
                return f"More than 2 numbers left. Current state: {game.get_cur_state()}"
            for op in ["+", "-", "*", "/"]:
                op_func = game.get_op_func(op)
                for i, j in [(0, 1), (1, 0)]:
                    if op_func(i, j):
                        if game.is_solved():
                            return f"SOLVED! Current state: {game.get_cur_state()}"
                    game.undo()
            return f"Not solved yet. Current state: {game.get_cur_state()}"

        # Store tools for later use
        tools = [perform_op, undo_op, direct_solve]

        # Create the ReAct agent
        system_prompt = """
### Identity
You are an expert at solving the 24 game.
Use the four basic arithmetic operations (+, -, *, /) to make exactly 24 from the given numbers.

### Rules
* You start with 4 numbers
* You can add, subtract, multiply, or divide any two numbers
* After each operation, the two numbers are removed and their result is added to the end of the list
* Continue until you have exactly one number that equals 24
* You have at most 3 operations to solve it
* You can only use each number once

### Strategy
* Think about which operations might get you factors of 24 like 2, 3, 4, 6, 8, 12,
  then try to create their counterparts like 12, 8, 6, 4, 3, 2 using the other numbers
* You can undo the last operation if you think you made a mistake
* Remember that order matters for subtraction and division
* Only use direct_solve when there are only two numbers left

### Tools
* perform_op(idx1, idx2, op): Perform an operation on two numbers at the given indices, op is one of +, -, *, /
* undo_op(): Undo the last operation
* direct_solve(): Try to solve the game when there are only two numbers left

### Instructions
* Think step by step and use the tools to solve the puzzle!"""

        # Agent creation
        self.agent = create_react_agent(
            model,
            tools=tools,
            prompt=system_prompt,
            state_schema=AgentState,
        ).with_config(recursion_limit=50)

    def solve(self, game: Game) -> Optional[str]:
        """Solve a game using the ReAct agent."""
        # Create initial message with current state
        initial_state = game.get_cur_state()
        message = f"Current state: {initial_state}"
        _ = self.agent.invoke(
            {
                "messages": [{"role": "user", "content": message}],
                "game": game,
            },
        )

        # Check solution
        if game.is_solved():
            return game.get_solution_expr()
        else:
            return None
