import itertools
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from game import Game, OP_FUNCS


class Solver(ABC):
    """
    Solver is the base class for all solvers.
    """
    @abstractmethod
    def solve(self, game: Game) -> Optional[str]:
        """Solve a game using the solver."""


class BruteForceSolver(Solver):
    """
    Brute Force Solver tries all possible ops recursively.
    """
    def _solve_dfs(self, game: Game) -> Optional[str]:
        """Recursively try all possible ops"""
        # Base case: check if solved
        if game.is_solved():
            return game.get_solution_expr()
        # If we have only one number left but it's not 24, this path failed
        cur_state = game.get_cur_state()
        if len(cur_state) == 1:
            return None
        # Try all possible pairs of numbers and all ops
        for i, j in itertools.permutations(range(len(cur_state)), 2):
            for _, op_func in OP_FUNCS.items():
                # Try the op
                if op_func(game, i, j):
                    # Recursively solve with the new state
                    result = self._solve_dfs(game)
                    if result:
                        return result
                    # Backtrack
                    game.undo()
        return None
    
    def solve(self, game: Game) -> Optional[str]:
        """
        Solve the 24 game for given numbers using the Game class ops.
        Returns the solution expression or None if not solvable.
        """
        solution = self._solve_dfs(game)
        return solution


class SmartSolver(Solver):
    """
    Smart Solver uses example games to build a memo mapping of
    ordered state to the next operation to perform.
    """
    ops_memo = {}

    def __init__(self, num_games: int = 1000):
        """Initialize SmartSolver and generate training games."""
        for i, state in enumerate(itertools.combinations_with_replacement(range(1, 14), 4)):
            game = Game(state)
            self._solve_dfs_with_memo(game)
            if i % 100 == 0:
                print(f"Solved {i + 1} games")
            if i == num_games:
                break

    def _ordered_to_unordered_idx(self, state: List[float], ordered_idx1: int, ordered_idx2: int) -> Tuple[int, int]:
        """Convert ordered state idx to unordered state idx."""
        ordered_state_with_idx = tuple(sorted((float(v), i) for i, v in enumerate(state)))
        i1, i2 = 0, 0
        for i, (_, idx) in enumerate(ordered_state_with_idx):
            if i == ordered_idx1:
                i1 = idx
            if i == ordered_idx2:
                i2 = idx
        return i1, i2

    def _unordered_to_ordered_idx(self, state: List[float], unordered_idx1: int, unordered_idx2: int) -> Tuple[int, int]:
        """Convert unordered state idx to ordered state idx."""
        ordered_state_with_idx = tuple(sorted((float(v), i) for i, v in enumerate(state)))
        i1, i2 = 0, 0
        for i, (_, idx) in enumerate(ordered_state_with_idx):
            if idx == unordered_idx1:
                i1 = i
            if idx == unordered_idx2:
                i2 = i
        return i1, i2
    
    def _complete_solution_expr(self, game: Game) -> Optional[str]:
        """Complete the solution expression from the current state using the memo."""
        next_op = self._get_next_op(game.get_cur_state())
        while next_op is not None:
            idx1, idx2, op = next_op
            idx1, idx2 = self._ordered_to_unordered_idx(game.get_cur_state(), idx1, idx2)
            op_func = OP_FUNCS[op]
            op_func(game, idx1, idx2)
            next_op = self._get_next_op(game.get_cur_state())
        return game.get_solution_expr()

    def _save_next_op(self, state: List[float], idx1: int, idx2: int, op: str):
        """Save the next op to the memo."""
        i1, i2 = self._unordered_to_ordered_idx(state, idx1, idx2)
        self.ops_memo[tuple(sorted(state))] = (i1, i2, op)

    def _get_next_op(self, state: List[float]) -> Optional[Tuple[int, int, str]]:
        """Get the next op from the memo."""
        ordered_state = tuple(sorted(state))
        if ordered_state in self.ops_memo:
            return self.ops_memo[ordered_state]
        return None
    
    def _mark_unsolvable_state(self, state: List[float]):
        """Mark the state as unsolvable."""
        ordered_state = tuple(sorted(state))
        self.ops_memo[ordered_state] = (-1, -1, "X")

    def _is_unsolvable_state(self, state: List[float]) -> bool:
        """Check if the state is unsolvable."""
        ordered_state = tuple(sorted(state))
        if ordered_state in self.ops_memo and self.ops_memo[ordered_state][0] == -1:
            return True
        return False

    def _solve_dfs_with_memo(self, game: Game) -> Optional[str]:
        """Solve a game using memoization."""
        cur_state = game.get_cur_state().copy()
        # Check if the state is unsolvable
        if self._is_unsolvable_state(cur_state):
            return None
        # Check if the state is in memo
        next_op = self._get_next_op(cur_state)
        if next_op is not None:
            return self._complete_solution_expr(game)
        # Check if the state is solved
        if game.is_solved():
            return self._complete_solution_expr(game)
        # Check if the state has failed
        if len(cur_state) == 1:
            return None
        # Try all possible ops
        for i, j in itertools.permutations(range(len(cur_state)), 2):
            for op, op_func in OP_FUNCS.items():
                # Try the op
                if op_func(game, i, j):
                    # Recursively solve with the new state
                    result = self._solve_dfs_with_memo(game)
                    if result is not None:
                        self._save_next_op(cur_state, i, j, op)
                        return result
                    # Backtrack
                    self._mark_unsolvable_state(game.get_cur_state())
                    game.undo()
        return None
    
    def solve(self, game: Game) -> Optional[str]:
        """Solve a game using the trained model."""
        return self._solve_dfs_with_memo(game)


class LLMSolver(Solver):
    """
    LLMSolver uses create_react_agent from LangGraph to solve the 24 game.
    """
    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        # Initialize LLM
        model = ChatOpenAI(model=model, **kwargs)
        
        # Define tools that will be bound to the game instance during solve
        @tool
        def perform_op(idx1: int, idx2: int, op: Literal["+", "-", "*", "/"]) -> str:
            """Perform an operation on two numbers at the given indices.
            
            Args:
                idx1: Index of first number
                idx2: Index of second number
                op: Operation to perform (+, -, *, /)
            """
            game: Game = perform_op._game
            if not game.validate_idx(idx1, idx2):
                return f"Invalid indices: {idx1}, {idx2}. Current state has {len(game.get_cur_state())} numbers."
            op_func = OP_FUNCS[op]
            if op_func(game, idx1, idx2):
                return f"Performed {op} on {idx1} and {idx2}. Current state: {game.get_cur_state()}"
            else:
                return f"Failed to perform {op} on {idx1} and {idx2}"

        @tool
        def undo_op() -> str:
            """Undo the last operation."""
            game: Game = undo_op._game
            game.undo()
            return f"Undone the last operation. Current state: {game.get_cur_state()}"
        
        @tool
        def is_solved() -> str:
            """Check if the current game state is solved (equals 24)."""
            game: Game = is_solved._game
            current_state = game.get_cur_state()
            if game.is_solved():
                return f"SOLVED! Current state: {current_state}"
            else:
                return f"Not solved yet. Current state: {current_state}"
        
        # Store tools for later use
        self.tools = [perform_op, undo_op, is_solved]
        
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
* Think about which operations might get you factors of 24 like 2, 3, 4, 6, 8, 12
* You can undo the last operation if you think you made a mistake
* Remember that order matters for subtraction and division
* Only use is_solved to verify when there is only one number left

### Tools
* perform_op(idx1, idx2, op): Perform an operation on two numbers at the given indices, op is one of +, -, *, /
* undo_op(): Undo the last operation
* is_solved(): Check if current state equals 24

### Instructions
* Think step by step and use the tools to solve the puzzle!"""
        
        self.agent = create_react_agent(
            model,
            tools=self.tools,
            prompt=system_prompt,
        )
    
    def solve(self, game: Game) -> Optional[str]:
        """Solve a game using the ReAct agent."""
        try:
            # Bind the game instance to the tools
            for tool in self.tools:
                tool._game = game
            
            # Create initial message with current state
            initial_state = game.get_cur_state()
            
            # Run the agent
            message = f"Current state: {initial_state}"
            _ = self.agent.invoke({"messages": [{"role": "user", "content": message}]})
            
            # Check if solved
            if game.is_solved():
                return game.get_solution_expr()
            else:
                return None
                
        except Exception as e:
            return None
        finally:
            # Clean up tool bindings
            for tool in self.tools:
                if hasattr(tool, '_game'):
                    delattr(tool, '_game')
