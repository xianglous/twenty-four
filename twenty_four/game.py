import functools
import itertools
from typing import Annotated, Callable, Collection, Generator, Literal, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
)


def _swap_and_pop(arr: list[float], idx: int):
    """Remove number by swapping and popping."""
    if idx == len(arr) - 1:
        arr.pop()
    else:
        arr[-1], arr[idx] = arr[idx], arr[-1]
        arr.pop()


def _convert_int_to_float(nums: Collection[Union[int, float]], info: ValidationInfo) -> list[float]:
    """Validate the initial state of the game. Convert ints to floats."""
    if len(nums) != info.data["num_vals"]:
        raise ValueError(f"There must be {info.data['num_vals']} numbers")
    for n in nums:
        if n < 1 or n > info.data["max_val"]:
            raise ValueError(f"Invalid number: {n}")
    return [float(n) for n in nums]


class Game(BaseModel):
    """
    Game class represents a 24 game.
    It contains the interface to perform operations and update states.
    """

    model_config = ConfigDict(validate_assignment=True)

    target: float = 24.0
    max_val: int = 13
    num_vals: int = 4
    epsilon: float = 1e-9

    init_state: Annotated[list[float], BeforeValidator(_convert_int_to_float)]
    cur_state: Annotated[
        list[float],
        Field(
            default_factory=lambda data: data["init_state"].copy(),
            min_length=1,
            max_length=num_vals,
        ),
    ]
    op_history: Annotated[
        list[tuple[int, int, str, float, list[float]]],
        Field(default=[]),
    ]

    def _update_state(self, idx1: int, idx2: int, op: str, result: float) -> bool:
        """Perform the op and update game state."""
        prev_state = self.cur_state.copy()
        max_idx = max(idx1, idx2)
        min_idx = min(idx1, idx2)
        _swap_and_pop(self.cur_state, max_idx)
        _swap_and_pop(self.cur_state, min_idx)
        self.cur_state.append(result)
        # Record the op in history
        self.op_history.append((idx1, idx2, op, result, prev_state))
        return True

    def _is_valid_idx(self, idx1: int, idx2: int) -> bool:
        """Validate that idx are different and within bounds"""
        if idx1 == idx2:
            return False
        if idx1 < 0 or idx1 >= len(self.cur_state):
            return False
        if idx2 < 0 or idx2 >= len(self.cur_state):
            return False
        return True

    def operation(op: Literal["+", "-", "*", "/"]):
        """Decorator to wrap the operation function with id check and state update."""

        def decorator(func: Callable[[int, int], float]) -> Callable[[int, int], bool]:
            @functools.wraps(func)
            def wrapper(self: "Game", idx1: int, idx2: int) -> bool:
                if not self._is_valid_idx(idx1, idx2):
                    return False
                result = func(self, idx1, idx2)
                return self._update_state(idx1, idx2, op, result)

            return wrapper

        return decorator

    @operation("+")
    def add(self, idx1: int, idx2: int) -> float:
        """Add two numbers at given idx."""
        return self.cur_state[idx1] + self.cur_state[idx2]

    @operation("-")
    def subtract(self, idx1: int, idx2: int) -> float:
        """Subtract number at idx2 from number at idx1."""
        return self.cur_state[idx1] - self.cur_state[idx2]

    @operation("*")
    def multiply(self, idx1: int, idx2: int) -> float:
        """Multiply two numbers at given idx."""
        return self.cur_state[idx1] * self.cur_state[idx2]

    @operation("/")
    def divide(self, idx1: int, idx2: int) -> float:
        """Divide number at idx1 by number at idx2."""
        return self.cur_state[idx1] / self.cur_state[idx2]

    def undo(self) -> bool:
        """Undo the last op. Returns True if successful."""
        if not self.op_history:
            return False
        _, _, _, _, prev_state = self.op_history.pop()
        self.cur_state = prev_state
        return True

    def get_valid_ops(self) -> Generator[tuple[int, int, str], None, None]:
        """Get all valid operations for the current state"""
        for i, j in itertools.combinations(range(len(self.cur_state)), 2):
            for op in ["+", "-", "*", "/"]:
                if op == "-":
                    # Subtraction no need for negative result
                    if self.cur_state[i] < self.cur_state[j]:
                        yield (j, i, op)
                    else:
                        yield (i, j, op)
                elif op in ["+", "*"]:
                    # Addition and multiplication are commutative
                    yield (i, j, op)
                else:
                    # Divide by zero is not allowed
                    if abs(self.cur_state[j]) > self.epsilon:
                        yield (i, j, op)
                    if abs(self.cur_state[i]) > self.epsilon:
                        yield (j, i, op)

    def get_op_func(self, op: Literal["+", "-", "*", "/"]) -> Callable[[int, int], bool]:
        """Get the op function for the given op."""
        if op == "+":
            return self.add
        elif op == "-":
            return self.subtract
        elif op == "*":
            return self.multiply
        elif op == "/":
            return self.divide

    def is_solved(self) -> bool:
        """Check if the game is solved (one number remaining that equals target)"""
        if len(self.cur_state) == 1:
            return abs(self.cur_state[0] - self.target) < self.epsilon
        return False

    def get_cur_state(self) -> list[float]:
        """Get the current available numbers"""
        return self.cur_state.copy()

    def get_solution_expr(self) -> str:
        """Reconstruct the solution expression from op history"""
        if not self.op_history:
            return ""
        exprs = [str(int(num)) for num in self.init_state]  # Expressions at each idx
        expr_ops = [""] * len(self.init_state)  # Operations at each idx
        for idx1, idx2, op, _, _ in self.op_history:
            # Replicate the same order of removal as in _update_state
            max_idx = max(idx1, idx2)
            min_idx = min(idx1, idx2)
            expr1 = exprs[idx1]
            expr2 = exprs[idx2]
            _swap_and_pop(exprs, max_idx)
            _swap_and_pop(exprs, min_idx)
            op1 = expr_ops[idx1]
            op2 = expr_ops[idx2]
            _swap_and_pop(expr_ops, max_idx)
            _swap_and_pop(expr_ops, min_idx)
            # Handle parentheses
            if op == "*":
                if op1 in ["+", "-"]:
                    expr1 = f"({expr1})"
                if op2 in ["+", "-", "/"]:
                    expr2 = f"({expr2})"
            elif op == "/":
                if op1 in ["+", "-"]:
                    expr1 = f"({expr1})"
                if op2 != "":
                    expr2 = f"({expr2})"
            # Add the new expression and operation
            exprs.append(f"{expr1} {op} {expr2}")
            expr_ops.append(op)
        return " ".join(exprs)
