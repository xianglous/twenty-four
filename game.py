from typing import Callable, Collection, Dict, List, Literal



def _remove(arr: List[float], idx: int):
    """Remove number by swapping and popping."""
    if idx == len(arr) - 1:
        arr.pop()
    else:
        arr[-1], arr[idx] = arr[idx], arr[-1]
        arr.pop()


class Game:
    """
    Game class represents a 24 game.
    It contains the interface to perform operations and undo them.
    """
    target = 24.0
    epsilon = 1e-9

    def __init__(self, nums: Collection[int]):
        if len(nums) != 4:
            raise ValueError("There must be 4 numbers")
        for n in nums:
            if n < 1 or n > 13:
                raise ValueError(f"Invalid number: {n}")
        self.init_state = sorted(map(float, nums))
        self.cur_state = self.init_state.copy()
        self.op_history = []
    
    def _perform_op(self, idx1: int, idx2: int, op: str, result: float) -> bool:
        """Perform the op and update game state."""
        prev_state = self.cur_state.copy()
        max_idx = max(idx1, idx2)
        min_idx = min(idx1, idx2)
        _remove(self.cur_state, max_idx)
        _remove(self.cur_state, min_idx)
        self.cur_state.append(result)
        # Record the op in history
        self.op_history.append((idx1, idx2, op, result, prev_state))
        return True
    
    def validate_idx(self, idx1: int, idx2: int) -> bool:
        """Validate that idx are different and within bounds"""
        if idx1 == idx2:
            return False
        if idx1 < 0 or idx1 >= len(self.cur_state):
            return False
        if idx2 < 0 or idx2 >= len(self.cur_state):
            return False
        return True
    
    def add(self, idx1: int, idx2: int) -> bool:
        """Add two numbers at given idx. Returns True if successful."""
        if not self.validate_idx(idx1, idx2):
            return False
        val1 = self.cur_state[idx1]
        val2 = self.cur_state[idx2]
        result = val1 + val2
        return self._perform_op(idx1, idx2, "+", result)
    
    def subtract(self, idx1: int, idx2: int) -> bool:
        """Subtract number at idx2 from number at idx1. Returns True if successful."""
        if not self.validate_idx(idx1, idx2):
            return False
        val1 = self.cur_state[idx1]
        val2 = self.cur_state[idx2]
        result = val1 - val2
        return self._perform_op(idx1, idx2, "-", result)
    
    def multiply(self, idx1: int, idx2: int) -> bool:
        """Multiply two numbers at given idx. Returns True if successful."""
        if not self.validate_idx(idx1, idx2):
            return False
        val1 = self.cur_state[idx1]
        val2 = self.cur_state[idx2]
        result = val1 * val2
        return self._perform_op(idx1, idx2, "*", result)
    
    def divide(self, idx1: int, idx2: int) -> bool:
        """Divide number at idx1 by number at idx2. Returns True if successful."""
        if not self.validate_idx(idx1, idx2):
            return False
        val1 = self.cur_state[idx1]
        val2 = self.cur_state[idx2]
        if abs(val2) < self.epsilon:
            return False  # Division by zero
        result = val1 / val2
        return self._perform_op(idx1, idx2, "/", result)
    
    def undo(self) -> bool:
        """Undo the last op. Returns True if successful."""
        if not self.op_history:
            return False
        _, _, _, _, prev_state = self.op_history.pop()
        self.cur_state = prev_state
        return True
    
    def is_solved(self) -> bool:
        """Check if the game is solved (one number remaining that equals target)"""
        if len(self.cur_state) == 1:
            return abs(self.cur_state[0] - self.target) < self.epsilon
        return False
    
    def get_cur_state(self) -> List[float]:
        """Get the current available numbers"""
        return self.cur_state.copy()
    
    def get_solution_expr(self) -> str:
        """Reconstruct the solution expression from op history"""
        if not self.op_history:
            return ""
        exprs = [str(int(num)) for num in self.init_state]  # Expressions at each idx
        expr_ops = [''] * len(self.init_state)  # Operations at each idx
        for idx1, idx2, op, _, _ in self.op_history:
            # Replicate the same order of removal as in _perform_op
            max_idx = max(idx1, idx2)
            min_idx = min(idx1, idx2)
            expr1 = exprs[idx1]
            expr2 = exprs[idx2]
            _remove(exprs, max_idx)
            _remove(exprs, min_idx)
            op1 = expr_ops[idx1]
            op2 = expr_ops[idx2]
            _remove(expr_ops, max_idx)
            _remove(expr_ops, min_idx)
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


OP_FUNCS: Dict[Literal["+", "-", "*", "/"], Callable[[Game, int, int], bool]] = {
    "+": lambda game, i1, i2: game.add(i1, i2),
    "-": lambda game, i1, i2: game.subtract(i1, i2),
    "*": lambda game, i1, i2: game.multiply(i1, i2),
    "/": lambda game, i1, i2: game.divide(i1, i2)
}
