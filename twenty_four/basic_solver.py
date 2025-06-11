import itertools
from abc import ABC, abstractmethod
from typing import Optional

from twenty_four.game import Game


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
        # If we one number left but not 24, this path failed
        cur_state = game.get_cur_state()
        if len(cur_state) == 1:
            return None
        # DFS
        for i, j, op in game.get_valid_ops():
            op_func = game.get_op_func(op)
            if op_func(i, j):
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

    States are sorted to reduce search space.
    The ops_memo maps ordered state to the next operation (idx1, idx2, op) that leads to a solution.
    """

    ops_memo = {}

    def __init__(self, num_games: int = 1000):
        """Initialize SmartSolver and generate training games."""
        print("======Training Started======")
        for i, state in enumerate(itertools.combinations_with_replacement(range(1, 14), 4)):
            game = Game(init_state=state)
            self._solve_dfs_with_memo(game)
            if i % 100 == 0:
                print(f"Solved {i + 1} games")
            if i == num_games:
                break
        print("========Training Done========")

    def _ordered_to_unordered_idx(self, state: list[float], ordered_idx1: int, ordered_idx2: int) -> tuple[int, int]:
        """Convert ordered state idx to unordered state idx."""
        ordered_state_with_idx = tuple(sorted((float(v), i) for i, v in enumerate(state)))
        i1, i2 = 0, 0
        for i, (_, idx) in enumerate(ordered_state_with_idx):
            if i == ordered_idx1:
                i1 = idx
            if i == ordered_idx2:
                i2 = idx
        return i1, i2

    def _unordered_to_ordered_idx(
        self, state: list[float], unordered_idx1: int, unordered_idx2: int
    ) -> tuple[int, int]:
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
            op_func = game.get_op_func(op)
            op_func(idx1, idx2)
            next_op = self._get_next_op(game.get_cur_state())
        return game.get_solution_expr()

    def _save_next_op(self, state: list[float], idx1: int, idx2: int, op: str):
        """Save the next op to the memo."""
        i1, i2 = self._unordered_to_ordered_idx(state, idx1, idx2)
        self.ops_memo[tuple(sorted(state))] = (i1, i2, op)

    def _get_next_op(self, state: list[float]) -> Optional[tuple[int, int, str]]:
        """Get the next op from the memo."""
        ordered_state = tuple(sorted(state))
        if ordered_state in self.ops_memo:
            return self.ops_memo[ordered_state]
        return None

    def _mark_unsolvable_state(self, state: list[float]):
        """Mark the state as unsolvable."""
        ordered_state = tuple(sorted(state))
        self.ops_memo[ordered_state] = (-1, -1, "X")

    def _is_unsolvable_state(self, state: list[float]) -> bool:
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
        # DFS
        for i, j, op in game.get_valid_ops():
            op_func = game.get_op_func(op)
            if op_func(i, j):
                result = self._solve_dfs_with_memo(game)
                if result is not None:
                    # Update memo
                    self._save_next_op(cur_state, i, j, op)
                    return result
                # Backtrack
                self._mark_unsolvable_state(game.get_cur_state())
                game.undo()
        return None

    def solve(self, game: Game) -> Optional[str]:
        """Solve a game using the trained model."""
        return self._solve_dfs_with_memo(game)
