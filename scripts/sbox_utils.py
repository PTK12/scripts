from functools import cache
from collections.abc import Sequence
from typing import Iterable, Generator

__all__ = ["UTILS", "pp"]

def pp(x: Sequence):
    _pp(x)

def _pp(x: Sequence, depth: int = 1):
    if isinstance(x, Sequence) and len(x) > 0 and isinstance(x[0], Sequence):
        print(end="[")
        _pp(x[0], depth + 1)

        for i in x[1:]:
            print()
            print(end = depth * " ")
            _pp(i, depth + 1)

        print(end="]")

    else: print(str(x), end="")

    if depth == 1: print()

class UTILS:
    def __init__(self, G) -> None:
        """Accepts a Polynomial Ring."""
        self.F = G.base_ring()
        self.FF = self.F._cache
        self.G = G
        self.S = self.F.base_ring().cardinality()
        self.SIZE = self.F.cardinality()
        self.ordered = [self.F.from_integer(i) for i in range(self.SIZE)]

    def ddt(self, sbox: Iterable[int]) -> tuple[tuple[int]]:
        """Generates a DDT for the given SBOX."""
        sbox2 = {self.FF.fetch_int(i):self.FF.fetch_int(j) for i, j in enumerate(sbox)}
        table = []
        for delta_in in self.ordered:
            row = [0] * self.SIZE
            for element in self.ordered:
                delta_out = (sbox2[element + delta_in] - sbox2[element]).to_integer()
                row[delta_out] += 1
            table.append(tuple(row))
        return tuple(table)

    @cache
    def _R(self, i: int, j: int, ddt: tuple[tuple[int]]): # return possible delta outs
        out = []
        delta_in = (self.FF.fetch_int(i) - self.FF.fetch_int(j)).to_integer()
        for delta_out in range(self.SIZE):
            if ddt[delta_in][delta_out] > 0:
                out.append(delta_out)
        return out

    def _zero(self, ddt1: tuple[tuple[int]], ddt2: tuple[tuple[int]]) -> bool: # check if matching 0s and non 0s
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if (ddt1[i][j] == 0) != (ddt2[i][j] == 0):
                    return False
        return True

    @cache
    def _fast(self, x: int, y: int) -> int: # int to int - fast add
        return (self.FF.fetch_int(x) + self.FF.fetch_int(y)).to_integer()

    def _guess(self, vec: list[int], i: int, ddt: tuple[tuple[int]]) -> Generator[list[int], None, None]:
        L = []
        if i < self.SIZE:
            L = set(range(self.SIZE))
            for j in range(i):
                L &= set(self._fast(vec[j], x) for x in self._R(i, j, ddt))
                if len(L) == 0: break
            else:
                L = list(L)
                for x in L:
                    vec[i] = x
                    yield from self._guess(vec, i + 1, ddt)

        elif self._zero(self.ddt(vec), ddt):
            yield vec

    def recover_ddt(self, ddt: tuple[tuple[int]], redundant_check: bool = False, verbose: bool = False):
        """Reconstructs SBOXs given the DDT via The Improved Guess-and-Determine Algorithm (Dunkelman & Huang)."""
        vec = [0] * self.SIZE
        values = []

        for possible_sbox in self._guess(vec, 1, ddt):
            if verbose: print(possible_sbox)
            if len(values) > 1 and possible_sbox[1] > values[-2][1]:
                if redundant_check: values.append(possible_sbox.copy())
                break
            values.append(possible_sbox.copy())

        return values

    def get_poly(self, sbox: list[int]):
        """Gets the polynomial of the SBOX via Lagrange interpolation."""
        return self.G.lagrange_polynomial((self.FF.fetch_int(x), self.FF.fetch_int(y)) for x, y in enumerate(sbox))

    def get_box(self, f) -> list[int]:
        """Evaluates the polynomial to generate the SBOX."""
        return [f(i).to_integer() for i in self.ordered]

    def linear_map(self, sbox: list[int]) -> list[list[int]]:
        """Produces a table on how close every line equation resembles the SBOX."""
        sbox2 = [self.ordered[i] for i in sbox]
        grid = [[0] * self.SIZE for _ in range(self.SIZE)]

        for i, grad in enumerate(self.ordered):
            for j, element in enumerate(self.ordered):
                offset = (sbox2[j] - grad * element).to_integer()
                grid[i][offset] += 1

        return grid

    @cache
    def _mask_sum(self, x, y):
        return sum(i * j for i, j in zip(x.list(), y.list()))

    def lat(self, sbox: list[int]) -> list[list[list[int]]]:
        """Produces a (generalised) linear approximation table for the given SBOX."""
        sbox2 = [self.ordered[i] for i in sbox]
        table = [[[0] * self.S for _ in range(self.SIZE)] for _ in range(self.SIZE)]
        for i, inp_mask in enumerate(self.ordered):
            for j, out_mask in enumerate(self.ordered):
                for e, element in enumerate(self.ordered):
                    res = self._mask_sum(inp_mask, element) + self._mask_sum(out_mask, sbox2[e])
                    table[i][j][res] += 1
        return table
