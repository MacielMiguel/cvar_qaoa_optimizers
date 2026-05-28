"""
Classical solvers for quantum optimization loops.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize as opt
from numpy.typing import NDArray

try:
    import pdfo as _pdfo
    _PDFO_AVAILABLE = True
except ImportError:
    _pdfo = None  # type: ignore[assignment]
    _PDFO_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClassicalOptimizer(ABC):
    """Base class for classical optimizers."""

    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        self.maxiter = maxiter
        self.tol = tol

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Optimize the objective function.

        Args:
            objective_function: Function to minimize.
            initial_point: Starting point for optimization array of shape (N,).
            bounds: Optional variable bounds [(min, max), ...].

        Returns:
            Dict containing 'x', 'fun', 'nfev', 'success', and 'message'.
        """
        pass


class ScipyOptimizer(ClassicalOptimizer):
    """Generic Scipy Optimizer integrating DRY principles."""
    
    def __init__(
        self,
        method: str,
        maxiter: int = 100,
        tol: float = 1e-2,
        supports_bounds: bool = False
    ):
        super().__init__(maxiter=maxiter, tol=tol)
        self.method = method
        self.supports_bounds = supports_bounds

    def optimize(
        self,
        objective_function: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Execute `scipy.optimize.minimize` gracefully."""
        
        if bounds is not None and not self.supports_bounds:
            logger.warning(
                f"Optimizer {self.method} does not support bounds. They will be ignored."
            )
            bounds = None

        logger.info(f"Starting {self.method} classical optimization...")
        result = opt.minimize(
            objective_function,
            initial_point,
            method=self.method,
            tol=self.tol,
            bounds=bounds,
            options={'maxiter': self.maxiter}
        )
        logger.debug(f"{self.method} optimization finished in {getattr(result, 'nfev', 'Unknown')} evaluations.")

        return {
            'x': result.x,
            'fun': result.fun,
            'nfev': getattr(result, 'nfev', None),
            'success': result.success,
            'message': result.message
        }


class COBYLA(ScipyOptimizer):
    """
    COBYLA optimizer using scipy.optimize.
    (Recommended for standard QAOA pipelines).
    """
    
    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        super().__init__(method="COBYLA", maxiter=maxiter, tol=tol, supports_bounds=False)

class COBYQA(ScipyOptimizer):
    """
    COBYQA optimizer using scipy.optimize.
    """
    
    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        super().__init__(method="COBYQA", maxiter=maxiter, tol=tol, supports_bounds=False)


class ConjugateGradient(ScipyOptimizer):
    """Conjugate Gradient optimizer (scipy CG method)."""

    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        super().__init__(method="CG", maxiter=maxiter, tol=tol, supports_bounds=False)


class BFGS(ScipyOptimizer):
    """BFGS Quasi-Newton optimizer."""
    
    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        super().__init__(method="BFGS", maxiter=maxiter, tol=tol, supports_bounds=False)


class Powell(ScipyOptimizer):
    """Powell derivative-free optimizer."""
    
    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        super().__init__(method="Powell", maxiter=maxiter, tol=tol, supports_bounds=True)


class SimulatedAnnealing(ClassicalOptimizer):
    """Simulated Annealing optimizer via scipy dual_annealing."""
    
    def __init__(self, maxiter: int = 100):
        # SA primarily relies on internal temperature drops, generic tol is bypassed
        super().__init__(maxiter=maxiter, tol=0.0)
        
    def optimize(
        self,
        objective_function: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        bounds: Optional[Sequence[Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """Execute `scipy.optimize.dual_annealing`."""
        
        if bounds is None:
            logger.info("dual_annealing requires bounds. Defaulting to [-pi, pi] per parameter.")
            bounds = [(-np.pi, np.pi)] * len(initial_point)
            
        logger.info("Starting Simulated Annealing (dual_annealing) optimization...")
        result = opt.dual_annealing(
            objective_function,
            bounds=bounds,
            x0=initial_point,
            maxiter=self.maxiter
        )
        logger.debug(f"Simulated Annealing finished in {getattr(result, 'nfev', 'Unknown')} evals.")
        
        return {
            'x': result.x,
            'fun': result.fun,
            'nfev': getattr(result, 'nfev', None),
            'success': getattr(result, 'success', True),
            'message': getattr(result, 'message', '')
        }


class NelderMead(ScipyOptimizer):
    """Nelder-Mead simplex optimizer."""

    def __init__(self, maxiter: int = 100, tol: float = 1e-2):
        super().__init__(method="Nelder-Mead", maxiter=maxiter, tol=tol, supports_bounds=True)


# ---------------------------------------------------------------------------
# PDFO optimizers (Powell's Derivative-Free Optimization solvers)
# Requires: pip install pdfo  (installed in the "quantum" conda environment)
# Reference: https://www.pdfo.net
# ---------------------------------------------------------------------------

class PdfoOptimizer(ClassicalOptimizer):
    """Base class for pdfo optimizers (NEWUOA, COBYLA, BOBYQA, LINCOA, UOBYQA).

    pdfo uses `maxfev` (max function evaluations) rather than `maxiter`.
    The `maxiter` constructor argument is forwarded as `maxfev`.
    """

    def __init__(self, method: str, maxiter: int = 100, supports_bounds: bool = False):
        super().__init__(maxiter=maxiter, tol=0.0)  # pdfo manages its own tolerance
        if not _PDFO_AVAILABLE:
            raise ImportError(
                "pdfo is not installed. Run: pip install pdfo  "
                "(or: conda install -c conda-forge pdfo  inside the 'quantum' env)"
            )
        self.method = method
        self.supports_bounds = supports_bounds

    def optimize(
        self,
        objective_function: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "method": self.method,
            "options": {"maxfev": self.maxiter},
        }
        if bounds is not None and self.supports_bounds:
            kwargs["bounds"] = bounds
        elif bounds is not None:
            logger.warning(
                "%s does not support bounds — they will be ignored.", self.method
            )

        assert _pdfo is not None  # guaranteed by __init__ ImportError guard
        logger.info("Starting %s (pdfo) optimisation…", self.method)
        result = _pdfo.pdfo(objective_function, initial_point, **kwargs)
        logger.debug(
            "%s finished in %s evaluations.", self.method, getattr(result, "nfev", "?")
        )
        return {
            "x":       result.x,
            "fun":     result.fun,
            "nfev":    getattr(result, "nfev", None),
            "success": getattr(result, "success", True),
            "message": getattr(result, "message", ""),
        }


class NEWUOA(PdfoOptimizer):
    """NEWUOA — NEW Unconstrained Optimization Algorithm (Powell, 2006).

    Model-based trust-region method for unconstrained problems.
    Typically the most efficient pdfo solver when no constraints are needed.
    """

    def __init__(self, maxiter: int = 100):
        super().__init__(method="newuoa", maxiter=maxiter, supports_bounds=False)


class COBYLA_PDFO(PdfoOptimizer):
    """COBYLA via pdfo — Powell's original Fortran implementation.

    Uses linear approximations to handle nonlinear constraints.
    More robust than scipy's COBYLA for ill-conditioned problems.
    """

    def __init__(self, maxiter: int = 100):
        super().__init__(method="cobyla", maxiter=maxiter, supports_bounds=False)


class BOBYQA(PdfoOptimizer):
    """BOBYQA — Bound Optimization BY Quadratic Approximation (Powell, 2009).

    Handles bound constraints natively; uses quadratic models.
    """

    def __init__(self, maxiter: int = 100):
        super().__init__(method="bobyqa", maxiter=maxiter, supports_bounds=True)


class LINCOA(PdfoOptimizer):
    """LINCOA — LINearly Constrained Optimization Algorithm (Powell, 2014).

    Handles linear inequality constraints; uses quadratic models.
    Works well for problems with box constraints passed as bounds.
    """

    def __init__(self, maxiter: int = 100):
        super().__init__(method="lincoa", maxiter=maxiter, supports_bounds=True)


class UOBYQA(PdfoOptimizer):
    """UOBYQA — Unconstrained Optimization BY Quadratic Approximation (Powell, 2002).

    Predecessor of NEWUOA; generally slower but sometimes more robust on
    very low-dimensional problems (n ≤ 2·reps for QAOA).
    """

    def __init__(self, maxiter: int = 100):
        super().__init__(method="uobyqa", maxiter=maxiter, supports_bounds=False)
