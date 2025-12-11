"""Neural ODE Boosting.

Models gradient boosting as a continuous-time ODE:
    df/dt = tree(x, t; θ)

Instead of discrete boosting rounds, we solve an ODE where predictions
evolve continuously. This provides:
- Implicit regularization through ODE dynamics
- Adaptive "step sizes" (like adaptive learning rate)
- Novel theoretical framework connecting boosting to gradient flows

Requires: diffrax (pip install diffrax)

Example:
    >>> from jaxboost.aggregation.ode_boosting import ODEBoosting
    >>> ode_boost = ODEBoosting(depth=4, t_span=(0.0, 1.0))
    >>> params = ode_boost.init_params(key, num_features=10)
    >>> predictions = ode_boost.forward(params, X)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxboost.routing import soft_routing
from jaxboost.splits import HyperplaneSplit
from jaxboost.structures import ObliviousTree

# Lazy import diffrax to make it optional
_diffrax = None


def _get_diffrax():
    """Lazy import of diffrax."""
    global _diffrax
    if _diffrax is None:
        try:
            import diffrax

            _diffrax = diffrax
        except ImportError as e:
            raise ImportError(
                "diffrax is required for ODE boosting. "
                "Install it with: pip install diffrax"
            ) from e
    return _diffrax


class ODEBoostingParams(NamedTuple):
    """Parameters for ODE boosting.

    Attributes:
        tree_params: Parameters for the tree that defines the ODE dynamics.
        time_embedding: Optional time embedding weights for time-dependent dynamics.
    """

    tree_params: Any  # ObliviousTreeParams
    time_embedding: Array | None  # (embed_dim,) or None


class ODEBoosting:
    """Neural ODE Boosting: continuous-time gradient boosting.

    Models boosting as solving an ODE:
        df(x, t)/dt = tree(x; θ)

    The prediction at time t=1 is the solution to this ODE starting from f(x, 0) = 0.

    Key insight: Traditional boosting with n trees and learning rate η is
    equivalent to Euler discretization of this ODE with step size η.
    ODE solvers can be more accurate and adaptive.

    Example:
        >>> ode_boost = ODEBoosting(depth=4)
        >>> params = ode_boost.init_params(key, num_features=10)
        >>> preds = ode_boost.forward(params, X)
        >>>
        >>> # With time-dependent dynamics
        >>> ode_boost = ODEBoosting(depth=4, time_dependent=True)
    """

    def __init__(
        self,
        depth: int = 4,
        t_span: tuple[float, float] = (0.0, 1.0),
        solver: str = "tsit5",
        dt0: float | None = None,
        max_steps: int = 100,
        temperature: float = 1.0,
        time_dependent: bool = False,
        time_embed_dim: int = 8,
    ) -> None:
        """Initialize ODE boosting.

        Args:
            depth: Depth of the tree defining dynamics.
            t_span: Time interval to integrate over, (t0, t1).
            solver: ODE solver to use. Options:
                - "euler": Simple Euler method (equivalent to traditional boosting)
                - "heun": Heun's method (2nd order)
                - "tsit5": Tsitouras 5(4) (adaptive, recommended)
                - "dopri5": Dormand-Prince 5(4) (adaptive)
            dt0: Initial step size. If None, adaptive.
            max_steps: Maximum number of ODE solver steps.
            temperature: Soft routing temperature.
            time_dependent: If True, tree dynamics depend on time t.
            time_embed_dim: Dimension of time embedding (if time_dependent).
        """
        self.depth = depth
        self.t_span = t_span
        self.solver_name = solver
        self.dt0 = dt0
        self.max_steps = max_steps
        self.temperature = temperature
        self.time_dependent = time_dependent
        self.time_embed_dim = time_embed_dim

        # Components
        self.split_fn = HyperplaneSplit()
        self.tree = ObliviousTree()

    def init_params(
        self,
        key: Array,
        num_features: int,
    ) -> ODEBoostingParams:
        """Initialize ODE boosting parameters.

        Args:
            key: JAX PRNG key.
            num_features: Number of input features.

        Returns:
            Initialized parameters.
        """
        keys = jax.random.split(key, 2)

        # Tree parameters (optionally with extra dims for time embedding)
        effective_features = num_features
        if self.time_dependent:
            effective_features += self.time_embed_dim

        tree_params = self.tree.init_params(
            keys[0],
            self.depth,
            effective_features,
            self.split_fn,
            init_leaf_scale=0.1,  # Larger for ODE dynamics
        )

        # Time embedding (sinusoidal encoding weights)
        time_embedding = None
        if self.time_dependent:
            time_embedding = jax.random.normal(keys[1], (self.time_embed_dim,))

        return ODEBoostingParams(
            tree_params=tree_params,
            time_embedding=time_embedding,
        )

    def _get_solver(self):
        """Get the ODE solver instance."""
        diffrax = _get_diffrax()

        solvers = {
            "euler": diffrax.Euler(),
            "heun": diffrax.Heun(),
            "tsit5": diffrax.Tsit5(),
            "dopri5": diffrax.Dopri5(),
            "reversible_heun": diffrax.ReversibleHeun(),
        }

        if self.solver_name not in solvers:
            raise ValueError(
                f"Unknown solver: {self.solver_name}. "
                f"Available: {list(solvers.keys())}"
            )

        return solvers[self.solver_name]

    def _time_embed(self, t: Array, embedding: Array) -> Array:
        """Sinusoidal time embedding.

        Args:
            t: Time value (scalar).
            embedding: Embedding weights, shape (embed_dim,).

        Returns:
            Time features, shape (embed_dim,).
        """
        # Sinusoidal encoding: sin/cos at different frequencies
        freqs = jnp.exp(embedding)  # Learnable frequencies
        angles = t * freqs
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])[: len(embedding)]

    def _dynamics(
        self,
        t: Array,
        y: Array,
        args: tuple,
    ) -> Array:
        """ODE dynamics: dy/dt = tree(x; θ).

        This is the "vector field" that the ODE solver integrates.

        Args:
            t: Current time.
            y: Current state (predictions), shape (batch,).
            args: (params, x) tuple.

        Returns:
            Time derivative dy/dt, shape (batch,).
        """
        params, x = args

        def routing_fn(score):
            return soft_routing(score, self.temperature)

        # Optionally augment input with time embedding
        if self.time_dependent and params.time_embedding is not None:
            time_features = self._time_embed(t, params.time_embedding)
            # Broadcast time features to batch
            batch_size = x.shape[0]
            time_features = jnp.broadcast_to(time_features, (batch_size, len(time_features)))
            x_augmented = jnp.concatenate([x, time_features], axis=-1)
        else:
            x_augmented = x

        # Tree output is the velocity field
        velocity = self.tree.forward(
            params.tree_params, x_augmented, self.split_fn, routing_fn
        )

        return velocity

    def forward(
        self,
        params: ODEBoostingParams,
        x: Array,
    ) -> Array:
        """Forward pass: solve ODE to get predictions.

        Args:
            params: Model parameters.
            x: Input features, shape (batch, num_features).

        Returns:
            Predictions at t=t1, shape (batch,).
        """
        diffrax = _get_diffrax()

        batch_size = x.shape[0]
        t0, t1 = self.t_span

        # Initial condition: predictions start at 0
        y0 = jnp.zeros(batch_size)

        # Setup ODE problem
        term = diffrax.ODETerm(self._dynamics)
        solver = self._get_solver()

        # Step size controller
        if self.dt0 is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
            dt0 = (t1 - t0) / 10  # Initial guess
        else:
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = self.dt0

        # Solve ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=(params, x),
            stepsize_controller=stepsize_controller,
            max_steps=self.max_steps,
        )

        return solution.ys[0]  # Final state

    def forward_trajectory(
        self,
        params: ODEBoostingParams,
        x: Array,
        save_times: Array | None = None,
    ) -> tuple[Array, Array]:
        """Forward pass returning full trajectory (for visualization).

        Args:
            params: Model parameters.
            x: Input features, shape (batch, num_features).
            save_times: Times at which to save state. If None, uses 10 points.

        Returns:
            Tuple of (times, predictions) where predictions has shape
            (num_times, batch).
        """
        diffrax = _get_diffrax()

        batch_size = x.shape[0]
        t0, t1 = self.t_span

        if save_times is None:
            save_times = jnp.linspace(t0, t1, 10)

        y0 = jnp.zeros(batch_size)

        term = diffrax.ODETerm(self._dynamics)
        solver = self._get_solver()

        if self.dt0 is None:
            stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
            dt0 = (t1 - t0) / 10
        else:
            stepsize_controller = diffrax.ConstantStepSize()
            dt0 = self.dt0

        saveat = diffrax.SaveAt(ts=save_times)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=(params, x),
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=self.max_steps,
        )

        return solution.ts, solution.ys


class EulerBoosting:
    """Explicit Euler discretization of ODE boosting.

    This is mathematically equivalent to traditional gradient boosting
    when the tree output is scaled by the learning rate (step size).

    Useful for:
    - Understanding the connection between ODE and discrete boosting
    - Fast training without diffrax dependency
    - Baseline comparison

    f_{n+1} = f_n + η * tree(x; θ_n)

    With a single tree reused at each step, this becomes:
    f_N = N * η * tree(x; θ) = tree(x; θ)  (if N*η = 1)
    """

    def __init__(
        self,
        depth: int = 4,
        num_steps: int = 10,
        step_size: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        """Initialize Euler boosting.

        Args:
            depth: Tree depth.
            num_steps: Number of Euler steps (like number of boosting rounds).
            step_size: Step size η (like learning rate).
            temperature: Soft routing temperature.
        """
        self.depth = depth
        self.num_steps = num_steps
        self.step_size = step_size
        self.temperature = temperature

        self.split_fn = HyperplaneSplit()
        self.tree = ObliviousTree()

    def init_params(self, key: Array, num_features: int) -> Any:
        """Initialize parameters (just tree params)."""
        return self.tree.init_params(
            key, self.depth, num_features, self.split_fn, init_leaf_scale=0.1
        )

    def forward(self, params: Any, x: Array) -> Array:
        """Forward pass using Euler integration.

        Args:
            params: Tree parameters.
            x: Input features, shape (batch, num_features).

        Returns:
            Predictions, shape (batch,).
        """

        def routing_fn(score):
            return soft_routing(score, self.temperature)

        # Euler integration: y_{n+1} = y_n + h * f(y_n)
        y = jnp.zeros(x.shape[0])

        for _ in range(self.num_steps):
            velocity = self.tree.forward(params, x, self.split_fn, routing_fn)
            y = y + self.step_size * velocity

        return y

    def forward_scan(self, params: Any, x: Array) -> Array:
        """Forward pass using jax.lax.scan (more efficient, JIT-friendly).

        Args:
            params: Tree parameters.
            x: Input features, shape (batch, num_features).

        Returns:
            Predictions, shape (batch,).
        """

        def routing_fn(score):
            return soft_routing(score, self.temperature)

        def step_fn(y, _):
            velocity = self.tree.forward(params, x, self.split_fn, routing_fn)
            y_new = y + self.step_size * velocity
            return y_new, None

        y0 = jnp.zeros(x.shape[0])
        y_final, _ = jax.lax.scan(step_fn, y0, None, length=self.num_steps)

        return y_final

