"""PettingZoo AECEnv wrapper around gameEngine.

Action contract — Approach 1 pass-through:
    env.step(action) accepts an engine.Action dataclass directly. The
    factoring of the action into (asset, verb, target) lives in the policy
    network's heads, not in the env. action_space is nominal.

Observation contract:
    See `observation.build_observation()` — dict of (entity_tokens, globals).
    Legal actions are delivered via self.infos[agent]["legal_actions"] per
    PettingZoo convention.

Reward contract:
    Sparse zero-sum, terminal-only. Winner: +1, Loser: -1, otherwise 0.
    Potential-based shaping (held in reserve per project plan) would be
    added inside _settle_rewards if/when training stalls.

Termination vs truncation:
    terminated: engine.winner is not None (capture of enemy home port).
    truncated:  turn count exceeds self.max_turns (avoid infinite games).
"""

from typing import Any, Dict, Optional

from gymnasium.spaces import Space
from pettingzoo import AECEnv

from battleboats.core.actions import Action, EndTurnAction
from battleboats.core.gameEngine import gameEngine
from battleboats.envs import observation

AGENTS = ("player_0", "player_1")
DEFAULT_MAX_TURNS = 500


class BattleboatsAEC(AECEnv):
    """Two-player turn-based naval game wrapped as a PettingZoo AECEnv.

    Lifecycle (PettingZoo convention):
        env = BattleboatsAEC(map_json_path)
        env.reset(seed=0)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            if terminated or truncated:
                action = None
            else:
                action = my_policy(obs, info["legal_actions"])
            env.step(action)
    """

    metadata = {"name": "battleboats_aec_v0", "is_parallelizable": False}

    def __init__(self, map_json_path: str, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        """Construct env without resetting.

        PettingZoo convention: agents / spaces / state are not finalized until
        reset() is called. Store config here; populate state in reset().
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent: str) -> Space:
        """Nominal observation space.

        Our actual obs is a dict with a variable-length token tensor, which
        doesn't cleanly fit any single gym Space primitive. Options:
          - Declare a Dict with Box entries using nominal shapes (token shape
            with N=some upper bound; will not match real obs but satisfies
            tooling that introspects the space)
          - Declare a placeholder Space() subclass
        Pick what matches your training loop's expectations.
        """
        raise NotImplementedError

    def action_space(self, agent: str) -> Space:
        """Nominal action space (Approach 1: actions are engine.Action objects).

        Return whatever placeholder works for your training framework. Most
        PettingZoo wrappers don't actually use this; we use it for tooling
        compatibility only.
        """
        raise NotImplementedError

    # --------------------------------------------------------------- lifecycle
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Start a new game.

        Steps:
            1. self.engine = gameEngine(self.map_json_path); engine.reset(seed)
            2. self.agents = list(AGENTS)
            3. self.agent_selection = AGENTS[0]  # mirrors engine.current_player == 0
            4. Initialize all per-agent dicts:
                self.rewards = {a: 0.0 for a in self.agents}
                self._cumulative_rewards = {a: 0.0 for a in self.agents}
                self.terminations = {a: False for a in self.agents}
                self.truncations = {a: False for a in self.agents}
                self.infos = {a: {} for a in self.agents}
            5. Populate self.infos[agent_selection]["legal_actions"] for the
               first acting agent.

        PettingZoo convention: reset returns None; observations come via
        observe() / last().
        """
        raise NotImplementedError

    def step(self, action: Optional[Action]) -> None:
        """Apply `action` for self.agent_selection.

        Lifecycle:
            1. If the current agent is already terminated/truncated:
               - PettingZoo's "dead agent" convention says the action should
                 be None. We honor that — no engine.step, just bookkeeping.
               - Advance agent_selection to the next live agent (or remove
                 from self.agents and end the cycle).
               - Return early.
            2. Delegate to engine.step(action). Engine ignores illegal actions
               (silent no-op); if you want hard failure on illegal, validate
               via engine.enumerate_legal() before stepping.
            3. Reset self.rewards to zero for all agents (per-step delta).
            4. Check for terminal state:
                 - engine.winner is not None → terminated for both, deliver +1/-1
                 - engine.turn >= self.max_turns → truncated for both, deliver 0/0
            5. Accumulate rewards into self._cumulative_rewards.
            6. Update agent_selection:
                 - If action was EndTurnAction, engine.current_player has
                   flipped; mirror that flip in agent_selection.
                 - Otherwise agent_selection stays put (same player continues
                   with another sub-action).
            7. Refresh self.infos[agent_selection]["legal_actions"] = engine.enumerate_legal(...)
        """
        raise NotImplementedError

    def observe(self, agent: str) -> Dict[str, Any]:
        """Build the fog-of-war-filtered obs dict for `agent`.

        Delegates to observation.build_observation(self.engine, player_id).
        """
        raise NotImplementedError

    def render(self) -> None:
        """Optional human-readable rendering. Skip for now; pygame UI later."""
        pass

    def close(self) -> None:
        """No persistent resources to release."""
        pass

    # ----------------------------------------------------------------- helpers
    def _player_id(self, agent: str) -> int:
        """`"player_0"` → 0, `"player_1"` → 1."""
        return int(agent.split("_")[1])

    def _agent_name(self, player_id: int) -> str:
        """0 → `"player_0"`, 1 → `"player_1"`."""
        return f"player_{player_id}"

    def _settle_terminal_rewards(self) -> None:
        """Write +1/-1 into self.rewards when engine.winner is set.

        Called from step() once the engine reports a terminal state.
        Reward shaping additions would go here (e.g., potential-based shaping
        based on min distance to known enemy home port).
        """
        raise NotImplementedError
