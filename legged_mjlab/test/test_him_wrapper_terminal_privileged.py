import pytest
import torch

from legged_mjlab.wrappers.him_wrapper import HIMRslRlWrapper


class FakeHimEnv:
    """Small vector-env double for the wrapper's terminal-observation contract."""

    num_envs = 4
    num_actions = 2
    num_privileged_obs = 3
    device = torch.device("cpu")

    def __init__(self, terminated, truncated, terminal_candidate=None):
        self.terminated = torch.as_tensor(terminated, dtype=torch.bool)
        self.truncated = torch.as_tensor(truncated, dtype=torch.bool)
        self.terminal_candidate = terminal_candidate

    @staticmethod
    def _actor(value):
        return torch.full((FakeHimEnv.num_envs, 45), float(value))

    @staticmethod
    def _critic(value):
        return torch.arange(
            value,
            value + FakeHimEnv.num_envs * FakeHimEnv.num_privileged_obs,
            dtype=torch.float32,
        ).reshape(FakeHimEnv.num_envs, FakeHimEnv.num_privileged_obs)

    def reset(self, env_ids=None):
        del env_ids
        return {
            "actor": self._actor(0),
            "critic": self._critic(100),
        }, {}

    def step(self, actions):
        assert tuple(actions.shape) == (self.num_envs, self.num_actions)
        infos = {}
        if self.terminal_candidate is not None:
            infos["termination_privileged_obs"] = self.terminal_candidate
        return (
            {"actor": self._actor(1), "critic": self._critic(200)},
            torch.ones(self.num_envs),
            self.terminated,
            self.truncated,
            infos,
        )


def _wrapper(env):
    wrapper = HIMRslRlWrapper(
        env,
        expected_privileged_obs_dim=env.num_privileged_obs,
        action_dim=env.num_actions,
    )
    reset_result = wrapper.reset()
    assert len(reset_result) == 2
    return wrapper


def test_terminal_privileged_full_batch_is_selected_for_partial_done():
    candidate = torch.arange(1000, 1012, dtype=torch.float32).reshape(4, 3)
    wrapper = _wrapper(
        FakeHimEnv(
            terminated=[False, True, False, False],
            truncated=[False, False, False, True],
            terminal_candidate=candidate,
        )
    )

    result = wrapper.step(torch.zeros(4, 2))

    assert len(result) == 7
    _, privileged, _, dones, infos, termination_ids, terminal_privileged = result
    assert torch.equal(termination_ids, torch.tensor([1, 3]))
    assert torch.equal(dones, torch.tensor([False, True, False, True]))
    assert torch.equal(terminal_privileged, candidate[[1, 3]])
    assert torch.equal(infos["termination_privileged_obs"], candidate[[1, 3]])
    assert privileged.shape == (4, 3)


def test_terminal_privileged_compact_batch_stays_aligned_with_partial_done_ids():
    candidate = torch.tensor(
        [[2000.0, 2001.0, 2002.0], [3000.0, 3001.0, 3002.0]]
    )
    wrapper = _wrapper(
        FakeHimEnv(
            terminated=[False, True, False, True],
            truncated=[False, False, False, False],
            terminal_candidate=candidate,
        )
    )

    result = wrapper.step(torch.zeros(4, 2))

    _, _, _, _, infos, termination_ids, terminal_privileged = result
    assert torch.equal(termination_ids, torch.tensor([1, 3]))
    assert torch.equal(terminal_privileged, candidate)
    assert torch.equal(infos["termination_privileged_obs"], candidate)


def test_no_done_returns_empty_terminal_privileged_batch_and_preserves_contract():
    wrapper = _wrapper(
        FakeHimEnv(
            terminated=[False, False, False, False],
            truncated=[False, False, False, False],
        )
    )

    result = wrapper.step(torch.zeros(4, 2))

    history, privileged, rewards, dones, infos, termination_ids, terminal_privileged = result
    assert history.shape == (4, 270)
    assert privileged.shape == (4, 3)
    assert rewards.shape == (4,)
    assert not bool(dones.any().item())
    assert termination_ids.shape == (0,)
    assert terminal_privileged.shape == (0, 3)
    assert infos["termination_privileged_obs"].shape == (0, 3)
    assert wrapper.termination_ids.shape == (0,)
    assert wrapper.termination_privileged_obs.shape == (0, 3)


@pytest.mark.parametrize("candidate_rows", [3, 5])
def test_terminal_candidate_batch_is_checked(candidate_rows):
    candidate = torch.zeros(candidate_rows, 3)
    wrapper = _wrapper(
        FakeHimEnv(
            terminated=[False, True, False, False],
            truncated=[False, False, False, False],
            terminal_candidate=candidate,
        )
    )

    # Neither N=4 nor K=1 matches this candidate batch.
    with pytest.raises(ValueError, match="termination privileged obs"):
        wrapper.step(torch.zeros(4, 2))
