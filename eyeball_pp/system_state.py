from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import subprocess
from typing import Iterable, Optional

from eyeball_pp.recorders import Checkpoint
from eyeball_pp.utils import time_range_to_str, time_to_str


@dataclass
class SystemState:
    start_time: datetime  # inclusive
    end_time: datetime  # exclusive
    start_commit_hash: Optional[str] = None
    end_commit_hash: Optional[str] = None
    description: Optional[str] = None
    rerun_id: Optional[str] = None
    checkpoints: Optional[list[Checkpoint]] = None

    def __str__(self) -> str:
        if self.rerun_id is not None:
            return f"Rerun on {time_to_str(datetime.fromisoformat(self.rerun_id))}"
        elif self.description is not None:
            return (
                time_range_to_str(self.start_time, self.end_time)
                + ": "
                + self.description
            )
        else:
            return time_range_to_str(self.start_time, self.end_time)

    def __hash__(self) -> int:
        return hash(
            (
                self.start_time,
                self.end_time,
                self.start_commit_hash,
                self.end_commit_hash,
                self.rerun_id,
            )
        )


def bucketize_checkpoints(
    checkpoints: Iterable[Checkpoint],
) -> dict[SystemState, list[Checkpoint]]:
    checkpoints_by_state: dict[SystemState, list[Checkpoint]] = defaultdict(list)
    since: Optional[datetime] = None
    till: Optional[datetime] = None
    rerun_ids: list[str] = []

    for checkpoint in checkpoints:
        if since is None or checkpoint.created_at < since:
            since = checkpoint.created_at
        if till is None or checkpoint.created_at > till:
            till = checkpoint.created_at
        if checkpoint.rerun_metadata:
            rerun_ids.append(checkpoint.rerun_metadata["id"])

    if till is not None:
        till = till + timedelta(days=1)

    buckets = get_recent_system_states(since=since, till=till, rerun_ids=rerun_ids)

    already_bucketized: set[str] = set()

    for bucket in reversed(buckets):
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id in already_bucketized:
                continue

            # For a checkpoint which has a rerun, it should be bucketized in a bucket with a rerun id
            if checkpoint.rerun_metadata:
                if (
                    bucket.rerun_id is not None
                    and checkpoint.rerun_metadata["id"] == bucket.rerun_id
                ):
                    checkpoints_by_state[bucket].append(checkpoint)
                    already_bucketized.add(checkpoint.checkpoint_id)
            elif bucket.start_time <= checkpoint.created_at < bucket.end_time:
                checkpoints_by_state[bucket].append(checkpoint)
                already_bucketized.add(checkpoint.checkpoint_id)

    return checkpoints_by_state


def get_recent_system_states(
    since: Optional[datetime] = None,
    till: Optional[datetime] = None,
    rerun_ids: Optional[Iterable[str]] = None,
) -> list[SystemState]:
    # Returns a list of system states that cover the time period between since and till
    # They are sorted by time

    now = datetime.now()

    if till is None:
        till = now

    if since is None:
        since = till - timedelta(days=14)

    if rerun_ids is None:
        rerun_ids = []

    system_sates: list[SystemState] = []

    sorted_rerun_ids = sorted(rerun_ids)
    start_time = since
    rerun_index = 0
    while start_time < till or rerun_index < len(sorted_rerun_ids):
        end_time = min(till, start_time + timedelta(days=2))
        if rerun_index < len(sorted_rerun_ids):
            rerun_id = sorted_rerun_ids[rerun_index]
            rerun_time = datetime.fromisoformat(rerun_id)
            if rerun_time < end_time:
                rerun_index += 1
                # System state till rerun
                system_sates.append(SystemState(start_time, rerun_time))
                # System state with rerun
                end_time = rerun_time + timedelta(milliseconds=1)
                system_sates.append(
                    SystemState(
                        rerun_time,
                        end_time,
                        rerun_id=rerun_id,
                    )
                )
            else:
                system_sates.append(SystemState(start_time, end_time))
        else:
            system_sates.append(SystemState(start_time, end_time))

        start_time = end_time

    return system_sates


_SYSTEM_TAGS: Optional[list[str]] = None

def get_system_tags() -> list[str]:
    global _SYSTEM_TAGS
    if _SYSTEM_TAGS is not None:
        return _SYSTEM_TAGS
    
    tags = []
    result = subprocess.run(
        ["""echo "$(git branch --show-current),$(git config user.email)" """],
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        stripped_result = result.stdout.strip()
        if stripped_result:
            split_result = stripped_result.split(",")
            if len(split_result) == 2:
                branch, user_email = split_result
                if branch:
                    tags.append(f"git_branch:{split_result[0]}")
                if user_email:
                    tags.append(f"git_user_email:{split_result[1]}")

    _SYSTEM_TAGS = tags
    return tags
