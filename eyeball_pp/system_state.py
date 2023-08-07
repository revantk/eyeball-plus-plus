from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import git


@dataclass
class SystemState:
    start_time: datetime
    end_time: datetime
    start_commit_hash: Optional[str] = None
    end_commit_hash: Optional[str] = None
    description: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.start_time} - {self.end_time}: {self.description}"


def get_recent_system_states(since: Optional[datetime] = None) -> list[SystemState]:
    now = datetime.now()

    if since is None:
        since = now - timedelta(days=7)

    repo = git.Repo(".")
    for commit in repo.iter_commits():
        print(commit.diff())
        break
    return [SystemState(now, now)]
