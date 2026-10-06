"""Who is using which document, so a document can be deleted when nobody is.

A document id is a content hash, so two people uploading the same text share one
document. Deleting it when one of them leaves would break the other. Each browser
tab therefore registers as an owner, and a document goes only when its last owner
releases it (page reload or close) or has been idle past the TTL.

State is in this process. That is correct for one worker (the free demo) and
wrong for several, so expiry is off unless DOCUMENT_EXPIRY_ENABLED is set.
"""
from __future__ import annotations

import threading
import time
from typing import Callable


class DocumentRegistry:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._owners: dict[str, dict[str, float]] = {}  # document -> {tab id: last seen}
        self._lock = threading.Lock()

    def touch(self, document_id: str, tab_id: str) -> None:
        """Register or refresh an owner."""
        with self._lock:
            self._owners.setdefault(document_id, {})[tab_id] = self._clock()

    def release(self, document_id: str, tab_id: str) -> bool:
        """Drop one owner. True when nobody is left, so the caller deletes the document."""
        with self._lock:
            owners = self._owners.get(document_id)
            if owners is None:
                return False
            owners.pop(tab_id, None)
            if owners:
                return False
            del self._owners[document_id]
            return True

    def expired(self, ttl_seconds: float) -> list[str]:
        """Remove owners idle past the TTL; return documents left with none."""
        cutoff = self._clock() - ttl_seconds
        gone: list[str] = []
        with self._lock:
            for doc, owners in list(self._owners.items()):
                for tab, seen in list(owners.items()):
                    if seen < cutoff:
                        del owners[tab]
                if not owners:
                    del self._owners[doc]
                    gone.append(doc)
        return gone

    def tracked(self) -> int:
        with self._lock:
            return len(self._owners)
