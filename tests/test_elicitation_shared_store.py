import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from genesis.elicitation import ElicitationSession, ElicitationSessionStore


def session(session_id: str) -> ElicitationSession:
    return ElicitationSession(
        session_id=session_id,
        specification_id="study",
        workflow_id="workflow",
        workflow_version="1",
        model_profile_id="model",
        researcher_id="researcher",
        current_stage="foundation",
        base_specification_version=1,
    )


def test_instances_preserve_each_others_sessions_and_refresh_reads(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    first = ElicitationSessionStore(path)
    second = ElicitationSessionStore(path)
    first.put(session("first"))
    second.put(session("second"))
    assert {s.session_id for s in first.list_sessions()} == {"first", "second"}
    second.update("first", lambda s: setattr(s, "current_question", "Updated question"))
    assert first.get("first").current_question == "Updated question"
    first.delete("second")
    assert {s.session_id for s in second.list_sessions()} == {"first"}


def test_idempotency_is_shared_and_mutations_are_serialized(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    stores = [ElicitationSessionStore(path), ElicitationSessionStore(path)]
    stores[0].put(session("first"))
    barrier = Barrier(2)

    def submit(store: ElicitationSessionStore) -> dict:
        barrier.wait(timeout=5)

        def mutate() -> dict:
            updated = store.update(
                "first",
                lambda s: setattr(
                    s, "base_specification_version", s.base_specification_version + 1
                ),
            )
            return {"version": updated.base_specification_version}

        return store.run_idempotent("first", "request", "payload", mutate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, stores))
    assert results == [{"version": 2}, {"version": 2}]
    assert stores[1].get("first").base_specification_version == 2
    assert stores[0].idempotency_size == stores[1].idempotency_size == 1


def _increment_in_process(path: Path) -> None:
    store = ElicitationSessionStore(path)
    for _ in range(12):
        store.update(
            "first",
            lambda s: setattr(s, "base_specification_version", s.base_specification_version + 1),
        )


def test_processes_do_not_lose_concurrent_updates(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    store = ElicitationSessionStore(path)
    store.put(session("first"))
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        list(executor.map(_increment_in_process, [path, path]))
    assert store.get("first").base_specification_version == 25
