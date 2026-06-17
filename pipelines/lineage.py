"""M4.4b — emit OpenLineage events to Marquez, recording the pipeline's lineage.

Each pipeline stage is a JOB; one execution is a RUN (START + COMPLETE events);
inputs/outputs are DATASETS. Marquez stitches them into a graph by matching dataset
names — the OUTPUT of one job is the INPUT of the next:

    huggingface.flare-fpb -> [ingest] -> silver.financial_phrasebank
                          -> [train]  -> model.baseline
                          -> [register]-> model.fpb-sentiment

View after running: http://localhost:3001  (namespace "adapterforge")
"""

from datetime import datetime, timezone

from openlineage.client import OpenLineageClient
from openlineage.client.run import Dataset, Job, Run, RunEvent, RunState
from openlineage.client.uuid import generate_new_uuid

MARQUEZ_URL = "http://localhost:5000"
NAMESPACE = "adapterforge"
PRODUCER = "https://github.com/adapterforge/adapterforge"

client = OpenLineageClient(url=MARQUEZ_URL)


def ds(name: str) -> Dataset:
    """Shorthand for a dataset node in our namespace."""
    return Dataset(namespace=NAMESPACE, name=name)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_job(job_name: str, inputs: list[Dataset], outputs: list[Dataset]) -> None:
    """Emit START then COMPLETE for one job run, carrying its input/output datasets."""
    run = Run(runId=str(generate_new_uuid()))
    job = Job(namespace=NAMESPACE, name=job_name)
    client.emit(RunEvent(eventType=RunState.START, eventTime=_now(),
                         run=run, job=job, producer=PRODUCER,
                         inputs=inputs, outputs=outputs))
    client.emit(RunEvent(eventType=RunState.COMPLETE, eventTime=_now(),
                         run=run, job=job, producer=PRODUCER,
                         inputs=inputs, outputs=outputs))


def main() -> None:
    """Emit the 3-stage pipeline lineage. Output of each stage = input of the next."""
    emit_job("ingest_phrasebank",
             [ds("huggingface.flare-fpb")], [ds("silver.financial_phrasebank")])
    emit_job("train_baseline",
             [ds("silver.financial_phrasebank")], [ds("model.baseline")])
    emit_job("register_model",
             [ds("model.baseline")], [ds("model.fpb-sentiment")])
    print("emitted lineage to Marquez")


if __name__ == "__main__":
    main()
