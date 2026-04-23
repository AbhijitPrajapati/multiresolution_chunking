import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def register_evaluation_batch(qids, prompts):
    with open("groq_batch/input.jsonl", "w") as f:
        for qid, p in zip(qids, prompts):
            r = {
                "custom_id": f"{qid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "openai/gpt-oss-120b",
                    "messages": [
                        {"role": "user", "content": p},
                    ],
                },
            }
            json.dump(r, f)
            f.write("\n")
    quit()

    file_id = client.files.create(
        file=open("groq_batch/input.jsonl", "rb"), purpose="batch"
    ).id
    assert file_id is not None

    batch_id = client.batches.create(
        completion_window="24h", endpoint="/v1/chat/completions", input_file_id=file_id
    ).id

    return batch_id


def get_evaluation_batch_response(id):
    response = client.batches.retrieve(batch_id=id)
    req_counts = response.request_counts
    assert req_counts is not None
    if response.status != "completed":
        raise Exception(
            f"Batch job is incomplete.\nCompleted Requests: {req_counts.completed}\nFailed Requests: {req_counts.failed}\nRemaining Requests: {req_counts.total - req_counts.completed - req_counts.failed}"
        )
    file_id = response.output_file_id
    assert file_id is not None
    output = client.files.content(file_id)
    output.write_to_file("groq_batch/output.jsonl")
    ids = []
    messages = []
    with open("groq_batch/output.jsonl", "r") as f:
        for line in f:
            r = json.loads(line)
            ids.append(r["custom_id"])
            messages.append(r["response"]["body"]["choices"][0]["message"]["content"])
    return ids, messages
