import argparse
import json
import time
from typing import Any, Dict, List, Optional

import requests
from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection


MONGO_URI = "mongodb://10.3.70.10:27017/"
COLLECTION_NAME = "news_v1"
SEGMENT_API_URL = "http://localhost:9900/segment"


def json_safe(value: Any) -> Any:
    """
    Convert MongoDB/ObjectId values into JSON-safe values.
    """
    if isinstance(value, ObjectId):
        return str(value)

    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    return value


def extract_block_text(block: Any) -> str:
    """
    Extract text from a content block.

    Supports:
      - string block
      - dict block with common text fields
      - fallback: JSON string version of the block
    """
    if isinstance(block, str):
        return block

    if isinstance(block, dict):
        for key in ("text", "content", "body", "value"):
            value = block.get(key)
            if isinstance(value, str):
                return value

    return json.dumps(json_safe(block), ensure_ascii=False)

def call_segment_api(text: str, timeout: float = 120.0) -> Dict[str, Any]:
    response = requests.post(
        SEGMENT_API_URL,
        params={"text": text},
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} from segment API. "
            f"Response body: {response.text[:1000]}"
        )

    try:
        return response.json()
    except ValueError:
        return {
            "raw_response": response.text
        }

def process_collection(
    collection: Collection,
    output_path: str,
    limit: Optional[int] = None,
    sleep_seconds: float = 0.0,
) -> None:
    projection = {
        "_id": 1,
        "content_blocks": 1,
    }

    cursor = collection.find({}, projection, no_cursor_timeout=True)

    if limit is not None:
        cursor = cursor.limit(limit)

    results: List[Dict[str, Any]] = []

    processed_docs = 0
    processed_blocks = 0
    failed_blocks = 0

    try:
        for doc in cursor:
            doc_id = str(doc.get("_id"))
            content_blocks = doc.get("content_blocks", [])

            if not isinstance(content_blocks, list):
                content_blocks = [content_blocks]

            doc_result = {
                "_id": doc_id,
                "blocks": [],
            }

            for block_index, block in enumerate(content_blocks):
                text = extract_block_text(block)

                if not text.strip():
                    doc_result["blocks"].append({
                        "block_index": block_index,
                        "original_block": json_safe(block),
                        "input_text": text,
                        "segmentation": None,
                        "error": "empty_text",
                    })
                    continue

                try:
                    segmentation = call_segment_api(text)

                    doc_result["blocks"].append({
                        "block_index": block_index,
                        "original_block": json_safe(block),
                        "input_text": text,
                        "segmentation": segmentation,
                        "error": None,
                    })

                    processed_blocks += 1

                except Exception as exc:
                    failed_blocks += 1

                    doc_result["blocks"].append({
                        "block_index": block_index,
                        "original_block": json_safe(block),
                        "input_text": text,
                        "segmentation": None,
                        "error": str(exc),
                    })

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            results.append(doc_result)
            processed_docs += 1

            if processed_docs % 100 == 0:
                print(
                    f"Processed docs={processed_docs}, "
                    f"blocks={processed_blocks}, "
                    f"failed_blocks={failed_blocks}"
                )

    finally:
        cursor.close()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output file: {output_path}")
    print(f"Processed docs: {processed_docs}")
    print(f"Processed blocks: {processed_blocks}")
    print(f"Failed blocks: {failed_blocks}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        required=True,
        help="MongoDB database name",
    )
    parser.add_argument(
        "--output",
        default="segmented_news_v1.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between API calls in seconds",
    )

    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db = client[args.database]
    collection = db[COLLECTION_NAME]

    process_collection(
        collection=collection,
        output_path=args.output,
        limit=args.limit,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()