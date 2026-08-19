import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from sort_images import (
    Verdict,
    VisionClient,
    destination_for,
    iter_images,
    main,
    parse_verdict,
    transfer_match,
)


class ParseVerdictTests(unittest.TestCase):
    def test_plain_json(self):
        verdict = parse_verdict(
            '{"match": true, "confidence": 0.91, "reason": "A bicycle is visible."}'
        )
        self.assertTrue(verdict.match)
        self.assertEqual(verdict.confidence, 0.91)

    def test_fenced_json(self):
        verdict = parse_verdict(
            '```json\n{"match": false, "confidence": 0.8, "reason": "No cat."}\n```'
        )
        self.assertFalse(verdict.match)

    def test_rejects_string_boolean(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"match": "yes", "confidence": 0.9, "reason": ""}')

    def test_recovers_when_only_reason_is_truncated(self):
        raw = (
            '{\n  "match": true,\n  "confidence": 1.0,\n'
            '  "reason": "There is a large watermark that'
        )
        verdict = parse_verdict(raw)
        self.assertTrue(verdict.match)
        self.assertEqual(verdict.confidence, 1.0)
        self.assertEqual(
            verdict.reason,
            "There is a large watermark that [response truncated]",
        )

    def test_does_not_recover_without_complete_critical_fields(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"match": true, "confid')

    def test_does_not_recover_malformed_complete_object(self):
        with self.assertRaises(ValueError):
            parse_verdict('{"match": true, "confidence": 1.0,}')


class FileHandlingTests(unittest.TestCase):
    def test_recursive_listing_excludes_nested_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = input_dir / "output"
            (input_dir / "nested").mkdir(parents=True)
            output_dir.mkdir()
            (input_dir / "a.jpg").write_bytes(b"a")
            (input_dir / "nested" / "b.PNG").write_bytes(b"b")
            (input_dir / "notes.txt").write_text("x")
            (output_dir / "old.jpg").write_bytes(b"old")

            found = list(iter_images(input_dir, output_dir, recursive=True))

            self.assertEqual(found, [input_dir / "a.jpg", input_dir / "nested" / "b.PNG"])

    def test_destination_preserves_relative_tree(self):
        self.assertEqual(
            destination_for(Path("input/a/b.jpg"), Path("input"), Path("output")),
            Path("output/a/b.jpg"),
        )

    def test_copy_and_skip_existing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpg"
            destination = root / "out" / "source.jpg"
            source.write_bytes(b"image")
            self.assertEqual(
                transfer_match(
                    source,
                    destination,
                    mode="copy",
                    dry_run=False,
                    overwrite=False,
                ),
                "copy",
            )
            self.assertEqual(destination.read_bytes(), b"image")
            self.assertEqual(
                transfer_match(
                    source,
                    destination,
                    mode="copy",
                    dry_run=False,
                    overwrite=False,
                ),
                "skipped-existing",
            )


class VisionClientTests(unittest.TestCase):
    def test_multimodal_request_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "test.jpg"
            image.write_bytes(b"not-a-real-image")
            client = VisionClient(
                base_url="http://localhost:8080/v1",
                model="local-model",
                api_key="x",
                timeout=1,
                retries=0,
                max_tokens=64,
                temperature=0,
                use_json_schema=True,
            )
            response = Mock(spec=requests.Response)
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"match": True, "confidence": 0.9, "reason": "visible"}
                            )
                        }
                    }
                ]
            }
            client.session.post = Mock(return_value=response)

            verdict = client.classify(image, "a bicycle")

            self.assertTrue(verdict.match)
            payload = client.session.post.call_args.kwargs["json"]
            content = payload["messages"][1]["content"]
            self.assertEqual(content[0]["type"], "text")
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            self.assertEqual(payload["response_format"]["type"], "json_schema")
            self.assertEqual(
                payload["response_format"]["json_schema"]["schema"]["properties"][
                    "reason"
                ]["maxLength"],
                160,
            )


class CommandTests(unittest.TestCase):
    def test_main_copies_match_and_writes_audit_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            log_file = root / "logs" / "result.csv"
            input_dir.mkdir()
            (input_dir / "bike.jpg").write_bytes(b"image")
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(
                    VisionClient,
                    "classify",
                    return_value=Verdict(True, 0.95, "Bicycle visible.", "raw-json"),
                ):
                    result = main(
                        [
                            "--input-dir",
                            str(input_dir),
                            "--output-dir",
                            str(output_dir),
                            "--feature",
                            "a bicycle",
                            "--log-file",
                            str(log_file),
                        ]
                    )
            finally:
                os.chdir(old_cwd)

            self.assertEqual(result, 0)
            self.assertEqual((output_dir / "bike.jpg").read_bytes(), b"image")
            with log_file.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["match"], "True")
            self.assertEqual(rows[0]["action"], "copy")


if __name__ == "__main__":
    unittest.main()
