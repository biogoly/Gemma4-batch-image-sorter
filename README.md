# Gemma Vision Sorter

Gemma Vision Sorter scans an image folder with a locally served vision-language
model and copies or moves matching images into an output folder.

Examples of characteristics:

- `a bicycle is clearly visible anywhere in the image`
- `at least one domestic cat is visible`
- `a person with visibly blonde hair is present`

The tool is model-agnostic. It sends standard OpenAI-compatible multimodal chat
requests, so the model can be swapped without changing the Python code.

## What it does

1. Walks `input/` (including subfolders by default in `run_sort.bat`).
2. Sends each image and the characteristic from `prompts/feature.txt` to the
   local model server.
3. Requires a structured verdict: match, confidence, and a brief reason.
4. Copies matches to `output/`, preserving the relative folder structure.
5. Writes every verdict, timing, action, error, and raw model response to CSV.

Copy is the default because it leaves the source collection untouched. Move is
available with `--mode move`.

## Requirements

- Windows 10/11
- Python 3.12
- A current `llama-server` build or another OpenAI-compatible multimodal server
- A vision-capable model and its multimodal projector (`mmproj`) when required

## Quick start

### 1. Install

Double-click `setup.bat`, or run:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Start the vision model server

If you already launch models through `llama-server`, keep using your normal
launcher and ensure multimodal support is enabled. A local-file example is:

```bat
llama-server.exe ^
  -m C:\models\model.gguf ^
  --mmproj C:\models\mmproj-model.gguf ^
  --host 127.0.0.1 --port 8080 ^
  -ngl all -c 8192 -fa on -ts 1,1
```

You can also edit `start_server_example.bat`. Current llama.cpp builds can load
supported Hugging Face model repositories with `-hf`, which normally obtains
the matching projector automatically.

### 3. Choose the characteristic

Edit `prompts\feature.txt`. Be literal about what counts as a match. For
example:

```text
a person with visibly blonde hair is present; dyed blonde counts, hats that
fully conceal the hair do not
```

### 4. Add images and run

Place images beneath `input\`, then double-click `run_sort.bat`. Matches appear
beneath `output\`. Results and timings appear beneath `logs\`.

## Command-line examples

One-off feature without editing the prompt file:

```bat
.venv\Scripts\python.exe sort_images.py --feature "a cat is visible" --recursive
```

Test the first ten images without copying anything:

```bat
.venv\Scripts\python.exe sort_images.py --feature "a bicycle is visible" --limit 10 --dry-run
```

Require a model-reported confidence of at least 0.75:

```bat
.venv\Scripts\python.exe sort_images.py --feature "a bicycle is visible" --min-confidence 0.75
```

Move matches instead of copying them:

```bat
.venv\Scripts\python.exe sort_images.py --feature "a bicycle is visible" --mode move
```

Label a comparison run:

```bat
.venv\Scripts\python.exe sort_images.py --feature "a bicycle is visible" --run-label gemma4-31b-q6
```

Run `python sort_images.py --help` for all options.

## Comparing models

Use the same input set, prompt, temperature, and output mode for each model.
Set a distinct `--run-label` such as `gemma4-e4b-q6`, `gemma4-31b-q6`, or a
custom 12B label. The CSV captures per-image latency, verdict, confidence,
reason, model argument, and run label.

For meaningful accuracy numbers, make a small hand-labelled test set and add a
ground-truth column after each run. Raw model confidence is useful for sorting
and error analysis but should not be treated as calibrated probability.

## Notes on Gemma model names

As of July 2026, llama.cpp's supported multimodal list includes Gemma 4 E2B,
E4B, 26B-A4B, and 31B. It does not list an official Gemma 4 12B checkpoint.
If `Gemma-4-12B` is a custom checkpoint, this sorter will still work provided
your server exposes it as a vision model. If you meant the official Gemma 3
12B vision model, llama.cpp lists `ggml-org/gemma-3-12b-it-GGUF`.

## Behaviour worth knowing

- A model verdict is accepted when `match=true` and it meets
  `--min-confidence`. The default threshold is zero, so the boolean verdict is
  decisive.
- Existing destination files are skipped unless `--overwrite` is supplied.
- Failed images do not stop the batch. Errors are recorded in the CSV.
- The CSV is flushed after every image, so completed results survive Ctrl+C or
  most interruptions.
- `--temperature 0` is the default for repeatable classification.
- The default generation allowance is 512 tokens because reasoning-capable
  models may spend part of it internally. If a response is cut off only after
  complete `match` and `confidence` fields, the sorter safely recovers those
  fields and marks the reason as truncated.
- If a server does not accept JSON Schema response formatting, retry with
  `--no-json-schema`.

## Supported input extensions

JPG, JPEG, PNG, WebP, BMP, GIF, TIFF, and TIF. Actual decode support depends on
the model server.

## Run tests

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```
