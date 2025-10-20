#!/usr/bin/env python3
"""
Evaluation script for Whisper Turbo KSC2 model on test_radio dataset.
Measures WER (Word Error Rate) and CER (Character Error Rate).
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import evaluate
from jiwer import mer
from tqdm import tqdm
import json
import time
from datetime import datetime
import os
from pathlib import Path
import soundfile as sf
import argparse


def load_model_and_processor(model_id: str, device: str = "cuda:0"):
    """Load the Whisper model and processor."""
    print(f"Loading model: {model_id}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
    )

    return pipe


def load_local_dataset(data_dir: str):
    """Load the test dataset from local directory."""
    print(f"Loading dataset from local directory: {data_dir}")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Directory {data_dir} does not exist")

    # Find all .flac files
    audio_files = sorted(list(data_path.glob("*.flac")))

    if not audio_files:
        raise ValueError(f"No .flac files found in {data_dir}")

    print(f"Found {len(audio_files)} audio files")

    # Create dataset structure
    dataset = []
    for audio_file in audio_files:
        # Get corresponding text file
        text_file = audio_file.with_suffix('.txt')

        if not text_file.exists():
            print(f"Warning: Text file not found for {audio_file.name}, skipping...")
            continue

        # Read transcription
        with open(text_file, 'r', encoding='utf-8') as f:
            transcription = f.read().strip()

        dataset.append({
            'audio_path': str(audio_file),
            'text': transcription
        })

    print(f"Loaded {len(dataset)} samples with transcriptions")
    return dataset


def evaluate_model(pipe, dataset, max_samples: int = None, batch_size: int = 1):
    """
    Evaluate the model on the dataset.

    Args:
        pipe: The ASR pipeline
        dataset: The test dataset (list of dicts with 'audio_path' and 'text')
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Batch size for processing (default: 1)

    Returns:
        Dictionary with evaluation results
    """
    # Load metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []

    # Process samples
    samples_to_process = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    print(f"Evaluating on {samples_to_process} samples with batch size {batch_size}...")

    start_time = time.time()

    # Process in batches
    for batch_start in tqdm(range(0, samples_to_process, batch_size)):
        batch_end = min(batch_start + batch_size, samples_to_process)
        batch_samples = dataset[batch_start:batch_end]

        # Get batch data
        batch_audio_paths = [sample['audio_path'] for sample in batch_samples]
        batch_references = [sample['text'] for sample in batch_samples]

        # Run inference on batch
        try:
            batch_results = pipe(
                batch_audio_paths,
                generate_kwargs={"language": "kk", "task": "transcribe"},
                return_timestamps=False,
                batch_size=batch_size
            )

            # Handle single vs batch results
            if batch_size == 1:
                batch_predictions = [batch_results["text"]]
            else:
                batch_predictions = [result["text"] for result in batch_results]

        except Exception as e:
            print(f"\nError processing batch {batch_start}-{batch_end}: {e}")
            batch_predictions = [""] * len(batch_samples)

        predictions.extend(batch_predictions)
        references.extend(batch_references)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    mer_value = mer(references, predictions)  # jiwer.mer expects (reference, hypothesis)

    # Calculate additional statistics
    avg_time_per_sample = total_time / samples_to_process

    results = {
        "model": pipe.model.name_or_path if hasattr(pipe.model, 'name_or_path') else "unknown",
        "dataset_split": "test_radio",
        "num_samples": samples_to_process,
        "metrics": {
            "wer": wer * 100,  # Convert to percentage
            "cer": cer * 100,  # Convert to percentage
            "mer": mer_value * 100   # Convert to percentage
        },
        "timing": {
            "total_time_seconds": total_time,
            "avg_time_per_sample": avg_time_per_sample
        },
        "timestamp": datetime.now().isoformat(),
        "sample_predictions": [
            {
                "reference": references[i],
                "prediction": predictions[i]
            }
            for i in range(min(5, len(predictions)))  # Save first 5 examples
        ]
    }

    return results


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper models on test_radio dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model with batch size 8 (default)
  python evaluate_whisper.py

  # Use a different model
  python evaluate_whisper.py --model openai/whisper-large-v3-turbo

  # Use larger batch size for faster processing (requires more GPU memory)
  python evaluate_whisper.py --batch-size 16

  # Test on first 100 samples
  python evaluate_whisper.py --max-samples 100

  # Specify custom data directory
  python evaluate_whisper.py --data-dir /path/to/data

  # Combine options
  python evaluate_whisper.py --model openai/whisper-large-v3-turbo --batch-size 16 --max-samples 50
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="abilmansplus/whisper-turbo-ksc2",
        help="HuggingFace model ID (default: abilmansplus/whisper-turbo-ksc2)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="test_radio/radio",
        help="Path to directory with audio files and transcriptions (default: test_radio/radio)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu'). If not specified, uses CUDA if available"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: evaluation_results_TIMESTAMP.json)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing audio files (default: 8). Larger values are faster but use more GPU memory"
    )

    args = parser.parse_args()

    # Configuration
    MODEL_ID = args.model
    DATA_DIR = args.data_dir
    DEVICE = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    MAX_SAMPLES = args.max_samples
    BATCH_SIZE = args.batch_size

    print("="*80)
    print("Whisper Model Evaluation")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    if MAX_SAMPLES:
        print(f"Max samples: {MAX_SAMPLES}")
    print("="*80)

    # Load model
    pipe = load_model_and_processor(MODEL_ID, DEVICE)

    # Load dataset
    dataset = load_local_dataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} samples")

    # Evaluate
    results = evaluate_model(pipe, dataset, max_samples=MAX_SAMPLES, batch_size=BATCH_SIZE)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Number of samples: {results['num_samples']}")
    print(f"WER (Word Error Rate): {results['metrics']['wer']:.2f}%")
    print(f"CER (Character Error Rate): {results['metrics']['cer']:.2f}%")
    print(f"MER (Match Error Rate): {results['metrics']['mer']:.2f}%")
    print(f"Total time: {results['timing']['total_time_seconds']:.2f} seconds")
    print(f"Average time per sample: {results['timing']['avg_time_per_sample']:.2f} seconds")
    print("="*80)

    # Print sample predictions
    print("\nSample predictions:")
    for i, sample in enumerate(results['sample_predictions'], 1):
        print(f"\nExample {i}:")
        print(f"  Reference:  {sample['reference']}")
        print(f"  Prediction: {sample['prediction']}")

    # Save results to JSON
    if args.output:
        output_file = args.output
    else:
        # Create filename with model name
        model_name_safe = MODEL_ID.replace("/", "-").replace("\\", "-")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"evaluation_results_{model_name_safe}_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
