#!/usr/bin/env python3
"""
Evaluation script for SeamlessM4T v2 model on test_radio dataset.
Measures WER (Word Error Rate), CER (Character Error Rate), and MER (Match Error Rate).
"""

import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
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
import numpy as np
import traceback


def load_model_and_processor(model_id: str, device: str = "cuda:0"):
    """Load the SeamlessM4T model and processor for Speech-to-Text."""
    print(f"Loading model: {model_id}")

    processor = AutoProcessor.from_pretrained(model_id)

    # Use SeamlessM4Tv2ForSpeechToText for ASR (automatic speech recognition)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.to(device)

    print(f"Model loaded successfully on device: {device}")
    return model, processor


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


def load_audio(audio_path: str, target_sample_rate: int = 16000):
    """
    Load audio file and resample to target sample rate.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (SeamlessM4T uses 16kHz)

    Returns:
        Audio array and sample rate
    """
    try:
        audio, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sample_rate != target_sample_rate:
            # Simple resampling (for production, use librosa.resample)
            import scipy.signal
            num_samples = int(len(audio) * target_sample_rate / sample_rate)
            audio = scipy.signal.resample(audio, num_samples)

        return audio, target_sample_rate

    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return None, None


def evaluate_model(model, processor, dataset, max_samples: int = None, device: str = "cuda:0", src_lang: str = "kaz"):
    """
    Evaluate the model on the dataset.

    Args:
        model: The SeamlessM4T model
        processor: The SeamlessM4T processor
        dataset: The test dataset (list of dicts with 'audio_path' and 'text')
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to use
        src_lang: Source language code (default: kaz for Kazakh)

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

    print(f"Evaluating on {samples_to_process} samples...")
    print(f"Source language: {src_lang}")
    print("Note: SeamlessM4T processes audio sequentially (batch processing not supported for generation)")

    start_time = time.time()

    # Process samples one by one (SeamlessM4T doesn't support batched generation well)
    for idx in tqdm(range(samples_to_process)):
        sample = dataset[idx]

        # Get reference text
        reference = sample['text']

        # Load audio file
        audio, sr = load_audio(sample['audio_path'])

        # Run inference
        if audio is not None:
            try:
                # Prepare inputs for single sample
                inputs = processor(
                    audios=audio,
                    sampling_rate=16000,
                    return_tensors="pt"
                )

                # Move to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                # Generate transcription using .generate() method
                with torch.no_grad():
                    output_tokens = model.generate(
                        **inputs,
                        tgt_lang=src_lang,
                        num_beams=5,
                        max_new_tokens=256,
                        repetition_penalty=1.2,  # Penalty for repeating tokens
                        no_repeat_ngram_size=3   # Prevent repeating 3-grams
                    )

                # Decode to text
                transcription = processor.batch_decode(output_tokens, skip_special_tokens=True)
                prediction = transcription[0] if transcription else ""

            except Exception as e:
                print(f"\nError processing sample {idx} ({sample['audio_path']}): {e}")
                traceback.print_exc()
                prediction = ""
        else:
            prediction = ""

        predictions.append(prediction)
        references.append(reference)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    mer_value = mer(references, predictions)

    # Calculate additional statistics
    avg_time_per_sample = total_time / samples_to_process

    results = {
        "model": model.name_or_path if hasattr(model, 'name_or_path') else "SeamlessM4T",
        "dataset_split": "test_radio",
        "num_samples": samples_to_process,
        "source_language": src_lang,
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
        description="Evaluate SeamlessM4T v2 model on test_radio dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default SeamlessM4T model
  python evaluate_seamless.py

  # Use batch size 8 for faster processing
  python evaluate_seamless.py --batch-size 8

  # Test on first 100 samples
  python evaluate_seamless.py --max-samples 100

  # Specify custom data directory
  python evaluate_seamless.py --data-dir /path/to/data

  # Combine options
  python evaluate_seamless.py --batch-size 8 --max-samples 50

  # Use different language code (if needed)
  python evaluate_seamless.py --src-lang kaz
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="facebook/seamless-m4t-v2-large",
        help="HuggingFace model ID (default: facebook/seamless-m4t-v2-large)"
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
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for processing audio files (default: 4). Larger values are faster but use more GPU memory"
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
        help="Output JSON file path (default: evaluation_results_<model>_<timestamp>.json)"
    )

    parser.add_argument(
        "--src-lang",
        type=str,
        default="kaz",
        help="Source language code (default: kaz for Kazakh). See SeamlessM4T docs for supported codes."
    )

    args = parser.parse_args()

    # Configuration
    MODEL_ID = args.model
    DATA_DIR = args.data_dir
    DEVICE = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    MAX_SAMPLES = args.max_samples
    BATCH_SIZE = args.batch_size
    SRC_LANG = args.src_lang

    print("="*80)
    print("SeamlessM4T v2 Model Evaluation")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Source language: {SRC_LANG}")
    if MAX_SAMPLES:
        print(f"Max samples: {MAX_SAMPLES}")
    print("="*80)

    # Load model
    model, processor = load_model_and_processor(MODEL_ID, DEVICE)

    # Load dataset
    dataset = load_local_dataset(DATA_DIR)
    print(f"Dataset size: {len(dataset)} samples")

    # Evaluate
    results = evaluate_model(
        model,
        processor,
        dataset,
        max_samples=MAX_SAMPLES,
        device=DEVICE,
        src_lang=SRC_LANG
    )

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
