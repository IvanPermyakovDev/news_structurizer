import torch
from typing import List, Tuple
from transformers import AutoTokenizer
from pathlib import Path

from model import create_model


class NewsSegmenter:
    """Inference class for news segmentation."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        args = checkpoint['args']

        self.tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
        self.max_length = args['max_length']

        self.model = create_model(
            model_type=args['model_type'],
            model_name=args['model_name'],
            dropout=args['dropout']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def segment(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        Segment text into separate news items.

        Args:
            text: Combined news text
            threshold: Probability threshold for boundary detection

        Returns:
            List of segmented news texts
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0]

        outputs = self.model(input_ids, attention_mask)
        probs = torch.softmax(outputs['logits'], dim=-1)[0, :, 1]

        boundaries = [0]

        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start == 0 and end == 0:
                continue
            if attention_mask[0, i] == 0:
                continue

            if probs[i] > threshold and start > 0:
                if not boundaries or boundaries[-1] != start:
                    boundaries.append(start)

        boundaries.append(len(text))

        segments = []
        for i in range(len(boundaries) - 1):
            segment = text[boundaries[i]:boundaries[i + 1]].strip()
            if segment:
                segments.append(segment)

        return segments

    def get_boundary_probabilities(self, text: str) -> List[Tuple[str, float]]:
        """Get boundary probability for each token."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0]

        outputs = self.model(input_ids, attention_mask)
        probs = torch.softmax(outputs['logits'], dim=-1)[0, :, 1]

        result = []
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start == 0 and end == 0:
                continue
            if attention_mask[0, i] == 0:
                continue

            token_text = text[start:end]
            result.append((token_text, probs[i].item()))

        return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Segment news text")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()

    segmenter = NewsSegmenter(args.checkpoint)
    segments = segmenter.segment(args.text, args.threshold)

    print(f"\nFound {len(segments)} segments:\n")
    for i, seg in enumerate(segments, 1):
        print(f"--- Segment {i} ---")
        print(seg)
        print()


if __name__ == '__main__':
    main()
