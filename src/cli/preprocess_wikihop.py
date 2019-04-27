import argparse
import json
import os

import torch

from ..models import BERT
from ..preprocessor import Preprocessor
from ..utils.dataset_utils import load_wikihop
from ..utils.generic_utils import make_dir


def main(input_directory, output_directory):
    """Saves a preprocessed Wiki- or MedHop dataset to disk.

    Creates a json file for each partition in the the given Wiki- or MedHop dataset.
    (`input_directory`) which contains everything we need for graph construction along with a
    serialized torch Tensor containing mention encodings. The json files are saved to
    `output_directory/<partition>.json` and the serialized tensor to
    `output_directory/embeddings.pt` (note that this must me loaded with `torch.load()`).

    Args:
        input_directory (str): Path to the Wiki- or MedHop dataset.
        output_directory (str): Path to save the processed output for the Wiki- or MedHop dataset.

    Returns:
        Two-tuple containing the `processed_dataset`, a dictionary containing everything we need
        from the Wiki- or MedHop dataset at `input_directory` for graph construction, and
        `embeddings`, a .pt file containing a serialized tensor of encoded mentions.
    """
    dataset = load_wikihop(input_directory)
    preprocessor = Preprocessor()
    model = BERT()

    # Process the dataset, extracting what we need for graph construction
    (processed_dataset, encoded_mentions, encoded_mentions_split_sizes, targets,
     targets_split_sizes) = preprocessor.transform(dataset, model)

    # Make output directory if it does not exist
    make_dir(output_directory)

    for partition in processed_dataset:
        # Make a directory for each partition
        partition_directory = os.path.join(output_directory, partition)
        make_dir(partition_directory)

        # Create .json filepaths
        processed_dataset_filepath = \
            os.path.join(partition_directory, "processed_dataset.json")
        encoded_mentions_split_sizes_filepath = os.path.join(partition_directory,
                                                             'encoded_mentions_split_sizes.json')
        targets_split_sizes_filepath = os.path.join(partition_directory, 'targets_split_sizes.json')
        # Create PyTorch .pt filepaths
        encoded_mentions_filepath = os.path.join(partition_directory, 'encoded_mentions.pt')
        targets_filepath = os.path.join(partition_directory, 'targets.pt')

        # Write .json files to disk
        with open(processed_dataset_filepath, 'w') as f:
            json.dump(processed_dataset[partition], f, indent=2)
        with open(encoded_mentions_split_sizes_filepath, 'w') as f:
            json.dump(encoded_mentions_split_sizes[partition], f, indent=2)
        with open(targets_split_sizes_filepath, 'w') as f:
            json.dump(targets_split_sizes[partition], f, indent=2)

        # Write .pt files to disk
        torch.save(encoded_mentions[partition], encoded_mentions_filepath)
        torch.save(targets[partition], targets_filepath)

    return processed_dataset, encoded_mentions, encoded_mentions_split_sizes


if __name__ == '__main__':
    description = '''Creates a set of files for each partition in the the given Wiki- or MedHop
    dataset at `input_directory` which contain everything we need for graph construction and
    training.
    '''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the Wiki- or MedHop dataset.')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to save the processed output for the Wiki- or MedHop dataset.')

    args = parser.parse_args()

    main(args.input, args.output)
