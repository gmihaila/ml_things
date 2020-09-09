#!/usr/bin/python
"""Exampel of using arguments.
Run the file:
python arguments_example.py --splits_folder my/path \
                            --output_dir another/path \
                            --min_length 23 \
                            --avoid_repeat True \
                            --parsing_function detect_speaker
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description='Finetune transformers models on specific dataset.')
    parser.add_argument('--splits_folder', help='path to split files.', type=str)
    parser.add_argument('--output_dir', help='where to save all cleaned tsv files.', type=str)
    parser.add_argument('--min_length', help='number of minimum characters for an utterance.', default=9, type=int)
    parser.add_argument('--avoid_repeat', help='avoid previous response == current context.', default=False, type=bool)
    parser.add_argument('--parsing_function', help="Parsing file function: 'detect_speaker' or 'many_lines'",
                        default='detect_speaker', type=str)

    # parse arguments
    args = parser.parse_args()
    print("Used arguments:")
    [print("--%s %s" % (arg, value)) for arg, value in args.__dict__.items()]

    return


if __name__ == "__main__":
    main()
