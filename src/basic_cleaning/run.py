#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    # remove rows from wrong geolocation
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the clean dataframe to csv
    df.to_csv(args.output_artifact, index=False)

    # Upload the newly created artifact to W&B
    artifact = wandb.Artifact(
        name=args.output_artifact,
        description=args.output_description,
        type=args.output_type,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="the name of the input artifact, which will be cleaned in this step",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="the name of the resulting output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact. i.e. 'model', or 'dataset', etc.",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description of this component's resulting output artifact.",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price to keep. Every entry with a price below will be considered to be an outlier and is removed.",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price, every entry with a price above will be considered to be an outlier and is removed",
        required=True,
    )

    args = parser.parse_args()

    go(args)
