from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from pathlib import Path


class NYUFrankaPlayDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Right camera RGB observation.",
                                    ),
                                    "image_additional_view": tfds.features.Image(
                                        shape=(128, 128, 3),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Left camera RGB observation.",
                                    ),
                                    "depth": tfds.features.Tensor(
                                        shape=(128, 128, 1),
                                        dtype=np.int32,
                                        doc="Right camera depth observation.",
                                    ),
                                    "depth_additional_view": tfds.features.Tensor(
                                        shape=(128, 128, 1),
                                        dtype=np.int32,
                                        doc="Left camera depth observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(13,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [7x robot joint angles, 3x EE xyz, 3x EE rpy.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(15,),
                                dtype=np.float32,
                                doc="Robot action, consists of [7x joint velocities, "
                                "3x EE delta xyz, 3x EE delta rpy, 1x gripper position, 1x terminate episode].",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            "language_embedding": tfds.features.Tensor(
                                shape=(512,),
                                dtype=np.float32,
                                doc="Kona language embedding. "
                                "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            "train": self._generate_examples(path="data/train/**"),
            "val": self._generate_examples(path="data/val/**"),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # episode path is now a directory containing a bunch of subdirectories
            # each subdirectory is an episode
            # containing obses_0.pt, actions_0.pt
            # obses_0.pt: {
            #   "cam_0": (RGB Tx3x128x128 uint8, D Tx1x128x128 int32),
            #   "cam_1": (RGB Tx3x128x128 uint8, D Tx1x128x128 int32),
            #   "robot_state.pkl": concat(theta (7), xyz (3), rpy (3))
            # }
            # actions_0.pt: {
            #   "robot_state.pkl": concat(dtheta (7), dxyz (3), drpy (3))
            #   "gripper_state.pkl": target (-1 close, 1 open)
            # }
            # we need to assemble this into a list of dicts

            obses_path = Path(episode_path) / "obses_0.npy"
            actions_path = Path(episode_path) / "actions_0.npy"
            obses = np.load(obses_path, allow_pickle=True).item()
            actions = np.load(actions_path, allow_pickle=True).item()

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            episode_length = obses["cam_0"][0].shape[0]

            # concatenate actions: [dtheta (7), dxyz (3), drpy (3), gripper (1), terminate (1)]
            terminate_array = np.zeros((episode_length, 1))
            terminate_array[-1] = 1
            actions_flat = np.concatenate(
                [
                    actions["robot_state.pkl"][0],
                    actions["gripper_state.pkl"][0],
                    terminate_array,
                ],
                axis=1,
            ).astype(np.float32)

            for i in range(episode_length):
                # compute Kona language embedding
                language_embedding = self._embed(["play with the kitchen"])[0].numpy()

                episode.append(
                    {
                        "observation": {
                            "image": obses["cam_0"][0][i],
                            "image_additional_view": obses["cam_1"][0][i],
                            "depth": obses["cam_0"][1][i],
                            "depth_additional_view": obses["cam_1"][1][i],
                            "state": obses["robot_state.pkl"][0][i],
                        },
                        "action": actions_flat[i],
                        "discount": 1.0,
                        "reward": float(i == (episode_length - 1)),
                        "is_first": i == 0,
                        "is_last": i == (episode_length - 1),
                        "is_terminal": i == (episode_length - 1),
                        "language_instruction": "play with the kitchen",
                        "language_embedding": language_embedding,
                    }
                )

            # create output data sample
            sample = {"steps": episode, "episode_metadata": {"file_path": episode_path}}

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )
