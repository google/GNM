# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sampler for generating expressions and identities from a CVAE.

Usage:
  ```

    expression_sampler = ExpressionSampler(save_dir_decoder)
    generated_expressions = expression_sampler.sample_expression(
        class_label, num_samples)
    blended_expression = expression_sampler.blend_expressions(class_weights)

    identity_sampler = IdentitySampler(save_dir_decoder)
    generated_identities = identity_sampler.sample_identity(gender_class,
        ethnicity_class, num_samples)
    blended_identity = identity_sampler.blend_identities(gender_weights,
        ethnicity_weights, num_samples)

    generated_3D_points = gnm_np(identity=generated_identities,
        expression=generated_expressions)
    blended_3D_points = gnm_np(identity=blended_identity,
        expression=blended_expression)
    ```
"""

from collections.abc import Mapping
import enum
from etils import epath
import numpy as np
import tensorflow as tf

_pkg = __package__ or 'gnm.shape'
_DATA_DIR = epath.resource_path(_pkg) / 'data'
_EXPRESSION_DECODER_PATH = (
    _DATA_DIR / 'semantic_sampler' / 'expression_decoder_model.h5'
)
_IDENTITY_DECODER_PATH = (
    _DATA_DIR / 'semantic_sampler' / 'identity_decoder_model.h5'
)


class Gender(enum.IntEnum):
  """Gender classes for identity sampling."""

  FEMALE = 0
  MALE = 1


class Ethnicity(enum.IntEnum):
  """Ethnicity classes for identity sampling."""

  MIDDLE_EASTERN = 0
  ASIAN = 1
  WHITE = 2
  BLACK = 3


class Expression(enum.IntEnum):
  """Expression classes for expression sampling."""

  SURPRISE = 0
  DISGUST = 1
  SUCK = 2
  COMPRESS_FACE = 3
  STRETCH_FACE = 4
  HAPPY = 5
  SQUINT = 6
  PLATYSMA = 7
  BLOW = 8
  FUNNELER = 9
  SMILE_WIDE = 10
  CORNERS_DOWN = 11
  PUCKER = 12
  WINK_LEFT = 13
  WINK_RIGHT = 14
  MOUTH_LEFT = 15
  MOUTH_RIGHT = 16
  LIPS_ROLL_IN = 17
  SNARL = 18
  TONGUE_CENTER = 19


def _get_rng(rng: np.random.Generator | None) -> np.random.Generator:
  """Returns the provided RNG or a new default RNG if None."""
  return rng if rng is not None else np.random.default_rng()


class ExpressionSampler:
  """Samples expressions and identities from a Conditional VAE."""

  def __init__(
      self,
      decoder_model_path: str | None = None,
      verbose: bool = False,
  ) -> None:
    """Initializes the ExpressionSampler class.

    Loads the decoder model and stores parameters.

    Args:
      decoder_model_path: Path to the saved decoder model file. If None, the
        default model is loaded.
      verbose: If True, prints a summary of the loaded decoder model.
    """
    if decoder_model_path is None:
      decoder_model_path = (
          _EXPRESSION_DECODER_PATH  # pyrefly: ignore[bad-assignment]
      )
    self._decoder = tf.keras.models.load_model(str(decoder_model_path))
    self._expression_names = tuple(member.name.lower() for member in Expression)
    self._num_classes = self._decoder.inputs[1].shape[-1]
    self._latent_dim = self._decoder.inputs[0].shape[-1]
    if verbose:
      print('ExpressionSampler initialized and decoder model loaded.')
      self._decoder.summary()

  @property
  def expression_names(self) -> tuple[str, ...]:
    return tuple(self._expression_names)

  @property
  def expression_label_mapping(self) -> Mapping[int, str]:
    """Returns mapping of class index to expression name."""
    return {i: name for i, name in enumerate(self._expression_names)}

  def sample_expression(
      self,
      class_label: Expression,
      num_samples: int = 1,
      rng: np.random.Generator | None = None,
      verbose: bool = False,
  ) -> np.ndarray:
    """Generates expression vectors for a specific class.

    Args:
      class_label: The expression class to sample from.
      num_samples: The number of expression vectors to generate.
      rng: A numpy random number generator for reproducibility. If provided with
        a seed, it will be used to sample from the latent space, making the
        generated expressions deterministic. If None, a default RNG will be
        used, resulting in non-deterministic behavior.
      verbose: If True, prints a message about the generated samples.

    Returns:
      Array of generated expression vectors of shape (num_samples,
      expression_dim).
    """
    class_one_hot = tf.keras.utils.to_categorical(
        [class_label], num_classes=self._num_classes
    )
    class_one_hot = np.repeat(class_one_hot, num_samples, axis=0).astype(
        'float32'
    )

    rng = _get_rng(rng)
    z_sample = rng.normal(size=(num_samples, self._latent_dim)).astype(
        'float32'
    )

    generated_vectors = self._decoder.predict([z_sample, class_one_hot])
    class_name = self._expression_names[class_label]
    if verbose:
      print(
          f'Generated {num_samples} expression vectors for class index '
          f'{class_label} ({class_name}).'
      )
    return generated_vectors

  def blend_expressions(
      self,
      class_weights: Mapping[Expression, float],
      rng: np.random.Generator | None = None,
      verbose: bool = False,
  ) -> np.ndarray:
    """Blends multiple expressions based on provided class indices and weights.

    Args:
        class_weights: A mapping where keys are Expression enum members, and
          values are their corresponding weights (float). Weights do not need to
          sum to 1, but will be normalized.
        rng: A numpy random number generator for reproducibility. If provided
          with a seed, it will be used to sample from the latent space, making
          the generated expressions deterministic. If None, a default RNG will
          be used, resulting in non-deterministic behavior.
      verbose: If True, prints a message about the generated samples.

    Returns:
        A single blended expression vector.
    """
    if not class_weights:
      raise ValueError('class_weights mapping cannot be empty.')

    if any(weight < 0 for weight in class_weights.values()):
      raise ValueError('Weights cannot be negative.')

    # Normalize weights
    total_weight = sum(class_weights.values())
    if np.isclose(total_weight, 0):
      raise ValueError('Sum of class_weights cannot be 0.')
    normalized_weights = {
        index: weight / total_weight for index, weight in class_weights.items()
    }

    blended_latent_vector = np.zeros((1, self._latent_dim), dtype='float32')
    blended_one_hot_label = np.zeros((1, self._num_classes), dtype='float32')

    rng = _get_rng(rng)
    for class_index, weight in normalized_weights.items():

      # Sample a single latent vector for each class
      z_sample = rng.normal(size=(1, self._latent_dim)).astype('float32')
      class_one_hot = tf.keras.utils.to_categorical(
          [class_index], num_classes=self._num_classes
      ).astype('float32')

      # Accumulate weighted latent vectors and one-hot labels
      blended_latent_vector += z_sample * weight
      blended_one_hot_label += class_one_hot * weight

    # Decode the blended latent vector and weighted average of one-hot labels
    blended_expression = self._decoder.predict(
        [blended_latent_vector, blended_one_hot_label]
    )
    if verbose:
      print(
          'Generated a blended expression vector from class weights:'
          f' {class_weights}'
      )

    return blended_expression.flatten()

  def randomize_expressions(
      self,
      num_samples: int = 1,
      max_num_categories: int = 3,
      rng: np.random.Generator | None = None,
  ) -> np.ndarray:
    """Generates random blended expressions.

    This is done by mixing a random subset of classes.

    Args:
        num_samples: Number of vectors to generate.
        max_num_categories: Maximum number of distinct expressions to blend per
          sample.
        rng: A numpy random number generator for reproducibility. If provided
          with a seed, it will be used to sample from the latent space, making
          the generated expressions deterministic. If None, a default RNG will
          be used, resulting in non-deterministic behavior.

    Returns:
        Array of generated expression vectors.
    """
    generated_vectors = []
    rng = _get_rng(rng)
    for _ in range(num_samples):
      # Pick 2 to 'max_num_categories' number of classes to blend
      num_classes_to_blend = rng.integers(
          2, min(max_num_categories, self._num_classes) + 1
      )

      # Pick unique class indices
      chosen_classes = rng.choice(
          self._num_classes, size=num_classes_to_blend, replace=False
      )

      # Assign random weights
      weights = {Expression(index): rng.random() for index in chosen_classes}

      # Blend
      vec = self.blend_expressions(weights, rng=rng)
      generated_vectors.append(vec)

    return np.array(generated_vectors)


def _create_combined_one_hot_labels(
    raw_labels: np.ndarray,
    num_gender_classes: int,
    num_ethnicities_classes: int,
) -> np.ndarray:
  """Converts raw [Gender, Ethnicity] labels to a combined one-hot vector.

  Args:
      raw_labels: Expected shape: (N, 2), where raw_labels[:, 0] is gender (0 or
        1) and raw_labels[:, 1] is ethnicity (0-indexed: 0, 1, 2, 3).
      num_gender_classes: The number of gender classes (e.g., 2).
      num_ethnicities_classes: The number of ethnicity classes (e.g., 4).

  Returns:
      A float32 numpy array of shape (N, num_gender_classes +
      num_ethnicities_classes) containing the concatenated one-hot encoded
      labels.
  """
  # Gender OHE (2 dimensions)
  gender_indices = raw_labels[:, 0].astype(int)
  gender_ohe = tf.keras.utils.to_categorical(
      gender_indices, num_classes=num_gender_classes
  )

  # Ethnicities OHE (4 dimensions).
  ethn_indices = raw_labels[:, 1].astype(int)
  ethn_ohe = tf.keras.utils.to_categorical(
      ethn_indices, num_classes=num_ethnicities_classes
  )

  # Concatenate
  combined_ohe = np.concatenate([gender_ohe, ethn_ohe], axis=1)
  return combined_ohe.astype('float32')


class IdentitySampler:
  """Samples identities from a Conditional VAE."""

  # These dimensions were determined from the notebook state
  _LATENT_DIM = 64
  _NUM_GENDER_CLASSES = 2
  _NUM_ETHNICITIES_CLASSES = 4
  _GENDER_LABEL_MAP = {Gender.FEMALE: 'Female', Gender.MALE: 'Male'}
  _ETHNICITY_LABEL_MAP = {
      Ethnicity.MIDDLE_EASTERN: 'Middle Eastern',
      Ethnicity.ASIAN: 'Asian',
      Ethnicity.WHITE: 'White',
      Ethnicity.BLACK: 'Black',
  }

  def __init__(
      self,
      decoder_model_path: str | None = None,
      verbose: bool = False,
  ) -> None:
    """Initializes the IdentitySampler class.

    Loads the decoder model and stores parameters.

    Args:
      decoder_model_path: Path to the saved identity decoder model file. If
        None, the default model is loaded.
      verbose: If True, prints a summary of the loaded decoder model.
    """
    if decoder_model_path is None:
      decoder_model_path = (
          _IDENTITY_DECODER_PATH  # pyrefly: ignore[bad-assignment]
      )
    self._decoder = tf.keras.models.load_model(decoder_model_path)

    self._condition_dim = self._decoder.inputs[1].shape[-1]
    if verbose:
      print('IdentitySampler initialized and decoder model loaded.')
      self._decoder.summary()

  def sample_identity(
      self,
      gender_class: Gender,
      ethnicity_class: Ethnicity,
      num_samples: int = 1,
      rng: np.random.Generator | None = None,
      verbose: bool = False,
  ) -> np.ndarray:
    """Generates identity vectors for a specific Gender and Ethnicity.

    Args:
      gender_class: The gender class to sample from.
      ethnicity_class: The ethnicity class to sample from.
      num_samples: The number of identity vectors to generate.
      rng: A numpy random number generator for reproducibility. If provided with
        a seed, it will be used to sample from the latent space, making the
        generated identities deterministic. If None, a default RNG will be used,
        resulting in non-deterministic behavior.
      verbose: If True, prints a message about the generated samples.

    Returns:
      Array of generated identity vectors of shape (num_samples, identity_dim).
    """
    raw_label_combo = np.array([[gender_class, ethnicity_class]])
    combined_ohe_label = _create_combined_one_hot_labels(
        raw_label_combo,
        self._NUM_GENDER_CLASSES,
        self._NUM_ETHNICITIES_CLASSES,
    )
    labels_for_decoder = np.repeat(combined_ohe_label, num_samples, axis=0)

    rng = _get_rng(rng)
    z_sample = rng.normal(size=(num_samples, self._LATENT_DIM)).astype(
        'float32'
    )

    generated_vectors = self._decoder.predict(
        [z_sample, labels_for_decoder], verbose=0
    )
    gender_name = self._GENDER_LABEL_MAP.get(
        gender_class, f'Unknown Gender {gender_class}'
    )
    ethnicity_name = self._ETHNICITY_LABEL_MAP.get(
        ethnicity_class, f'Unknown Ethnicity {ethnicity_class}'
    )
    if verbose:
      print(
          f'Generated {num_samples} identity vectors for Gender: {gender_name},'
          f' Ethnicity: {ethnicity_name}.'
      )

    return generated_vectors

  def blend_identities(
      self,
      gender_weights: Mapping[Gender, float],
      ethnicity_weights: Mapping[Ethnicity, float],
      num_samples: int = 1,
      rng: np.random.Generator | None = None,
      verbose: bool = False,
  ) -> np.ndarray:
    """Blends multiple identities based on provided class weights.

    This method blends identities by using weighted averages of gender and
    ethnicity classes.

    Args:
      gender_weights: A mapping where keys are Gender enum members, and values
        are their corresponding weights (float). Weights do not need to sum to
        1, but will be normalized.
      ethnicity_weights: A mapping where keys are Ethnicity enum members, and
        values are their corresponding weights (float). Weights do not need to
        sum to 1, but will be normalized.
      num_samples: The number of blended identity vectors to generate.
      rng: A numpy random number generator for reproducibility. If provided with
        a seed, it will be used to sample from the latent space, making the
        generated identities deterministic. If None, a default RNG will be used,
        resulting in non-deterministic behavior.
      verbose: If True, prints a message about the generated samples.

    Returns:
      Array of generated blended identity vectors of shape (num_samples,
      identity_dim).
    """
    if not gender_weights or not ethnicity_weights:
      raise ValueError('Gender and ethnicity weights mappings cannot be empty.')

    if any(weight < 0 for weight in gender_weights.values()):
      raise ValueError('Gender weights cannot be negative.')

    if any(weight < 0 for weight in ethnicity_weights.values()):
      raise ValueError('Ethnicity weights cannot be negative.')

    # Normalize gender weights
    total_gender_weight = sum(gender_weights.values())
    if np.isclose(total_gender_weight, 0):
      raise ValueError('Sum of gender_weights cannot be 0.')
    normalized_gender_weights = {
        index: weight / total_gender_weight
        for index, weight in gender_weights.items()
    }

    # Normalize ethnicity weights
    total_ethnicity_weight = sum(ethnicity_weights.values())
    if np.isclose(total_ethnicity_weight, 0):
      raise ValueError('Sum of ethnicity_weights cannot be 0.')
    normalized_ethnicity_weights = {
        index: weight / total_ethnicity_weight
        for index, weight in ethnicity_weights.items()
    }

    blended_gender_ohe = np.zeros(
        (1, self._NUM_GENDER_CLASSES), dtype='float32'
    )
    for class_index, weight in normalized_gender_weights.items():
      gender_one_hot = tf.keras.utils.to_categorical(
          [class_index], num_classes=self._NUM_GENDER_CLASSES
      ).astype('float32')
      blended_gender_ohe += gender_one_hot * weight

    blended_ethnicity_ohe = np.zeros(
        (1, self._NUM_ETHNICITIES_CLASSES), dtype='float32'
    )
    for class_index, weight in normalized_ethnicity_weights.items():
      ethnicity_one_hot = tf.keras.utils.to_categorical(
          [class_index], num_classes=self._NUM_ETHNICITIES_CLASSES
      ).astype('float32')
      blended_ethnicity_ohe += ethnicity_one_hot * weight

    blended_combined_ohe = np.concatenate(
        [blended_gender_ohe, blended_ethnicity_ohe], axis=1
    )
    labels_for_decoder = np.repeat(blended_combined_ohe, num_samples, axis=0)

    rng = _get_rng(rng)
    z_sample = rng.normal(size=(num_samples, self._LATENT_DIM)).astype(
        'float32'
    )

    blended_identities = self._decoder.predict(
        [z_sample, labels_for_decoder], verbose=0
    )
    if verbose:
      print(
          f'Generated {num_samples} blended identity vectors from gender'
          f' weights: {gender_weights} and ethnicity weights:'
          f' {ethnicity_weights}.'
      )

    return blended_identities

  def randomize_identities(
      self, num_samples: int = 1, rng: np.random.Generator | None = None
  ) -> np.ndarray:
    """Generates random blended identities.

    This is done by assigning random weights to gender and ethnicity classes.

    Args:
        num_samples: The number of blended identity vectors to generate.
        rng: A numpy random number generator for reproducibility. If provided
          with a seed, it will be used to sample from the latent space, making
          the generated identities deterministic. If None, a default RNG will be
          used, resulting in non-deterministic behavior.

    Returns:
        An array of generated blended identity vectors.
    """
    generated_blended_identities = []

    rng = _get_rng(rng)
    for _ in range(num_samples):
      # Generate random gender weights
      gender_weights = {
          Gender.FEMALE: rng.random(),
          Gender.MALE: rng.random(),
      }

      # Generate random ethnicity weights
      ethnicity_weights = {
          Ethnicity.MIDDLE_EASTERN: rng.random(),
          Ethnicity.ASIAN: rng.random(),
          Ethnicity.WHITE: rng.random(),
          Ethnicity.BLACK: rng.random(),
      }

      blended_identity = self.blend_identities(
          gender_weights, ethnicity_weights, num_samples=1, rng=rng
      )
      generated_blended_identities.append(blended_identity[0])

    return np.array(generated_blended_identities)

  def explain_classes(self) -> Mapping[str, Mapping[int, str]]:
    """Returns a dictionary mapping class indices to identity attributes.

    The dictionary contains mappings for 'gender' and 'ethnicity'.
    """
    return {  # pyrefly: ignore[bad-return]
        'gender': self._GENDER_LABEL_MAP,
        'ethnicity': self._ETHNICITY_LABEL_MAP,
    }
