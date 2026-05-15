"""
Two-Tower Neural Retrieval Model
=================================
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Conditional TF import (graceful degradation if TF not installed) 
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed. TwoTowerModel will be unavailable.")


# Sub-modules

def _build_tower(
    name:           str,
    vocab_sizes:    Dict[str, int],
    numerical_dim:  int,
    embedding_dim:  int,
    hidden_layers:  List[int],
    dropout_rate:   float,
    l2_reg:         float,
    batch_norm:     bool,
) -> "keras.Model":
    """
    Build a single tower (user or product) as a Keras functional model.

    Inputs:
      - One integer input per categorical feature (for embedding lookup)
      - One float32 vector input for all numerical features

    Returns:
      keras.Model with L2-normalised output embedding.
    """
    inputs      = {}
    embeddings  = []
    reg         = regularizers.l2(l2_reg)

    # ── Categorical embeddings 
    for feat, vocab_size in vocab_sizes.items():
        inp = layers.Input(shape=(1,), name=f"{name}_{feat}_input", dtype="int32")
        inputs[feat] = inp
        emb_dim = min(embedding_dim, max(8, vocab_size // 4))
        emb = layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=emb_dim,
            embeddings_regularizer=reg,
            name=f"{name}_{feat}_embedding",
        )(inp)
        emb = layers.Flatten()(emb)
        embeddings.append(emb)

    # ── Numerical features 
    if numerical_dim > 0:
        num_inp = layers.Input(
            shape=(numerical_dim,), name=f"{name}_numerical_input", dtype="float32"
        )
        inputs["numerical"] = num_inp
        num_norm = layers.LayerNormalization(name=f"{name}_numerical_norm")(num_inp)
        embeddings.append(num_norm)

    # ── Concatenate all inputs 
    if len(embeddings) > 1:
        x = layers.Concatenate(name=f"{name}_concat")(embeddings)
    else:
        x = embeddings[0]

    # ── Feedforward layers 
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(
            units, kernel_regularizer=reg, name=f"{name}_dense_{i}"
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"{name}_bn_{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"{name}_dropout_{i}")(x)

    # ── Final projection + L2 normalisation 
    x = layers.Dense(
        hidden_layers[-1], kernel_regularizer=reg, name=f"{name}_projection"
    )(x)
    output = layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=-1),
        name=f"{name}_l2_norm",
    )(x)

    return keras.Model(inputs=list(inputs.values()), outputs=output, name=f"{name}_tower")


# Two-Tower Model


class TwoTowerModel:
    """
    Production Two-Tower Retrieval Model.

    Wraps the TensorFlow Keras model with:
      - Training with batch softmax loss
      - Negative sampling strategy
      - Embedding export for ANN index building
      - SavedModel serialisation
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: TwoTowerConfig instance from config/model_config.py
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for TwoTowerModel.")
        self.config = config
        self._user_tower    : Optional["keras.Model"] = None
        self._product_tower : Optional["keras.Model"] = None
        self._model         : Optional["keras.Model"] = None
        self._history       = None

    
    # Build

    def build(
        self,
        user_vocab_sizes:    Optional[Dict[str, int]] = None,
        product_vocab_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Instantiate both towers and the full two-tower model.

        Args:
            user_vocab_sizes:    Override vocab sizes (computed from actual data).
            product_vocab_sizes: Override vocab sizes.
        """
        cfg = self.config
        uvocab = user_vocab_sizes    or cfg.user_vocab_sizes
        pvocab = product_vocab_sizes or cfg.product_vocab_sizes

        logger.info("Building User Tower …")
        self._user_tower = _build_tower(
            name="user",
            vocab_sizes=uvocab,
            numerical_dim=len(cfg.user_numerical_features),
            embedding_dim=cfg.user_tower.embedding_dim,
            hidden_layers=cfg.user_tower.hidden_layers,
            dropout_rate=cfg.user_tower.dropout_rate,
            l2_reg=cfg.user_tower.l2_regularization,
            batch_norm=cfg.user_tower.batch_norm,
        )

        logger.info("Building Product Tower …")
        self._product_tower = _build_tower(
            name="product",
            vocab_sizes=pvocab,
            numerical_dim=len(cfg.product_numerical_features),
            embedding_dim=cfg.product_tower.embedding_dim,
            hidden_layers=cfg.product_tower.hidden_layers,
            dropout_rate=cfg.product_tower.dropout_rate,
            l2_reg=cfg.product_tower.l2_regularization,
            batch_norm=cfg.product_tower.batch_norm,
        )

        logger.info("Two-Tower model built. Params: user=%s, product=%s",
                    self._user_tower.count_params(),
                    self._product_tower.count_params())

    
    # Training
    

    def compile(self) -> None:
        """Compile with Adam optimiser and batch softmax loss."""
        if self._user_tower is None:
            raise RuntimeError("Call build() before compile().")

        self._optimizer = keras.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )
        # Batch softmax loss is implemented in the custom training loop below.

    def train(
        self,
        train_dataset: "tf.data.Dataset",
        val_dataset:   "tf.data.Dataset",
        checkpoint_dir: str = "/tmp/two_tower_checkpoints",
    ) -> Dict:
        """
        Custom training loop with batch softmax loss.

        Batch Softmax Loss:
          For each (user, positive_product) pair in the batch,
          treats all other products in the batch as negatives.
          Loss = -log( exp(sim(u,p+) / τ) / Σ exp(sim(u,pj) / τ) )

        Args:
            train_dataset: tf.data.Dataset yielding (user_inputs, product_inputs)
            val_dataset:   tf.data.Dataset for validation
            checkpoint_dir: Directory to save best model checkpoints

        Returns:
            dict with training history metrics
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        cfg = self.config
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_recall_at_10": []}

        for epoch in range(cfg.num_epochs):
            # ── Training pass 
            train_loss = tf.keras.metrics.Mean()
            for batch_user_inputs, batch_product_inputs in train_dataset:
                with tf.GradientTape() as tape:
                    user_embs    = self._user_tower(batch_user_inputs,    training=True)
                    product_embs = self._product_tower(batch_product_inputs, training=True)
                    loss         = self._batch_softmax_loss(user_embs, product_embs)

                grads = tape.gradient(
                    loss,
                    self._user_tower.trainable_variables
                    + self._product_tower.trainable_variables,
                )
                self._optimizer.apply_gradients(zip(
                    grads,
                    self._user_tower.trainable_variables
                    + self._product_tower.trainable_variables,
                ))
                train_loss.update_state(loss)

            # ── Validation pass 
            val_loss    = tf.keras.metrics.Mean()
            recall_at_k = tf.keras.metrics.Mean()
            for batch_user_inputs, batch_product_inputs in val_dataset:
                user_embs    = self._user_tower(batch_user_inputs,    training=False)
                product_embs = self._product_tower(batch_product_inputs, training=False)
                loss         = self._batch_softmax_loss(user_embs, product_embs)
                val_loss.update_state(loss)
                recall       = self._recall_at_k(user_embs, product_embs, k=10)
                recall_at_k.update_state(recall)

            tl = float(train_loss.result())
            vl = float(val_loss.result())
            vr = float(recall_at_k.result())
            history["train_loss"].append(tl)
            history["val_loss"].append(vl)
            history["val_recall_at_10"].append(vr)

            logger.info(
                "Epoch %2d/%d — train_loss: %.4f  val_loss: %.4f  val_recall@10: %.4f",
                epoch + 1, cfg.num_epochs, tl, vl, vr,
            )

            # ── Early stopping + checkpointing 
            if vl < best_val_loss:
                best_val_loss = vl
                patience_counter = 0
                self._save_towers(checkpoint_dir)
                logger.info("  ✓ New best model saved (val_loss=%.4f).", best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                    break

        # Restore best weights
        self._load_towers(checkpoint_dir)
        self._history = history
        return history


    # Inference

    def get_user_embedding(self, user_inputs: Dict) -> np.ndarray:
        """Encode a user into the shared embedding space."""
        return self._user_tower.predict(user_inputs)

    def get_product_embeddings(self, product_inputs: Dict) -> np.ndarray:
        """Encode all products into the shared embedding space (for ANN index)."""
        return self._product_tower.predict(product_inputs)

    def retrieve(
        self,
        user_embedding:     np.ndarray,
        product_embeddings: np.ndarray,
        product_ids:        np.ndarray,
        k:                  int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top-k products for a given user embedding via dot-product ANN.

        In production, replace this brute-force search with:
          - Google Vertex AI Matching Engine, or
          - FAISS IndexFlatIP, or
          - ScaNN

        Returns:
            (top_k_product_ids, top_k_scores)
        """
        # Cosine similarity = dot product (both vectors are L2-normalised)
        scores = product_embeddings @ user_embedding.T   # (n_products,)
        top_k_idx = np.argsort(scores.squeeze())[::-1][:k]
        return product_ids[top_k_idx], scores.squeeze()[top_k_idx]


    # Serialisation

    def save(self, output_dir: str) -> None:
        """Save both towers as TF SavedModels."""
        self._user_tower.save(os.path.join(output_dir, "user_tower"))
        self._product_tower.save(os.path.join(output_dir, "product_tower"))
        logger.info("Two-Tower model saved to %s", output_dir)

    @classmethod
    def load(cls, model_dir: str, config) -> "TwoTowerModel":
        """Load a previously saved Two-Tower model."""
        instance = cls(config)
        instance._user_tower    = keras.models.load_model(
            os.path.join(model_dir, "user_tower")
        )
        instance._product_tower = keras.models.load_model(
            os.path.join(model_dir, "product_tower")
        )
        logger.info("Two-Tower model loaded from %s", model_dir)
        return instance


    # Private helpers


    def _batch_softmax_loss(
        self,
        user_embs:    "tf.Tensor",
        product_embs: "tf.Tensor",
    ) -> "tf.Tensor":
        """
        In-batch softmax loss (also called InfoNCE / NT-Xent).

        Positive pairs: diagonal of the similarity matrix (user_i, product_i).
        Negatives:      all off-diagonal entries in the batch.

        Loss per sample = -log( exp(sim_pos / τ) / Σ_j exp(sim_j / τ) )
        """
        τ = self.config.temperature
        # Similarity matrix: (batch, batch)
        logits = tf.matmul(user_embs, product_embs, transpose_b=True) / τ
        # Labels: identity (each user matches its own product)
        labels = tf.eye(tf.shape(logits)[0])
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        return loss

    @staticmethod
    def _recall_at_k(
        user_embs:    "tf.Tensor",
        product_embs: "tf.Tensor",
        k:            int = 10,
    ) -> "tf.Tensor":
        """Compute in-batch Recall@K: fraction of positives in top-K."""
        logits  = tf.matmul(user_embs, product_embs, transpose_b=True)
        batch   = tf.shape(logits)[0]
        _, top_k_idx = tf.math.top_k(logits, k=k)
        labels  = tf.range(batch)
        hits    = tf.reduce_any(
            tf.equal(top_k_idx, tf.expand_dims(labels, axis=1)), axis=1
        )
        return tf.reduce_mean(tf.cast(hits, tf.float32))

    def _save_towers(self, directory: str) -> None:
        self._user_tower.save_weights(os.path.join(directory, "user_tower_best.h5"))
        self._product_tower.save_weights(os.path.join(directory, "product_tower_best.h5"))

    def _load_towers(self, directory: str) -> None:
        self._user_tower.load_weights(os.path.join(directory, "user_tower_best.h5"))
        self._product_tower.load_weights(os.path.join(directory, "product_tower_best.h5"))