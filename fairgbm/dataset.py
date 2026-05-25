# coding: utf-8
"""
Constraint group data handling for FairGBM.

Provides a FairGBMDataset wrapper that manages constraint group metadata
alongside a standard LightGBM Dataset.
"""

import numpy as np

import lightgbm


class FairGBMDataset:
    """Wrapper around lightgbm.Dataset that manages constraint group data.

    Parameters
    ----------
    data : object
        Data source for LightGBM Dataset (same as lightgbm.Dataset).
    label : array-like or None
        Labels for training data.
    constraint_group : numpy.ndarray or None
        Integer array identifying sensitive group membership per instance.
        Must be non-negative int32 with same length as training data.
        If None, all instances are treated as belonging to group 0.
    **kwargs
        Additional keyword arguments passed to lightgbm.Dataset.
    """

    def __init__(self, data, label=None, constraint_group=None, **kwargs):
        self._dataset = lightgbm.Dataset(data, label=label, **kwargs)
        self._constraint_group = None

        if constraint_group is not None:
            self.set_constraint_group(constraint_group)

    def set_constraint_group(self, group):
        """Set constraint group data.

        Parameters
        ----------
        group : array-like
            Integer array of constraint group membership.
            Must be non-negative integers with same length as training data.

        Raises
        ------
        ValueError
            If group has wrong dtype, contains negative values, or has
            wrong length relative to training data.
        """
        group = np.asarray(group)

        # Validate integer dtype
        if not np.issubdtype(group.dtype, np.integer):
            raise ValueError(
                f"constraint_group must be integer-typed, got dtype={group.dtype}"
            )

        # Validate non-negative
        if group.size > 0 and np.any(group < 0):
            raise ValueError("constraint_group must contain non-negative values")

        # Validate length matches data if label is available
        label = self._get_label_safe()
        if label is not None and len(group) != len(label):
            raise ValueError(
                f"constraint_group length ({len(group)}) does not match "
                f"training data length ({len(label)})"
            )

        self._constraint_group = np.ascontiguousarray(group, dtype=np.int32)

    def get_constraint_group(self):
        """Get constraint group data.

        Returns
        -------
        numpy.ndarray or None
            The constraint group array (int32), or None if not set.
        """
        return self._constraint_group

    @property
    def dataset(self):
        """Access the underlying lightgbm.Dataset."""
        return self._dataset

    def _get_label_safe(self):
        """Try to get labels from the underlying dataset.

        Returns None if labels are not yet available (dataset not constructed).
        """
        try:
            return self._dataset.get_label()
        except Exception:
            return None

    # --- Delegate common Dataset methods ---

    def get_label(self):
        """Get labels from the underlying dataset."""
        return self._dataset.get_label()

    def get_weight(self):
        """Get weights from the underlying dataset."""
        return self._dataset.get_weight()

    def construct(self):
        """Construct the underlying LightGBM dataset."""
        self._dataset.construct()
        return self
