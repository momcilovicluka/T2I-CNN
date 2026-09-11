"""
Overlap diagnostics for T2I image quality assessment.

When continuous feature coordinates are mapped to a discrete pixel grid,
multiple features can collide at the same pixel location. This causes
lossy compression — overlapping feature values must be averaged, losing
informational fidelity.

Two diagnostic metrics quantify this overlap (from DeepInsight FDM analysis):

OF (Percentage of Overlapped Features):
    OF = (number of features at occupied pixels / total features) × 100
    Higher OF means more features are sharing pixels.

OP (Percentage of Overlapped Pixels):
    OP = (pixels with >1 feature / total active pixels) × 100
    Higher OP means more pixels have collision.

Reference: Sharma et al. (2019), DeepInsight — Feature Density Matrix (FDM)
"""

import numpy as np


def compute_overlap(coordinates, image_size=None):
    """Compute overlap diagnostics from a feature-to-pixel coordinate map.

    Args:
        coordinates: dict or array mapping feature_index -> (x, y)
            - dict: {feature_idx: (x, y), ...}
            - array: shape (n_features, 2) where row i = (x_i, y_i)
        image_size: optional, for reporting active pixel ratio

    Returns:
        dict with keys:
            'n_features': total number of features
            'n_active_pixels': number of pixels with >=1 feature
            'n_overlapped_features': features at pixels with >1 feature
            'n_overlapped_pixels': pixels with >1 feature
            'OF': percentage of overlapped features (0-100)
            'OP': percentage of overlapped pixels (0-100)
    """
    # Convert to dict if array
    if isinstance(coordinates, np.ndarray):
        coords = {i: tuple(coordinates[i]) for i in range(len(coordinates))}
    else:
        coords = coordinates

    n_features = len(coords)

    # Build pixel -> list of feature indices mapping
    pixel_to_features = {}
    for feat_idx, (x, y) in coords.items():
        pixel = (int(round(x)), int(round(y)))
        if pixel not in pixel_to_features:
            pixel_to_features[pixel] = []
        pixel_to_features[pixel].append(feat_idx)

    # Count active pixels and overlapped pixels
    n_active_pixels = len(pixel_to_features)
    n_overlapped_pixels = 0
    n_overlapped_features = 0

    for pixel, feature_list in pixel_to_features.items():
        if len(feature_list) > 1:
            n_overlapped_pixels += 1
            n_overlapped_features += len(feature_list)

    # Compute percentages
    OF = (n_overlapped_features / n_features * 100) if n_features > 0 else 0.0
    OP = (n_overlapped_pixels / n_active_pixels * 100) if n_active_pixels > 0 else 0.0

    return {
        'n_features': n_features,
        'n_active_pixels': n_active_pixels,
        'n_overlapped_features': n_overlapped_features,
        'n_overlapped_pixels': n_overlapped_pixels,
        'OF': OF,
        'OP': OP,
    }


def compute_overlap_all_methods(X_train, y_train, image_size=32):
    """Compute overlap diagnostics for all four T2I methods.

    FIX (audit C5): IGTD's OF/OP are now MEASURED from its fitted coordinate
    map instead of hard-coded to 0 (naive remains structurally 0 and is
    annotated as such). DeepInsight and TINTO can have coordinate collisions.
    s_igtd was dropped from the study 2026-09-03 (duplicated igtd; see
    paper-statement-guide PART 13i).

    Returns:
        dict mapping method_name -> overlap_metrics_dict
    """
    results = {}

    # Naive: collisions are undefined for a padded one-feature-per-cell grid
    # that is then resized. Every feature has its own cell before the resize,
    # so OF=OP=0 is structural, not measured — annotate that instead of
    # emitting bare zeros (audit C5).
    results['naive'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0,
                        'n_features': X_train.shape[1],
                        'n_active_pixels': X_train.shape[1],
                        'n_overlapped_features': 0,
                        'n_overlapped_pixels': 0,
                        'note': 'pad+reshape: one feature per grid cell before resize'}

    # IGTD: MEASURED from the fitted coordinate map (audit C5), not asserted.
    # TINTOlib's IGTD assigns the D features to unique cells (a one-row strip
    # when D <= image_size), so OF/OP are expected to be 0 — but that must come
    # from the coordinates, because a library/version change could alter it.
    try:
        from .igtd import IGTD
        igtd = IGTD(image_size=image_size)
        igtd.fit(X_train, y_train)
        coords = igtd.get_coordinates()
        if coords is not None:
            base = compute_overlap(coords, image_size)
            results['igtd'] = {**base, 'of_percent': base['OF'], 'op_percent': base['OP']}
        else:
            results['igtd'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0,
                               'error': 'no coordinates'}
    except Exception as e:
        results['igtd'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0,
                           'error': str(e)}

    # DeepInsight: can have overlaps
    try:
        from .deepinsight import DeepInsight
        di = DeepInsight(image_size=image_size)
        di.fit(X_train, y_train)
        coords = di.get_coordinates()
        if coords is not None:
            base = compute_overlap(coords, image_size)
            results['deepinsight'] = {**base, 'of_percent': base['OF'], 'op_percent': base['OP']}
        else:
            results['deepinsight'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0, 'error': 'no coordinates'}
    except Exception as e:
        results['deepinsight'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0, 'error': str(e)}

    # TINTO: can have overlaps (uses PCA/t-SNE like DeepInsight)
    try:
        from .tinto import TINTO
        tinto = TINTO(image_size=image_size)
        tinto.fit(X_train, y_train)
        coords = tinto.get_coordinates()
        if coords is not None:
            base = compute_overlap(coords, image_size)
            results['tinto'] = {**base, 'of_percent': base['OF'], 'op_percent': base['OP']}
        else:
            results['tinto'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0, 'error': 'no coordinates'}
    except Exception as e:
        results['tinto'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0, 'error': str(e)}

    return results
