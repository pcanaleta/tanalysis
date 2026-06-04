import numpy as np
from collections import namedtuple

TrackEntry = namedtuple("TrackEntry", ["frame", "label", "centroid"])


def compute_centroids(label_volume):
    """Compute centroids for each label in a 3D label volume.

    Args:
        label_volume (numpy.ndarray): 3D integer array with labeled objects.

    Returns:
        dict: Mapping from label id to centroid as a tuple (z, y, x).
    """
    labels = np.unique(label_volume)
    labels = labels[labels != 0]
    centroids = {}
    for label in labels:
        positions = np.argwhere(label_volume == label)
        if positions.size:
            centroids[int(label)] = tuple(positions.mean(axis=0).tolist())
    return centroids


def match_labels(prev_centroids, curr_centroids, max_distance=10.0):
    """Match labels between two frames using nearest centroids.

    Args:
        prev_centroids (dict): label -> centroid from previous frame.
        curr_centroids (dict): label -> centroid from current frame.
        max_distance (float): maximum allowed centroid distance for a match.

    Returns:
        dict: Mapping from previous frame label to current frame label.
    """
    if not prev_centroids or not curr_centroids:
        return {}

    pairs = []
    for prev_label, prev_centroid in prev_centroids.items():
        for curr_label, curr_centroid in curr_centroids.items():
            dist = np.linalg.norm(np.array(prev_centroid) - np.array(curr_centroid))
            pairs.append((dist, prev_label, curr_label))

    pairs.sort(key=lambda x: x[0])
    matched_prev = set()
    matched_curr = set()
    mapping = {}
    for dist, prev_label, curr_label in pairs:
        if dist > max_distance:
            break
        if prev_label in matched_prev or curr_label in matched_curr:
            continue
        mapping[int(prev_label)] = int(curr_label)
        matched_prev.add(prev_label)
        matched_curr.add(curr_label)
    return mapping


def track_labeled_video(label_sequence, max_distance=10.0, min_length=2):
    """Track labeled objects over a 3D labeled video sequence.

    Args:
        label_sequence (numpy.ndarray): list of 3D label volumes.
        max_distance (float): maximum centroid distance to link objects between frames.
        min_length (int): minimum number of frames for a track to be returned.

    Returns:
        list: List of track dictionaries with keys 'track_id' and 'entries'.
    """
    tracks = []
    active_tracks = []
    next_track_id = 1

    # Normalize input: accept a list/sequence of 3D label volumes or a 4D ndarray (T, Z, Y, X)
    if isinstance(label_sequence, np.ndarray):
        if label_sequence.ndim == 4:
            frames = list(label_sequence)
        elif label_sequence.ndim == 3:
            frames = [label_sequence]
        else:
            raise ValueError("label_sequence must be a 3D volume or a 4D time series")
    else:
        # assume it's an iterable of 3D volumes
        frames = list(label_sequence)

    if len(frames) == 0:
        return []

    # Initialize tracks from first frame
    first_centroids = compute_centroids(frames[0])
    for label, centroid in first_centroids.items():
        track = {
            "track_id": next_track_id,
            "last_label": label,
            "last_centroid": centroid,
            "entries": [TrackEntry(frame=0, label=label, centroid=centroid)],
        }
        active_tracks.append(track)
        next_track_id += 1

    prev_centroids = first_centroids

    for frame_index, volume in enumerate(frames[1:], start=1):
        curr_centroids = compute_centroids(volume)
        mapping = match_labels(prev_centroids, curr_centroids, max_distance=max_distance)

        used_curr_labels = set(mapping.values())
        new_active_tracks = []

        for track in active_tracks:
            prev_label = track["last_label"]
            if prev_label in mapping:
                curr_label = mapping[prev_label]
                centroid = curr_centroids.get(curr_label)
                if centroid is None:
                    # current label not present (shouldn't happen), drop the track
                    tracks.append(track)
                    continue
                track["last_label"] = curr_label
                track["last_centroid"] = centroid
                track["entries"].append(TrackEntry(frame=frame_index, label=curr_label, centroid=centroid))
                new_active_tracks.append(track)
            else:
                tracks.append(track)

        for curr_label, centroid in curr_centroids.items():
            if curr_label not in used_curr_labels:
                track = {
                    "track_id": next_track_id,
                    "last_label": curr_label,
                    "last_centroid": centroid,
                    "entries": [TrackEntry(frame=frame_index, label=curr_label, centroid=centroid)],
                }
                new_active_tracks.append(track)
                next_track_id += 1

        active_tracks = new_active_tracks
        prev_centroids = {track["last_label"]: track["last_centroid"] for track in active_tracks}

    tracks.extend(active_tracks)
    filtered_tracks = [
        {"track_id": track["track_id"], "entries": track["entries"]}
        for track in tracks
        if len(track["entries"]) >= min_length
    ]
    return filtered_tracks


def summarize_track(track):
    """Summarize a single cell track."""
    frames = [entry.frame for entry in track["entries"]]
    centroids = [entry.centroid for entry in track["entries"]]
    return {
        "track_id": track["track_id"],
        "start_frame": min(frames) if frames else None,
        "end_frame": max(frames) if frames else None,
        "length": len(frames),
        "centroids": np.asarray(centroids),
    }
