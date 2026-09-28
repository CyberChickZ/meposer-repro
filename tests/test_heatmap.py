import numpy as np

from meposer.data.heatmap import heatmap_argmax, render_heatmaps


def test_peak_location_and_visibility():
    p2d = np.array([[320.0, 240.0], [-1.0, -1.0], [16.0, 8.0]], np.float32)
    hm, vis = render_heatmaps(p2d, (640, 480), (40, 30), sigma=1.0)
    assert hm.shape == (3, 30, 40)
    assert vis.tolist() == [True, False, True]
    peaks, conf = heatmap_argmax(hm)
    assert peaks[0].tolist() == [20.0, 15.0]
    assert peaks[2].tolist() == [1.0, 0.0]
    assert conf[1] == 0.0 and hm[1].sum() == 0.0
