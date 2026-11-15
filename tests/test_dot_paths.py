import numpy as np
from pipeline.schema import PhotoKind, MaskClass, NodeType
from pipeline.skeleton import extract_path
from pipeline.dot_paths import expand_dot_paths

def test_long_dot_strip_connects_at_off_center_contact():
    label = np.zeros((400,600),np.uint8)
    label[100:140,50:550] = 2
    label[140:350,140:180] = 1
    original = extract_path(label,PhotoKind.WIDE)
    before = original.model_dump_json()
    path = expand_dot_paths(original,label)
    assert original.model_dump_json() == before
    assert sum(e.block == MaskClass.DOT for e in path.edges) == 2
    junction = next(n for n in path.nodes if n.type == NodeType.BRANCH)
    assert abs(junction.pt.x-160) < 12
    assert abs(junction.pt.y-120) < 5
    assert all(e.u in {n.id for n in path.nodes} and e.v in {n.id for n in path.nodes} for e in path.edges)
