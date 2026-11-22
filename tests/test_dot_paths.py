import numpy as np
import cv2
import pytest
from pipeline.schema import PhotoKind, MaskClass, NodeType
from pipeline.skeleton import extract_path
from pipeline.dot_paths import expand_dot_paths
from pipeline.shape_rules import find_m3_candidates

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


@pytest.mark.parametrize('angle', [0, 35, 90, 150])
def test_dot_only_strip_has_two_ends_and_no_missing_dot_candidate(angle):
    label = np.zeros((600,600), np.uint8)
    label[270:330, 80:520] = 2
    label = cv2.warpAffine(label, cv2.getRotationMatrix2D((300,300),angle,1),
                           (600,600),flags=cv2.INTER_NEAREST)
    original = extract_path(label, PhotoKind.WIDE)
    layout = expand_dot_paths(original, label)
    assert len(layout.edges) == 1 and layout.edges[0].block == MaskClass.DOT
    assert len(layout.nodes) == 2 and all(n.type == NodeType.END for n in layout.nodes)
    assert layout.edges[0].length_px > 400
    assert find_m3_candidates(label, original) == []


def test_mixed_t_layout_has_two_dot_ends_and_a_join():
    label = np.zeros((500,600), np.uint8)
    label[120:170,50:550] = 2
    label[170:450,275:325] = 1
    original = extract_path(label, PhotoKind.WIDE)
    before = [vars(c) for c in find_m3_candidates(label, original)]
    layout = expand_dot_paths(original, label)
    dot_ends = [n for n in layout.nodes if n.block == MaskClass.DOT and n.type == NodeType.END]
    assert len(dot_ends) == 2
    assert min(n.pt.x for n in dot_ends) < 60
    assert max(n.pt.x for n in dot_ends) > 540
    assert sum(n.type == NodeType.BRANCH for n in layout.nodes) == 1
    assert [vars(c) for c in find_m3_candidates(label, original)] == before


def test_disconnected_dot_strips_are_not_joined():
    label = np.zeros((400,600),np.uint8)
    label[100:145,30:250] = 2
    label[100:145,300:570] = 2
    layout = expand_dot_paths(extract_path(label,PhotoKind.WIDE),label)
    assert len(layout.edges) == 2
    assert len({n.component for n in layout.nodes}) == 2
    for edge in layout.edges:
        assert not (min(p.x for p in edge.polyline) < 250 and max(p.x for p in edge.polyline) > 300)


def test_curved_dot_strip_stays_inside_mask():
    label = np.zeros((600,600),np.uint8)
    cv2.ellipse(label,(300,300),(210,160),0,25,155,2,45)
    layout = expand_dot_paths(extract_path(label,PhotoKind.WIDE),label)
    assert any(e.block == MaskClass.DOT for e in layout.edges)
    raster = np.zeros_like(label)
    for edge in layout.edges:
        pts=np.rint([[p.x,p.y] for p in edge.polyline]).astype(np.int32)
        cv2.polylines(raster,[pts],False,1,1)
    assert not np.any((raster != 0) & (label != 2))


@pytest.mark.parametrize('reverse', [False, True])
def test_long_virtual_connector_uses_observed_dot_exit(reverse):
    from pipeline.schema import PathStructure, PathNode, PathEdge, Point
    label = np.zeros((600,400),np.uint8)
    label[50:550,100:160] = 2
    label[40:80,180:350] = 1
    # An existing uncertain attachment traverses the entire DOT strip. The
    # layout may shorten its DOT part but must preserve the linear tip onward.
    tip, end, anchor = Point(x=180,y=60), Point(x=320,y=60), Point(x=130,y=300)
    points = [end, tip, anchor]
    path = PathStructure(photo=PhotoKind.WIDE,width=400,height=600,
        nodes=[PathNode(id=0,type=NodeType.END,pt=end,block=MaskClass.LINEAR,component=1),
               PathNode(id=1,type=NodeType.END,pt=anchor,block=MaskClass.DOT,component=1)],
        edges=[PathEdge(id=0,u=1 if reverse else 0,v=0 if reverse else 1,
            polyline=list(reversed(points)) if reverse else points,
            length_px=400,block=MaskClass.LINEAR)])
    original = path.model_dump_json()
    layout = expand_dot_paths(path,label)
    assert path.model_dump_json() == original
    edge = next(e for e in layout.edges if e.block == MaskClass.LINEAR)
    outward = edge.polyline if edge.u == 0 else list(reversed(edge.polyline))
    assert outward[:2] == [end,tip]
    assert outward[-1].y < 200  # no return to the old centroid at y=300
    assert label[round(outward[-2].y), round(outward[-2].x)] == MaskClass.DOT
    assert sum(e.length_px for e in layout.edges if e.block == MaskClass.DOT) > 450


@pytest.mark.parametrize('angle', [0, 30, 90, 160])
def test_broad_dot_pad_keeps_one_shared_linear_junction(angle):
    label = np.zeros((800,800),np.uint8)
    label[300:450,200:560] = 2
    label[100:300,365:405] = 1
    label[390:430,560:740] = 1
    label = cv2.warpAffine(label, cv2.getRotationMatrix2D((400,400),angle,1),
                           (800,800), flags=cv2.INTER_NEAREST)
    rule = extract_path(label,PhotoKind.WIDE)
    frozen = rule.model_dump_json()
    old_anchor = next(n for n in rule.nodes if n.block == MaskClass.DOT and len(rule.edges_of(n.id)) == 2)
    layout = expand_dot_paths(rule,label)
    assert rule.model_dump_json() == frozen
    joins = [n for n in layout.nodes if n.block == MaskClass.DOT
             and any(e.block == MaskClass.LINEAR for e in layout.edges_of(n.id))]
    assert len(joins) == 1
    assert len([e for e in layout.edges_of(joins[0].id) if e.block == MaskClass.LINEAR]) == 2
    assert joins[0].type == NodeType.PASS
    assert layout.model_dump() == rule.model_dump()  # No added route/branch.
    from pipeline.dot_paths import dot_pad_extents
    extents = dot_pad_extents(rule,label)
    assert len(extents) == 1 and len(extents[0]['endpoints']) == 2
    for edge in layout.edges:
        if edge.block == MaskClass.LINEAR:
            # Preserve the original approach instead of inventing an elbow.
            assert old_anchor.pt in edge.polyline
