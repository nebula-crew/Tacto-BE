"""Dot extent graph: two ends along a strip, with linear joins on the strip.

DOT edges describe warning-strip coverage, not linear travel direction. Keep
this layout separate from the rule graph so its ends do not cast C2/C3 rays
and dot-only bends never become missing-dot M3 candidates.
"""
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from .schema import MaskClass, NodeType, PathNode, PathEdge, Point


def expand_dot_paths(path, label, *, _include_broad_pads=False):
    if label.ndim != 2 or label.dtype != np.uint8 or label.shape != (path.height, path.width):
        raise ValueError('점형 경로에는 path와 같은 크기의 uint8 라벨 마스크가 필요합니다')
    out = path.model_copy(deep=True)
    count, labels = cv2.connectedComponents((label == MaskClass.DOT).astype(np.uint8))
    for di in range(1, count):
        ys, xs = np.nonzero(labels == di)
        if len(xs) < 400:
            continue
        coords = np.column_stack((xs, ys)).astype(float)
        origin = coords.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov(coords.T))
        if vals[0] <= 0 or vals[1] / vals[0] < 2:
            continue
        axis = vecs[:, 1]
        along = (coords - origin) @ axis
        bins = np.floor((along - along.min()) / 8).astype(int)
        centers = np.array([coords[bins == i].mean(axis=0) for i in range(bins.max()+1) if np.any(bins == i)])
        # Fit the middle of a nearly straight strip; oblique end caps must not
        # create artificial bends. Accept only when the whole line stays inside.
        if len(centers) >= 8:
            perp = np.array([-axis[1], axis[0]])
            longitudinal = (centers-origin) @ axis
            lateral = (centers-origin) @ perp
            trim = max(1, len(centers)//5)
            fit = np.polyfit(longitudinal[trim:-trim], lateral[trim:-trim], 1)
            predicted = np.polyval(fit, longitudinal)
            width = max(1., float(np.ptp((coords-origin) @ perp)))
            if np.max(np.abs(predicted[trim:-trim]-lateral[trim:-trim])) < width*.08:
                straight = origin + longitudinal[:,None]*axis + predicted[:,None]*perp
                trial = np.zeros(label.shape,np.uint8)
                cv2.polylines(trial,[np.rint(straight).astype(np.int32)],False,1,1)
                if not np.any((trial != 0) & (labels != di)):
                    centers = straight
                else:
                    # Perspective/oblique caps skew their cross-section means.
                    # Clip only the two caps of a supported straight fit;
                    # an interior gap or genuine curve must use its own route.
                    ty, tx = np.nonzero(trial)
                    order = np.argsort((np.column_stack((tx, ty))-origin) @ axis)
                    pixels = np.column_stack((tx, ty))[order]
                    valid = labels[pixels[:,1], pixels[:,0]] == di
                    changes = np.diff(np.r_[False, valid, False].astype(np.int8))
                    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
                    if len(starts):
                        longest = int(np.argmax(ends-starts))
                        start, end = starts[longest], ends[longest]
                        if end-start >= .85 * len(pixels):
                            centers = pixels[start:end].astype(float)
        # Reject curves whose cross-section averages leave the actual dot region.
        raster = np.zeros(label.shape, np.uint8)
        cv2.polylines(raster, [np.rint(centers).astype(np.int32)], False, 1, 1)
        if np.any((raster != 0) & (labels != di)):
            # Curved strips cannot be represented by cross-section averages.
            # Follow the longest skeleton route only for elongated dot regions.
            from .skeleton import _pixel_graph
            g = _pixel_graph(skeletonize(labels == di))
            if not g:
                continue
            g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
            if not nx.is_tree(g):
                continue  # Do not silently cut a real loop.
            start = min(g)
            lengths = nx.single_source_dijkstra_path_length(g, start)
            a = max(lengths, key=lengths.get)
            distances = nx.single_source_dijkstra_path_length(g,a)
            b = max(distances,key=distances.get)
            route = nx.shortest_path(g,a,b,weight='weight')
            # Reject substantial branches rather than discarding their topology.
            distance_to_route = nx.multi_source_dijkstra_path_length(g,route)
            radius = cv2.distanceTransform((labels == di).astype(np.uint8),cv2.DIST_L2,3)
            if max(distance_to_route.values()) > max(12.,float(np.median(radius[ys,xs]))*3):
                continue
            centers = np.array([[x,y] for y,x in route],dtype=float)
        candidates = [n for n in out.nodes if n.block == MaskClass.DOT and labels[round(n.pt.y), round(n.pt.x)] == di]
        if len(candidates) != 1:
            continue
        old = candidates[0]
        # A broad warning pad with multiple incoming routes is a 2-D junction,
        # not a thin strip with independently projected joins. Preserve its
        # existing shared anchor; perpendicular projections can create false
        # right-angle elbows and split one junction into several branches.
        shared_pad = vals[1] / vals[0] < 6 and len(out.edges_of(old.id)) >= 2
        if shared_pad and not _include_broad_pads:
            # A pad's span is coverage, not two additional travel branches.
            # Keep its actual rule topology and expose the span separately.
            continue
        joins = {}
        unsupported = False
        for edge in out.edges_of(old.id):
            if edge.block != MaskClass.LINEAR or len(edge.polyline) < 2:
                continue
            tip = edge.polyline[1] if edge.u == old.id else edge.polyline[-2]
            idx = int(np.argmin(np.sum((centers - [tip.x, tip.y])**2, axis=1)))
            # A nearest-point link across a concave strip can cut through
            # background. Keep the original graph if that connector is unsafe.
            a, b = centers[idx], np.array([tip.x, tip.y])
            samples = np.rint(np.linspace(a, b, max(2, int(np.linalg.norm(a-b))*2+1))).astype(int)
            samples[:, 0] = np.clip(samples[:, 0], 0, path.width-1)
            samples[:, 1] = np.clip(samples[:, 1], 0, path.height-1)
            values = label[samples[:, 1], samples[:, 0]]
            missing = values == MaskClass.BACKGROUND
            transitions = np.diff(np.r_[False, missing, False].astype(np.int8))
            gaps = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
            # Sampling is at <=0.5px: tolerate only the same 8px boundary
            # uncertainty used by skeleton attachments, not a real wide gap.
            step = np.linalg.norm(a-b) / (len(samples)-1)
            max_gap = float(gaps.max() * step) if len(gaps) else 0.0
            preserve_anchor = max_gap > 8 or np.any(np.isin(values, [MaskClass.CROSSWALK, MaskClass.ROAD]))
            preserve_anchor = preserve_anchor or shared_pad
            contact = None
            if preserve_anchor and not shared_pad and np.hypot(old.pt.x-tip.x, old.pt.y-tip.y) > 80:
                # A centroid-to-tip connector may traverse most of a long DOT
                # strip. Find its actual DOT exit and attach there instead of
                # drawing the whole virtual segment as a linear diagonal.
                # This only shortens the existing connector: its uncertain
                # exit-to-linear-tip part is preserved, not a new connection.
                old_xy = np.array([old.pt.x, old.pt.y])
                traced = np.rint(np.linspace(old_xy, b,
                    max(2, int(np.linalg.norm(old_xy-b))*2+1))).astype(int)
                on_dot = labels[traced[:,1], traced[:,0]] == di
                changes = np.diff(np.r_[False, on_dot, False].astype(np.int8))
                starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
                runs = [(s, e) for s, e in zip(starts, ends) if e-s >= 6]
                if runs:
                    exit_xy = traced[runs[-1][1]-1]
                    candidate_idx = int(np.argmin(np.sum((centers-exit_xy)**2, axis=1)))
                    bridge = np.zeros(label.shape, np.uint8)
                    cv2.line(bridge, tuple(np.rint(centers[candidate_idx]).astype(int)),
                             tuple(exit_xy), 1, 1)
                    if (np.linalg.norm(exit_xy-old_xy) > 40 and
                            not np.any((bridge != 0) & (labels != di))):
                        idx = candidate_idx
                        contact = Point(x=float(exit_xy[0]), y=float(exit_xy[1]))
                        preserve_anchor = False
            if preserve_anchor:
                # Do not discard an entire dot strip because its pre-existing
                # linear connector is uncertain. Preserve that connector and
                # attach the extent via the original DOT anchor, inside DOT.
                idx = int(np.argmin(np.sum((centers - [old.pt.x, old.pt.y])**2, axis=1)))
                bridge = np.zeros(label.shape, np.uint8)
                cv2.line(bridge, tuple(np.rint(centers[idx]).astype(int)),
                         (round(old.pt.x), round(old.pt.y)), 1, 1)
                if np.any((bridge != 0) & (labels != di)):
                    unsupported = True
                    break
            joins.setdefault(idx, []).append((edge, preserve_anchor, contact))
        if unsupported:
            continue
        indices = sorted({0, len(centers)-1, *joins})
        new_nodes = {}
        nid = max(n.id for n in out.nodes)+1
        for idx in indices:
            pt = Point(x=float(centers[idx,0]), y=float(centers[idx,1]))
            node = PathNode(id=nid, type=NodeType.PASS, pt=pt, block=MaskClass.DOT, component=old.component)
            nid += 1
            new_nodes[idx] = node
            out.nodes.append(node)
            for edge, preserve_anchor, contact in joins.get(idx, []):
                if edge.u == old.id:
                    edge.u = node.id
                    if preserve_anchor:
                        edge.polyline.insert(0, pt)
                    else:
                        edge.polyline[0] = pt
                        if contact is not None:
                            edge.polyline.insert(1, contact)
                else:
                    edge.v = node.id
                    if preserve_anchor:
                        edge.polyline.append(pt)
                    else:
                        edge.polyline[-1] = pt
                        if contact is not None:
                            edge.polyline.insert(len(edge.polyline)-1, contact)
                edge.length_px = sum(np.hypot(a.x-b.x,a.y-b.y) for a,b in zip(edge.polyline,edge.polyline[1:]))
        out.nodes.remove(old)
        for a,b in zip(indices,indices[1:]):
            pts = [Point(x=float(x),y=float(y)) for x,y in centers[a:b+1]]
            out.edges.append(PathEdge(id=max((e.id for e in out.edges),default=-1)+1,u=new_nodes[a].id,v=new_nodes[b].id,polyline=pts,length_px=float(np.linalg.norm(np.diff(centers[a:b+1],axis=0),axis=1).sum()),block=MaskClass.DOT))
        for node in new_nodes.values():
            degree = len(out.edges_of(node.id))
            node.type = NodeType.END if degree == 1 else NodeType.BRANCH if degree > 2 else NodeType.PASS
    return out


def dot_pad_extents(path, label):
    """Independent coverage annotations; never add nodes/edges to the path."""
    _, ids = cv2.connectedComponents((label == MaskClass.DOT).astype(np.uint8))
    selected = []
    for node in path.nodes:
        if node.block != MaskClass.DOT or len(path.edges_of(node.id)) < 2:
            continue
        component = ids[round(node.pt.y), round(node.pt.x)]
        ys, xs = np.nonzero(ids == component) if component else ([], [])
        if len(xs) < 400:
            continue
        values = np.linalg.eigvalsh(np.cov(np.column_stack((xs, ys)).T))
        if values[0] > 0 and 2 <= values[1]/values[0] < 6:
            selected.append(int(component))
    if not selected:
        return []
    detailed = expand_dot_paths(path, label, _include_broad_pads=True)
    extents = []
    for component in sorted(set(selected)):
        edges = [e for e in detailed.edges if e.block == MaskClass.DOT and e.polyline
                 and ids[round(e.polyline[0].y), round(e.polyline[0].x)] == component]
        degree = {}
        for e in edges:
            for nid in (e.u, e.v):
                degree[nid] = degree.get(nid, 0)+1
        if edges:
            extents.append({'dot_component': component,
                'polylines': [[p.model_dump() for p in e.polyline] for e in edges],
                'endpoints': [detailed.node(nid).pt.model_dump() for nid, d in degree.items() if d == 1]})
    return extents


def draw_dot_pad_extents(image, extents):
    """Dashed coverage spans distinguish area extent from connected routes."""
    for extent in extents:
        for line in extent['polylines']:
            phase = 0.0
            for p, q in zip(line, line[1:]):
                a, b = np.array([p['x'],p['y']]), np.array([q['x'],q['y']])
                length = np.linalg.norm(b-a)
                steps = max(1,int(np.ceil(length)))
                for i in range(steps):
                    if int((phase+i*length/steps)//9) % 2 == 0:
                        xy = tuple(np.rint(a+(b-a)*(i/steps)).astype(int))
                        cv2.circle(image,xy,1,(0,200,230),-1)
                phase += length
        for p in extent['endpoints']:
            cv2.circle(image,(round(p['x']),round(p['y'])),5,(0,200,230),1)
    return image
