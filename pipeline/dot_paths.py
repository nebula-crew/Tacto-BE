"""Detailed layout graph for review; rule graph remains backward compatible."""
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from .schema import MaskClass, NodeType, PathNode, PathEdge, Point


def expand_dot_paths(path, label):
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
            trim = max(1, len(centers)//10)
            fit = np.polyfit(longitudinal[trim:-trim], lateral[trim:-trim], 1)
            predicted = np.polyval(fit, longitudinal)
            width = max(1., float(np.ptp((coords-origin) @ perp)))
            if np.max(np.abs(predicted[trim:-trim]-lateral[trim:-trim])) < width*.08:
                straight = origin + longitudinal[:,None]*axis + predicted[:,None]*perp
                trial = np.zeros(label.shape,np.uint8)
                cv2.polylines(trial,[np.rint(straight).astype(np.int32)],False,1,1)
                if not np.any((trial != 0) & (labels != di)):
                    centers = straight
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
            a = max(nx.single_source_shortest_path_length(g,start), key=nx.single_source_shortest_path_length(g,start).get)
            distances = nx.single_source_shortest_path_length(g,a)
            b = max(distances,key=distances.get)
            route = nx.shortest_path(g,a,b)
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
        joins = {}
        for edge in out.edges_of(old.id):
            if edge.block != MaskClass.LINEAR or len(edge.polyline) < 2:
                continue
            tip = edge.polyline[1] if edge.u == old.id else edge.polyline[-2]
            idx = int(np.argmin(np.sum((centers - [tip.x, tip.y])**2, axis=1)))
            joins.setdefault(idx, []).append(edge)
        indices = sorted({0, len(centers)-1, *joins})
        new_nodes = {}
        nid = max(n.id for n in out.nodes)+1
        for idx in indices:
            pt = Point(x=float(centers[idx,0]), y=float(centers[idx,1]))
            node = PathNode(id=nid, type=NodeType.PASS, pt=pt, block=MaskClass.DOT, component=old.component)
            nid += 1
            new_nodes[idx] = node
            out.nodes.append(node)
            for edge in joins.get(idx, []):
                if edge.u == old.id:
                    edge.u, edge.polyline[0] = node.id, pt
                else:
                    edge.v, edge.polyline[-1] = node.id, pt
                edge.length_px = sum(np.hypot(a.x-b.x,a.y-b.y) for a,b in zip(edge.polyline,edge.polyline[1:]))
        out.nodes.remove(old)
        for a,b in zip(indices,indices[1:]):
            pts = [Point(x=float(x),y=float(y)) for x,y in centers[a:b+1]]
            out.edges.append(PathEdge(id=max((e.id for e in out.edges),default=-1)+1,u=new_nodes[a].id,v=new_nodes[b].id,polyline=pts,length_px=float(np.linalg.norm(np.diff(centers[a:b+1],axis=0),axis=1).sum()),block=MaskClass.DOT))
        for node in new_nodes.values():
            degree = len(out.edges_of(node.id))
            node.type = NodeType.END if degree == 1 else NodeType.BRANCH if degree > 2 else NodeType.PASS
    return out
