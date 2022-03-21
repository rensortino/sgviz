from einops import rearrange
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
import cv2
from data.utils import imagenet_preprocess

from utils.data import img_to_PIL, inv_normalize

COLORS = [
            (137, 49, 239), # Blue-Violet
            (242, 202, 25), # Yellow
            (255, 0, 189), # Pink
            (0, 87, 233), # Blue
            (135, 233, 17), # Green
            (225, 24, 69), # Orange
            (1, 176, 231), # Cyan
            (138, 10, 183), # Violet
            (138, 59, 59), # Brown
            ]

HEXCOLORS = [
            '#8931EF', # Blue-Violet
            '#F2CA19', # Yellow
            '#FF00BD', # Pink
            '#0057E9', # Blue
            '#87E911', # Green
            '#FF1845', # Orange
            '#01B0E7', # Cyan
            '#8A0AB7', # Violet
            '#8A3B3B', # Brown
            ]

def preprocess_image(img):
    img_w_boxes = img.clone()
    img_w_boxes = inv_normalize(img_w_boxes) * 255
    img_w_boxes = rearrange(img_w_boxes, 'c h w -> h w c')
    return img_w_boxes.cpu().numpy().copy()

def postprocess_image(img_w_boxes):
    img_w_boxes = torch.tensor(img_w_boxes)
    img_w_boxes = rearrange(img_w_boxes, 'h w c -> c h w')
    img_w_boxes = imagenet_preprocess()(img_w_boxes)
    pil_img = img_to_PIL(img_w_boxes)
    return pil_img

def draw_box(img, box_coords, idx, category=None):
    '''
    img: shape [H,W,C]
    '''

    H,W = img.shape[:2]

    # Rescale [0,1] coordinates to image size
    box_coords = box_coords * torch.tensor([W,H,W,H]).to(box_coords)
    x0, y0, x1, y1 = box_coords.int().tolist()
    color = COLORS[idx % len(COLORS)]
    thickness = 1
    img_w_box = cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if category:
        cv2.putText(img_w_box, category, (x0, y0+20), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 2)

    # cv2.imwrite('boxes.png', img_w_box)

    return img_w_box

def remove_special_nodes(g, idx=0):
    '''
    Removes special __PAD__ and __image__ nodes with the respective
    relationships, mainly for clean visualization purposes
    '''
    batch_size = g.batch.max().item() + 1
    # Every batch has the same number of nodes (whether actual nodes or PAD)
    max_n_objs = g.batch.shape[0] // batch_size
    # Take all the indices of the selected batch, which are not padding
    keep_nodes = torch.logical_and(g.batch.eq(idx), g.y.ne(173))
    # and are not __image__
    keep_nodes = torch.logical_and(keep_nodes, g.y.ne(0))
    g.x = g.x[keep_nodes]
    g.y = g.y[keep_nodes]
    # Keep all edge indices within the selected graph index
    selected_edges = g.edge_batch == idx
    # and which are not referring to the __image__ node
    edges_with_actual_nodes = g.edge_index[1] < g.edge_index[:,selected_edges].max()
    keep_edges = torch.logical_and(selected_edges, edges_with_actual_nodes)
    g.edge_index = g.edge_index[:,keep_edges]
    if idx != 0:
        g.edge_index = g.edge_index % (max_n_objs * idx)
    g.edge_attr = g.edge_attr[keep_edges]
    return g

def visualize_graph(graph, idx, vocab):

    edge_labels = vocab['pred_idx_to_name']
    node_labels = vocab['object_idx_to_name']

    h = graph.clone()
    h = remove_special_nodes(h, idx)


    colors = [(r / 255, g / 255, b / 255) for (r,g,b) in COLORS]
    colors = [colors[i % len(COLORS)] for i in range(len(h.x))]
    edge_to_label = get_edge2label(h.edge_index, h.edge_attr, edge_labels)

    fig = plt.figure(figsize=(8,8))
    plt.xticks([])
    plt.yticks([])

    node_vocab = {i: node_labels[el] for i, el in enumerate(h.y)}

    nx_graph = to_networkx(h)
    pos = nx.planar_layout(nx_graph)
    nx.draw_networkx(nx_graph, pos=pos, node_size=800,
                        node_color=colors, with_labels=True, arrowsize=20)
    nx.draw_networkx_labels(nx_graph, pos, labels=node_vocab, font_color='black', font_weight='bold', verticalalignment='top', horizontalalignment='left', font_size=18)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_to_label,font_color='black', font_family='sans-serif', font_weight='bold', font_size=18)

    return fig

def get_edge2label(edges, relationships, rel_vocab):
    edge_to_label = {}
    for edge, rel in zip(edges.t(), relationships):
        edge_to_label[tuple(edge.tolist())] = rel_vocab[rel.item()]
    return edge_to_label