import json
from data.coco import CocoSceneGraphDataset
from utils.visualize import *
import yaml
from data.coco import coco_collate_fn
import json
from PIL import Image
import torchvision.transforms.functional as TF
from attrdict import AttrDict
from torch_geometric.data import Data
from pathlib import Path
import streamlit as st
from streamlit.server import server
import streamlit.components.v1 as components
from pyvis.network import Network
import copy

@st.cache(allow_output_mutation=True)
def setup_args(dset):
    with open("args.yaml") as y:
        args = AttrDict(yaml.safe_load(y))
    args.data_dir = Path(args.data_dir)

    if dset == 'Visual Genome':
        args.data_dir /= 'vg'
        config_file = 'config/vg'
        args.vocab_json = args.data_dir / args.vocab_json
        args.train_file = args.data_dir / args.train_file
        args.val_file = args.data_dir / args.val_file
        args.test_file = args.data_dir / args.test_file
    elif dset == 'COCO':
        args.data_dir /= 'coco'
        config_file = 'config/coco'

    with open(f"{config_file}.json") as conf:
        args.data = AttrDict(json.load(conf))
    return copy.deepcopy(args)

@st.cache
def setup_vocab(data_path):
    with open(data_path / 'vocab.json') as v:
        vocab = json.load(v)
    return vocab
        
@st.cache(allow_output_mutation=True)
def get_dataset(dset, args):
    print('Executing cached function')
    with st.spinner(text=f'Creating and caching Data Module {dset}'):
        if dset == 'COCO':
            # vocab, train_dset, _, _ = load_coco_dsets(args)
            train_dset = CocoSceneGraphDataset(args.data, train=True)
        elif dset == 'Visual Genome':
            vocab, train_dset, _, _ = build_vg_dsets(args)
    return train_dset

# Set up Streamlit
st.title('Scene Graph Visualization')
# dset = st.sidebar.selectbox('Dataset', options=['COCO', 'Visual Genome'])
dset = 'COCO'
img_idx = int(st.sidebar.number_input('Dataset Index', step=1))
# batch_idx = int(st.sidebar.number_input('Batch Index', step=1))
batch_idx = 0

args = setup_args(dset)
vocab = setup_vocab(args.data_dir)
st_dir = Path('out')
st_dir.mkdir(exist_ok=True)


node_vocab = vocab['object_name_to_idx']
rel_vocab = vocab['pred_name_to_idx']
rel2name = vocab['pred_idx_to_name']
node2name = vocab['object_idx_to_name']

# Data Loading
train_dset = get_dataset(dset, args)
# data_module = DataModule(args, args.data.img_size, data_dir=data_path, debug=args.debug)

if dset == 'Visual Genome':
    images, objects, boxes, triples, obj_to_img, triple_to_img, paths = vg_collate_fn([train_dset[img_idx]], padding=False, return_paths=True)
elif dset == 'COCO':
    images, objects, boxes, masks, triples, obj_to_img, triple_to_img, paths = coco_collate_fn([train_dset[img_idx]], padding=False, return_paths=True)

s,p,o = triples.chunk(3, dim=1)
p = p.squeeze(1)
edges = torch.cat((s,o), dim=1).t()

graph = Data(x=objects, edge_index=edges, edge_attr=p, y=objects, batch=obj_to_img, edge_batch=triple_to_img)

h = graph.clone()
h = remove_special_nodes(h, batch_idx)

node_objects = [node2name[i] for i in h.y]
node_relationships = [rel2name[i] for i in h.edge_attr]

im = Image.open(paths[batch_idx])
st.image(im, f'Image at {paths[batch_idx]}')


nt = Network('500px', '500px', directed=True)
nt.add_nodes(node_objects, label=node_objects,
                color=[HEXCOLORS[i % len(HEXCOLORS)] for i, _ in enumerate(h.y)])

for i, edge in enumerate(h.edge_index.t()):
    from_node = node2name[h.y[edge[0].item()]]
    to_node = node2name[h.y[edge[1].item()]]
    nt.add_edge(from_node, to_node, title=node_relationships[i])

graph_html_path = st_dir / 'scene_graph.html'
nt.save_graph(str(graph_html_path))
graph_html = open(graph_html_path, 'r', encoding='utf-8')
source_code = graph_html.read() 
components.html(source_code, height=500, width=500, )

im = TF.to_tensor(Image.open(paths[batch_idx]))
im = imagenet_preprocess()(im)
im = preprocess_image(im)
idx_mask = graph.batch == batch_idx
boxes = boxes[idx_mask]
for i, box in enumerate(boxes):
    category = node2name[graph.y[i]]
    im = draw_box(im, box, i, category)
im = postprocess_image(im)
st.image(im, 'Image With Boxes')