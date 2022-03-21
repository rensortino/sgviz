from data_modules.base import BaseDataModule
from utils.data import *
from torch_geometric.data import Data
from sg2im.coco import coco_collate_fn, load_coco_dsets
from data_modules.loader import *
from utils.data import *
from utils.tokens import *
import torchvision.transforms as T
from sg2im.coco import build_coco_dsets

class COCODataModule(BaseDataModule):
    def __init__(self, args, image_size, data_dir="data", debug=False, with_paths=False):
        super(COCODataModule, self).__init__(args, image_size, data_dir, debug, with_paths)
        if (self.args.data_dir / 'train.pt').exists() and \
            (self.args.data_dir / 'val.pt').exists() and \
            (self.args.data_dir / 'vocab.pt').exists():
            args.vocab, self.train_ds, self.val_ds, self.test_ds, =  load_coco_dsets(args)
        else:
            args.vocab, self.train_ds, self.val_ds, self.test_ds, =  build_coco_dsets(args)

    def graph_collate_fn(self, batch):
        '''
        boxes: (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        '''
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        images, objects, boxes, masks, triples, obj_to_img, triple_to_img = coco_collate_fn(batch)

        s,p,o = triples.chunk(3, dim=1)
        p = p.squeeze(1)
        edges = torch.cat((s,o), dim=1).t()
        # ys are labels used to keep track of each node's category after their processing by the gcn
        graph = Data(x=objects, edge_index=edges, edge_attr=p, y=objects, batch=obj_to_img, edge_batch=triple_to_img)
        
        return images, graph, boxes, masks
    
    def graph_with_paths_collate_fn(self, batch):
        '''
        boxes: (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        '''
        # tgt = [<SOS>, [embeddings], <EOS> (, [<PAD>] ) ]
        images, objects, boxes, masks, triples, obj_to_img, triple_to_img, paths = coco_collate_fn(batch, return_paths=True)

        s,p,o = triples.chunk(3, dim=1)
        p = p.squeeze(1)
        edges = torch.cat((s,o), dim=1).t()
        # ys are labels used to keep track of each node's category after their processing by the gcn
        graph = Data(x=objects, edge_index=edges, edge_attr=p, y=objects, batch=obj_to_img, edge_batch=triple_to_img)
        
        return images, graph, boxes, masks, paths
