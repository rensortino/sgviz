import json
from utils.visualize import *
import yaml
from tqdm import tqdm
from sg2im.coco import build_coco_dsets, coco_collate_fn
import json
from utils.misc import make_reproducible
from attrdict import AttrDict
from pathlib import Path

def main():
    with open("args.yaml") as y:
        args = AttrDict(yaml.safe_load(y))
    args.data_dir = Path(args.data_dir)


    make_reproducible(42)
    data_path = Path('data/coco')
    args.data_dir /= 'coco'
    collate_fn = coco_collate_fn
    config_file = 'sg2im/coco'

    with open(f"{config_file}.json") as conf:
        args.data = AttrDict(json.load(conf))

    with open(data_path / 'vocab.json') as v:
        vocab = json.load(v)
        node_vocab = vocab['object_name_to_idx']
        rel_vocab = vocab['pred_name_to_idx']
        rel2name = vocab['pred_idx_to_name']
        node2name = vocab['object_idx_to_name']


    # Data Loading
    vocab, train_dset, val_dset, _ = build_coco_dsets(args)
    dsets = {'train': train_dset, 'val': val_dset}
    # dsets = {'val': val_dset}

    for split, dset in dsets.items():
        sg_dicts = []
        for i, el in tqdm(enumerate(dset)):
            if i == 1000:
                break
            sg_dict = {}
            img_id = train_dset.image_ids[i]
            filename = train_dset.image_id_to_filename[img_id]
            with open(f'{split}_images.txt', 'a') as o:
                o.write(filename + '\n')
            sg_dict['img_id'] = img_id
            sg_dict['filename'] = filename
            image, objs, boxes, masks, triples, path = el
            sg_dict['objs'] = objs.tolist()
            subjects, predicates, objects = triples.chunk(3, dim=1)
            sg_dict['triples'] = []

            for t in triples:
                s,p,o = t.tolist()
                subject = vocab['object_idx_to_name'][objs[s]]
                predicate = vocab['pred_idx_to_name'][p]
                obj = vocab['object_idx_to_name'][objs[o]]
                sg_dict['triples'].append([s, p, o])

            sg_dict['boxes'] = []

            for b in boxes:
                sg_dict['boxes'].append([b.tolist()])

            sg_dict['segmentation'] = []
            for object_data in train_dset.image_id_to_objects[img_id]:
                seg = object_data['segmentation']
                box = object_data['bbox']
                sg_dict['segmentation'].append({'seg': seg, 'bbox': box}) # Bbox coordinates are needed in the collate function

            sg_dicts.append(sg_dict)

        ann_json = json.dumps(sg_dicts)
        with open(f'{data_path / "annotations" /  split}.json', 'w') as out:
            json.dump(json.loads(ann_json), out)
            # images, objects, boxes, masks, triples, obj_to_img, triple_to_img, paths = coco_collate_fn([el], padding=False, return_paths=True)


if __name__ == '__main__':
    main()