import os
import sys

sys.path.insert(0, os.getcwd())
import clip
import json
import torch
import GPUtil
import argparse
import itertools
import numpy as np
import dill as pickle
from tqdm import tqdm
import torchvision.transforms as transforms

from ovamc.data_loader import OVAD_Boxes
from ovamc.misc import object_attribute_templates, ovad_validate


def get_arguments():
    parser = argparse.ArgumentParser(description="Clip evaluation")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ovad2000",
        help="dataset name",
    )
    parser.add_argument(
        "--ann_file",
        type=str,
        default="datasets/ovad/ovad2000.json",
        help="annotation file with images and objects for attribute annotation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/OVAD_Benchmark2000/multimodal_ovac/clip/",
        help="dir where models are",
    )
    parser.add_argument(
        "--dir_data",
        type=str,
        default="datasets/ovad_box_instances/2000_img",
        help="image data dir",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="RN50",  # "RN50" #"RN101" #"RN50x4" #"ViT-B/32" #"ViT-B/16"
        help="architecture name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="all",
        help="prompt",
    )
    parser.add_argument(
        "--average_syn",
        action="store_true",
    )
    # parser.add_argument(
    #     "-bs",
    #     "--batch_size",
    #     type=int,
    #     default=1,
    # )

    return parser.parse_args()


def encode_clip(template_list, object_word, model, device):
    if isinstance(template_list[0], list):
        # it is a list of list of strings
        avg_synonyms = True
        sentences = list(itertools.chain.from_iterable(template_list))
        # print("flattened_sentences", len(sentences))
    elif isinstance(template_list[0], str):
        avg_synonyms = False
        sentences = template_list
    text = clip.tokenize([sentence.format(object_word=object_word) for sentence in sentences]).to(device)
    with torch.no_grad():
        if len(text) > 10000:
            text_features = torch.cat(
                [
                    model.encode_text(text[: len(text) // 2]),
                    model.encode_text(text[len(text) // 2 :]),
                ],
                dim=0,
            )
        else:
            text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # print("text_features.shape", text_features.shape)
    if avg_synonyms:
        synonyms_per_cat = [len(x) for x in template_list]
        text_features = text_features.split(synonyms_per_cat, dim=0)
        text_features = [x.mean(dim=0) for x in text_features]
        text_features = torch.stack(text_features, dim=0)
        # print("after stack", text_features.shape)

    return text_features


def main(args):
    # create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    use_prompts = ["a", "the", "none", "photo"]
    if args.prompt in use_prompts:  # or args.prompt == "photo":
        use_prompts = [args.prompt]

    # load annotations
    annotations = json.load(open(args.ann_file, "r"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    obj_label_to_word = {annotations["categories"][i]['id']: annotations["categories"][i]['name']
                         for i in range(len(annotations["categories"]))}

    print("Loading CLIP")
    model, _ = clip.load(args.model_arch, device=device)

    # Make transform
    channel_stats = dict(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224), interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats),
        ]
    )

    # Data loader
    dataset = OVAD_Boxes(root=args.dir_data, transform=transform)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=False,
    # )

    # Caption templates for each attribute
    all_att_templates = []
    for att_dict in annotations["attributes"]:
        att_w_type = att_dict["name"]
        att_type, att_list = att_w_type.split(":")
        is_has = att_dict["is_has_att"]
        dobj_name = (
            att_type.replace(" tone", "")
            # So far only for tone worked to remove the word
            # .replace(" color", "")
            # .replace(" pattern", "")
            # .replace(" expression", "")
            # .replace(" type", "")
            # .replace(" length", "")
        )

        # extend the maturity to include other words
        if att_list == "young/baby":
            att_list += "/kid/kids/child/toddler/boy/girl"
        elif att_list == "adult/old/aged":
            att_list += "/teen/elder"
        att_templates = []
        for syn in att_list.split("/"):
            for prompt in use_prompts:
                for template in object_attribute_templates[is_has][prompt]:
                    if is_has == "has":
                        att_templates.append(
                            template.format(
                                attr=syn, dobj=dobj_name, noun="{object_word}"
                            ).strip()
                        )
                    elif is_has == "is":
                        att_templates.append(
                            template.format(attr=syn, noun="{object_word}").strip()
                        )
        all_att_templates.append(att_templates)

    att_templates_syn = all_att_templates
    len_synonyms = [len(att_synonyms) for att_synonyms in all_att_templates]
    att_ids = [
        [att_dict["id"]] * len(att_synonyms)
        for att_dict, att_synonyms in zip(
            annotations["attributes"], att_templates_syn
        )
    ]
    att_ids = list(itertools.chain.from_iterable(att_ids))
    all_att_templates = list(itertools.chain.from_iterable(all_att_templates))


    # Caption base templates for each object
    all_base_templates = []
    for prompt in use_prompts:
        for template in object_attribute_templates[is_has][prompt]:
            if is_has == "has":
                all_base_templates.append(
                    template.format(attr="", dobj="", noun="{object_word}").strip()
                )
            elif is_has == "is":
                all_base_templates.append(
                    template.format(attr="", noun="{object_word}").strip()
                )

    # begin
    pred_vectors = []
    label_vectors = []
    indx_max_syn = []
    with torch.no_grad():
        # go through each data point one by one in order to treat each object word separately
        for i, (images, labels) in tqdm(enumerate(dataset), total=len(dataset)):
            att_label, obj_label = labels[0], labels[1]
            label_vectors.append(att_label)  #.cpu().numpy())
            object_word = obj_label_to_word[obj_label]

            # collect text features
            text_base_features = encode_clip(all_base_templates, object_word, model, device)
            text_attr_features = encode_clip(all_att_templates, object_word, model, device)

            # predict
            images = images.unsqueeze(dim=0).to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits_attr = image_features.to(dtype=text_attr_features.dtype) @ text_attr_features.T
            logits_base = image_features.to(dtype=text_base_features.dtype) @ text_base_features.T

            # subtract base
            logits_attr = logits_attr - logits_base.mean(dim=1)

            # split into synonyms
            x_attrs_syn = logits_attr.split(len_synonyms, dim=1)
            # take arg max
            x_attrs_maxsyn = []
            x_attrs_idxsyn = []
            for x_syn in x_attrs_syn:
                if args.average_syn:
                    xmax_val = x_syn.mean(axis=1)
                    xmax_idx = torch.zeros((1, args.batch_size))
                else:
                    xmax_val, xmax_idx = x_syn.max(axis=1)
                x_attrs_maxsyn.append(xmax_val)
                x_attrs_idxsyn.append(xmax_idx)
            idx_attrs = torch.stack(x_attrs_idxsyn, axis=1)
            x_attrs = torch.stack(x_attrs_maxsyn, axis=1)

            pred_vectors.append(x_attrs.cpu().numpy())
            indx_max_syn.append(idx_attrs.cpu().numpy())

            # if i % 50 == 0:
            #     print("Processed {} out of {}".format(i, len(dataset)), end="\r")
                # if 0 < i < 300:
                #     GPUtil.showUtilization(all=True)

    pred_vectors = np.concatenate(pred_vectors, axis=0)
    label_vectors = np.stack(label_vectors, axis=0)  # np.concatenate(label_vectors, axis=0)
    indx_max_syn = np.concatenate(indx_max_syn, axis=0)
    ovad_validate(
        annotations["attributes"],
        pred_vectors,
        label_vectors,
        args.output_dir,
        args.dataset_name,
    )


if __name__ == "__main__":
    args = get_arguments()
    main(args)
