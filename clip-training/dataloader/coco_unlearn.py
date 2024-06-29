import re

def coco_unlearn_object(annotations, object):
    image_id_remove = []

    caption_remove = []
    co_caption_remove = []
    image_remove = []

    pattern = r'\b{}\b'.format(object)

    for caption_info in annotations['annotations']:
        if re.search(pattern, caption_info['caption']):
            #print(caption_info['caption'])
            caption_remove.append(caption_info)
            image_id_remove.append(caption_info['image_id'])

    for caption_info in caption_remove:
        annotations['annotations'].remove(caption_info)

    print(len(caption_remove), "caption removed.")

    image_id_remove = set(image_id_remove)

    for caption_info in annotations['annotations']:
        for rm_id in image_id_remove:
            if rm_id == caption_info['image_id']:
                co_caption_remove.append(caption_info)

    for caption_info in co_caption_remove:
        annotations['annotations'].remove(caption_info)

    print(len(co_caption_remove), "corresponding caption removed.")

    print("Total", len(co_caption_remove)+len(caption_remove), "caption removed")

    for img_info in annotations['images']:
        for rm_id in image_id_remove:
            if rm_id == img_info['id']:
                image_remove.append(img_info)

    for img_info in image_remove:
        annotations['images'].remove(img_info)

    print(len(image_remove), "images removed")

    return annotations