from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm



def evaluate(model, dataloader, tokenizer, device, amp=True, recall_k_list=[5], 
             use_itm=False, use_qwen2vl=False):
    """
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn,Module
        CLIP-like model with `encode_image` and `encode_text`

    dataloader: torch.utils.data.Dataloader
        dataloader to use for evaluation

    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers

    device: cpu/cuda

    amp: whether to use automatic mixed precision

    recall_k_list: list of int
        recall@k k's to use

    Returns
    -------

    dict of retrieval metrics
    """
    print("Evaluating")
    print(f"Using ITM: {use_itm}")
    print(f"Using Qwen2VL: {use_qwen2vl}")
    assert not (use_itm and use_qwen2vl), "Only one of ITM and Qwen2VL can be used"
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    # list of batches of images and texts
    texts_tok_list, images_list = [], []
    dataloader = dataloader_with_indices(dataloader)
    autocast = torch.cuda.amp.autocast if amp else suppress
    for batch_images, batch_texts, inds in tqdm(dataloader):
        # tokenize all texts in the batch
        if use_qwen2vl:
            batch_images = list(batch_images)
            # do not tokenize the texts in the batch
            batch_texts_tok = [text for i, texts in enumerate(batch_texts) for text in texts]
            # ps: no need to put the images and texts to device, which will be done in the encode_image and encode_text
        else:
            batch_images = batch_images.to(device)
            batch_texts_tok = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        texts_tok_list.append(batch_texts_tok)
        images_list.append(batch_images)
        # compute the embedding of images and texts
        with torch.no_grad(), autocast():
            try:
                print("the data type of batch_images is: ", batch_images.dtype)
                print("the data type of batch_texts_tok is: ", batch_texts_tok.dtype)
            except:
                # str has no attribute dtype
                print("the data type of batch_images is: ", type(batch_images[0]))
                print("the data type of batch_texts_tok is: ", type(batch_texts_tok[0]))
            if use_qwen2vl:
                batch_images_emb = F.normalize(model.encode_image(batch_images, device), dim=-1)
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok, device), dim=-1)
            else:
                batch_images_emb = F.normalize(model.encode_image(batch_images), dim=-1)
                batch_texts_emb = F.normalize(model.encode_text(batch_texts_tok), dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)
    if use_itm:
        # concatenate all texts and images
        texts_tok = torch.cat(texts_tok_list)
        images = torch.cat(images_list)
        print("the shape of texts_tok is {}".format(texts_tok.shape))
        print("the shape of the images is {}".format(images.shape))
        print("the data type of images is: ", images.dtype)
        print("the data type of texts_tok is: ", texts_tok.dtype)

    # get the score for each text and image pair
    scores = texts_emb @ images_emb.t()

    # construct a the positive pair matrix, which tells whether each text-image pair is a positive or not
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    metrics = {}
    for recall_k in recall_k_list:
        # Note that recall_at_k computes **actual** recall i.e. nb_true_positive/nb_positives, where the number
        # of true positives, e.g. for text retrieval, is, for each image,  the number of retrieved texts matching that image among the top-k.
        # Also, the number of positives are the total number of texts matching the image in the dataset, as we have a set of captions
        # for each image, that number will be greater than 1 for text retrieval.
        # However, image/text retrieval recall@k, the way it is done in CLIP-like papers, is a bit different.
        # recall@k, in CLIP-like papers, is, for each image, either 1 or 0. It is 1 if atleast one text matches the image among the top-k.
        # so we can easily compute that using the actual recall, by checking whether there is at least one true positive,
        # which would be the case if the recall is greater than 0. One we compute the recal for each image (or text), we average
        # it over the dataset.
        if recall_k ==1  and use_itm:
            func = recall_top1_itm
        else:
            func = recall_at_k
            texts_tok = None
            images = None
            model=None
            
        # only use itm in image retrieval
        metrics[f'image_retrieval_recall@{recall_k}'] = (
                    batchify(func, scores, positive_pairs, batch_size, device, 
                             Z=texts_tok, T=images,
                             k=10 if use_itm and recall_k==1 else recall_k, 
                             model=model) > 0).float().mean().item()
        metrics[f'text_retrieval_recall@{recall_k}'] = (
                    batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device,
                             k=recall_k) > 0).float().mean().item()

    return metrics


def dataloader_with_indices(dataloader):
    start = 0
    for x, y in dataloader:
        end = start + len(x)
        inds = torch.arange(start, end)
        yield x, y, inds
        start = end


def recall_at_k(scores, positive_pairs, k, model=None):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def recall_top1_itm(scores, positive_pairs, texts_toks, images, k, model):
    """
    first get the top k based on similarity scores
    then recall the top 1 according to the itm score
    
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1] # nb_texts, k
    # Initialize a tensor to store top 1 indices
    top1_indices = torch.zeros(nb_texts, dtype=torch.long)
    print("top1_indices: {}".format(top1_indices.shape))
    print("topk_indices: {}".format(topk_indices.shape))
    print("scores: {}".format(scores.shape))
    
    assert nb_texts==topk_indices.shape[0]
    assert nb_texts==texts_toks.shape[0]
    assert nb_texts==positive_pairs.shape[0]
    print("recall the top 1 image using itm")
    for i, text_tok in enumerate(tqdm(texts_toks)):
        images_k = images[topk_indices[i]]
        text_tok_expanded = text_tok.unsqueeze(0).expand(k, -1)
        # Compute ITM scores for the top k images

        if i==0:
            print("the shape of the images_k is {}".format(images_k.shape))
            print("the shape of the text_tok_expanded is {}".format(text_tok_expanded.shape))
            print(f"Input text dtype: {text_tok_expanded.dtype}")
            print(f"Input image dtype: {images_k.dtype}")
        attention_mask = (text_tok_expanded > 0).to(text_tok_expanded.dtype)
        autocast = torch.cuda.amp.autocast
        with torch.no_grad(), autocast():
            itm_scores = model.get_itm_score(images_k, text_tok_expanded, attention_mask=attention_mask)
        print("itm_scores")
        print(itm_scores)
        # Get the index of the highest ITM score
        top1_idx = torch.argmax(itm_scores)
        print("top1_idx")
        print(top1_idx)
        # Store the index of the top 1 image in the original image set
        top1_indices[i] = topk_indices[i][top1_idx]
    
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1).cpu()
    # nb_texts,1 , nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(top1_indices, num_classes=nb_images).cpu()
    # a true positive means a positive among the topk
    positive_pairs = positive_pairs.cpu()
    nb_true_positive = (topk_indices_onehot * positive_pairs).sum(dim=(1)).cpu()
    # compute recall at 1
    recall_at_1 = (nb_true_positive / nb_positive)
    return recall_at_1


def batchify(func, X, Y, batch_size, device, Z=None, T=None, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        if Z is not None and T is not None:
            assert func==recall_top1_itm
            z = Z[start:end].to(device)
            t = T.to(device) # t should not be batchified
            result = func(x, y, z, t, *args, **kwargs).cpu()
        else:
            assert func==recall_at_k
            result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)
