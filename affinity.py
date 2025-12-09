import torch
import torch.nn.functional as F
import numpy as np


def compute_affinities(sentences, model, tokenizer, cxn_list, device=None, standard_bert=False):
    """
    Args:
        sentences: list[str]
            Each sentence must contain the token 'let' and the token 'alone'
            (as single tokenizer tokens).
        model: a masked language model (e.g. BertForMaskedLM)
        tokenizer: its tokenizer
        device: torch.device or str (optional). If None, will use model's device.

    Returns:
        prob_let: list[float]
        jsd_let_wo_alone: list[float]
        prob_alone: list[float]
        jsd_alone_wo_let: list[float]

        where:
        - prob_let[i] = P("let" | sentence_i with only "let" masked)
        - jsd_let_wo_alone[i] = JSD( P(. | let-masked-only) || P(. | let+alone-masked) )
          at the position of "let" for sentence i
        - prob_alone[i] = P("alone" | sentence_i with only "alone" masked)
        - jsd_alone_wo_let[i] = JSD( P(. | alone-masked-only) || P(. | let+alone-masked) )
          at the position of "alone" for sentence i
    """

    sentences = [s.split(".")[0] + "." for s in sentences]

    #print(sentences)
    #print("LENGTH OF INPUT SENTS:",len(sentences))

    if device is None:
        # Try to infer model device
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()

    # Token IDs for the words of interest and [MASK]
    if not standard_bert:
        let_id = tokenizer.convert_tokens_to_ids("Ġlet")
        #print(let_id)
        alone_id = tokenizer.convert_tokens_to_ids("Ġalone")
        much_id = tokenizer.convert_tokens_to_ids("Ġmuch")
        less_id = tokenizer.convert_tokens_to_ids("Ġless")
        not_id = tokenizer.convert_tokens_to_ids("Ġnot")
        mention_id = tokenizer.convert_tokens_to_ids("Ġmention")
        never_id = tokenizer.convert_tokens_to_ids("Ġnever")
        mind_id = tokenizer.convert_tokens_to_ids("Ġmind")
        or_id = tokenizer.convert_tokens_to_ids("Ġor")
    else:
        let_id = tokenizer.convert_tokens_to_ids("let")
        alone_id = tokenizer.convert_tokens_to_ids("alone")
        much_id = tokenizer.convert_tokens_to_ids("much")
        less_id = tokenizer.convert_tokens_to_ids("less")
        not_id = tokenizer.convert_tokens_to_ids("not")
        mention_id = tokenizer.convert_tokens_to_ids("mention")
        never_id = tokenizer.convert_tokens_to_ids("never")
        mind_id = tokenizer.convert_tokens_to_ids("mind")
        or_id = tokenizer.convert_tokens_to_ids("or")


    mask_id = tokenizer.mask_token_id

    if let_id is None or alone_id is None or mask_id is None:
        raise ValueError("Could not get token ids for 'let', 'alone' or mask token.")

    # Tokenize batch
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    input_ids = enc["input_ids"]
    #print(input_ids[0])# [B, L]

    #print(tokenizer.convert_ids_to_tokens(input_ids[0]))
    attention_mask = enc["attention_mask"]

    batch_size, seq_len = input_ids.shape

    # Find the FIRST occurrence of 'let' and 'alone' in each sentence
    let_positions = []
    alone_positions = []

    prob_let = []
    jsd_let_wo_alone = []
    prob_alone = []
    jsd_alone_wo_let = []

    t1_ids = []
    t2_ids = []

    for cxn in cxn_list:
        if cxn == "letalone":
            t1_ids.append(let_id)
            t2_ids.append(alone_id)

        elif cxn == "muchless":
            t1_ids.append(much_id)
            t2_ids.append(less_id)

        elif cxn == "nottomention":
            t1_id = not_id
            t2_id = mention_id
            t1_ids.append(not_id)
            t2_ids.append(mention_id)

        elif cxn == "nevermind":
            t1_id = never_id
            t2_id = mind_id
            t1_ids.append(never_id)
            t2_ids.append(mind_id)

        elif cxn == "or":
            t1_ids.append(or_id)
            t2_ids.append(or_id)

        else:
            raise ValueError(f"Unknown cxn type: {cxn}")

    # print(let_id, alone_id, much_id, less_id, or_id)
    # print("T1s:", t1_ids)
    # print("T2s:", t2_ids)
    for i in range(batch_size):
        ids = input_ids[i].tolist()
        t1_id = t1_ids[i]
        t2_id = t2_ids[i]

        #print("IDS:",ids)
        try:
            let_pos = ids.index(t1_id)
        except ValueError:
            raise ValueError(f"target1 token not found in sentence {i}: {sentences[i]}")
        try:
            alone_pos = ids.index(t2_id)
        except ValueError:
            raise ValueError(f"target2 token not found in sentence {i}: {sentences[i]}")

        let_positions.append(let_pos)
        alone_positions.append(alone_pos)

    let_positions = torch.tensor(let_positions, dtype=torch.long)
    alone_positions = torch.tensor(alone_positions, dtype=torch.long)
    len(let_positions)
    # print("LET POSITIONS", let_positions)

    # Construct masked versions
    let_masked_ids = input_ids.clone()
    alone_masked_ids = input_ids.clone()
    both_masked_ids = input_ids.clone()

    for i in range(batch_size):
        # print(input_ids[i])
        # print(let_positions[i])
        let_masked_ids[i, let_positions[i]] = mask_id
        alone_masked_ids[i, alone_positions[i]] = mask_id
        both_masked_ids[i, let_positions[i]] = mask_id
        both_masked_ids[i, alone_positions[i]] = mask_id

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    let_masked_ids = let_masked_ids.to(device)
    alone_masked_ids = alone_masked_ids.to(device)
    both_masked_ids = both_masked_ids.to(device)

    # print("LET MASKED IDS", let_masked_ids[0])
    # print(tokenizer.convert_ids_to_tokens(let_masked_ids[0].tolist()))


    # Helper: Jensen-Shannon divergence between two distributions (torch tensors)
    def jsd(p, q):
        # p, q: [V], already softmaxed
        m = 0.5 * (p + q)
        js = 0.5 * (
            F.kl_div(p.log(), m, reduction="sum") +
            F.kl_div(q.log(), m, reduction="sum")
        )
        return js.item()

    with torch.no_grad():
        # 3 forward passes, each on the full batch
        logits_let = model(input_ids=let_masked_ids, attention_mask=attention_mask).logits   # [B, L, V]
        logits_alone = model(input_ids=alone_masked_ids, attention_mask=attention_mask).logits
        logits_both = model(input_ids=both_masked_ids, attention_mask=attention_mask).logits

    vocab_size = logits_let.size(-1)

    # print(F.softmax(logits_let[0,9], dim=-1)[1339])


    for i in range(batch_size):
        t1_id = t1_ids[i]
        t2_id = t2_ids[i]
        # print(i)
        # print(F.softmax(logits_let[i, 9], dim=-1)[1339])
        lp = let_positions[i].item()
        ap = alone_positions[i].item()
        # print("LET Position:", lp)
        # print(t1_ids[i])
        # print(mask_id)
        # print(let_masked_ids[i, lp])
        # print(F.softmax(logits_let[i, lp], dim=-1)[1339])

        # print("POS", let_positions[i])
        # print("POS", let_positions[i].item())

        # ---- For "let" target ----
        # Distribution when only "let" is masked
        let_logits_only = logits_let[i, lp, :]  # [V]
        # print(F.softmax(logits_let[i, lp, :], dim=-1)[1339])

        # print(let_logits_only.shape)
        p_let_only = F.softmax(let_logits_only, dim=-1)  # [V]
        # print(F.softmax(let_logits_only, dim=-1)[1339])
        #
        # print("BEST PREDICTION", np.argmax(p_let_only.detach().cpu().numpy()))
        # print(t1_id)


        # Distribution when both "let" and "alone" are masked (at let position)
        let_logits_both = logits_both[i, lp, :]
        p_let_both = F.softmax(let_logits_both, dim=-1)

        # P("let" | only let masked)
        prob_let_token = p_let_only[t1_id].item()
        # print("PROB SAVED", prob_let_token)
        # print("PROB RAW", p_let_only)
        prob_let.append(prob_let_token)

        # JSD between full distributions
        jsd_let_wo_alone.append(jsd(p_let_only, p_let_both))

        # ---- For "alone" target ----
        # Distribution when only "alone" is masked
        alone_logits_only = logits_alone[i, ap, :]
        p_alone_only = F.softmax(alone_logits_only, dim=-1)

        # Distribution when both are masked (at alone position)
        alone_logits_both = logits_both[i, ap, :]
        p_alone_both = F.softmax(alone_logits_both, dim=-1)

        # P("alone" | only alone masked)
        prob_alone_token = p_alone_only[t2_id].item()
        prob_alone.append(prob_alone_token)

        # JSD between full distributions
        jsd_alone_wo_let.append(jsd(p_alone_only, p_alone_both))

    return prob_let, jsd_let_wo_alone, prob_alone, jsd_alone_wo_let
