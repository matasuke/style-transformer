from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from utils.evaluator import Evaluator
from utils.utils import indice2text, word_drop


def get_lengths(tokens, eos_idx):
    lengths = torch.cumsum(tokens == eos_idx, 1)
    lengths = (lengths == 0).long().sum(-1)
    lengths = lengths + 1  # +1 for <eos> token
    return lengths


def update_factor_rate(factors: Dict, pos_update_rate: Dict, neg_update_rate: Dict):
    # update pos factors:
    for target, update_rate in pos_update_rate.items():
        target = "pos_" + target
        if update_rate is not None and 0 < factors[target] <= 1.0:
            factors[target] = factors[target] * update_rate

    # update neg factors:
    for target, update_rate in neg_update_rate.items():
        target = "neg_" + target
        if update_rate is not None and 0 < factors[target] <= 1.0:
            factors[target] = factors[target] * update_rate

    return factors


def batch_preprocess(batch, pad_idx, eos_idx, split=False):
    batch_pos, batch_neg = batch
    diff = batch_pos.size(1) - batch_neg.size(1)
    if diff < 0:
        pad = torch.full_like(batch_neg[:, :-diff], pad_idx)
        batch_pos = torch.cat((batch_pos, pad), 1)
    elif diff > 0:
        pad = torch.full_like(batch_pos[:, :diff], pad_idx)
        batch_neg = torch.cat((batch_neg, pad), 1)

    pos_styles = torch.ones_like(batch_pos[:, 0])
    neg_styles = torch.zeros_like(batch_neg[:, 0])

    if split:
        pos_lengths = get_lengths(batch_pos, eos_idx)
        neg_lengths = get_lengths(batch_neg, eos_idx)
        return batch_pos, batch_neg, pos_lengths, neg_lengths, pos_styles, neg_styles
    else:
        tokens = torch.cat((batch_pos, batch_neg), 0)
        lengths = get_lengths(tokens, eos_idx)
        styles = torch.cat((pos_styles, neg_styles), 0)

        return tokens, lengths, styles


def d_step(vocab, model_F, model_D, optimizer_D, batch, temperature):
    model_F.eval()
    pad_idx = vocab.stoi["<pad>"]
    eos_idx = vocab.stoi["<eos>"]
    loss_fn = nn.NLLLoss(reduction="none")

    inp_tokens, inp_lengths, raw_styles = batch_preprocess(batch, pad_idx, eos_idx)
    rev_styles = 1 - raw_styles
    batch_size = inp_tokens.size(0)

    with torch.no_grad():
        raw_gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            raw_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )
        rev_gen_log_probs = model_F(
            inp_tokens,
            None,
            inp_lengths,
            rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

    raw_gen_soft_tokens = raw_gen_log_probs.exp()
    raw_gen_lengths = get_lengths(raw_gen_soft_tokens.argmax(-1), eos_idx)

    rev_gen_soft_tokens = rev_gen_log_probs.exp()
    rev_gen_lengths = get_lengths(rev_gen_soft_tokens.argmax(-1), eos_idx)

    if model_D.discriminator_method == "Multi":
        gold_log_probs = model_D(inp_tokens, inp_lengths)
        gold_labels = raw_styles + 1

        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = raw_styles + 1
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)
    else:
        raw_gold_log_probs = model_D(inp_tokens, inp_lengths, raw_styles)
        rev_gold_log_probs = model_D(inp_tokens, inp_lengths, rev_styles)
        gold_log_probs = torch.cat((raw_gold_log_probs, rev_gold_log_probs), 0)
        raw_gold_labels = torch.ones_like(raw_styles)
        rev_gold_labels = torch.zeros_like(rev_styles)
        gold_labels = torch.cat((raw_gold_labels, rev_gold_labels), 0)

        raw_gen_log_probs = model_D(raw_gen_soft_tokens, raw_gen_lengths, raw_styles)
        rev_gen_log_probs = model_D(rev_gen_soft_tokens, rev_gen_lengths, rev_styles)
        gen_log_probs = torch.cat((raw_gen_log_probs, rev_gen_log_probs), 0)
        raw_gen_labels = torch.ones_like(raw_styles)
        rev_gen_labels = torch.zeros_like(rev_styles)
        gen_labels = torch.cat((raw_gen_labels, rev_gen_labels), 0)

    adv_log_probs = torch.cat((gold_log_probs, gen_log_probs), 0)
    adv_labels = torch.cat((gold_labels, gen_labels), 0)
    adv_loss = loss_fn(adv_log_probs, adv_labels)
    assert len(adv_loss.size()) == 1
    adv_loss = adv_loss.sum() / batch_size
    loss = adv_loss

    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()

    model_F.train()

    return adv_loss.item()


def f_step(
    vocab,
    model_F,
    model_D,
    optimizer_F,
    batch,
    temperature,
    drop_decay,
    inp_drop_prob,
    factors,
    cyc_rec_enable=True,
    split=False,
):
    model_D.eval()

    pad_idx = vocab.stoi["<pad>"]
    eos_idx = vocab.stoi["<eos>"]
    loss_fn = nn.NLLLoss(reduction="none")

    def self_reconstruction(
        input_tokens, input_lengths, input_styles, input_mask, self_factor
    ):
        batch_size = input_tokens.size(0)
        optimizer_F.zero_grad()

        # self reconstruction loss
        noise_input_tokens = word_drop(
            input_tokens, input_lengths, inp_drop_prob * drop_decay
        )
        noise_input_lengths = get_lengths(noise_input_tokens, eos_idx)

        slf_log_probs = model_F(
            noise_input_tokens,
            input_tokens,
            noise_input_lengths,
            input_styles,
            generate=False,
            differentiable_decode=False,
            temperature=temperature,
        )

        slf_rec_loss = loss_fn(slf_log_probs.transpose(1, 2), input_tokens) * input_mask
        slf_rec_loss = slf_rec_loss.sum() / batch_size
        slf_rec_loss *= self_factor

        return slf_rec_loss

    def cycle_reconstruction(
        input_tokens,
        input_lengths,
        input_raw_styles,
        input_rev_styles,
        input_mask,
        cycle_factor,
    ):
        batch_size = input_tokens.size(0)

        gen_log_probs = model_F(
            input_tokens,
            None,
            input_lengths,
            input_rev_styles,
            generate=True,
            differentiable_decode=True,
            temperature=temperature,
        )

        gen_soft_tokens = gen_log_probs.exp()
        gen_lengths = get_lengths(gen_soft_tokens.argmax(-1), eos_idx)

        cyc_log_probs = model_F(
            gen_soft_tokens,
            input_tokens,
            gen_lengths,
            input_raw_styles,
            generate=False,
            differentiable_decode=False,
            temperature=temperature,
        )

        cyc_rec_loss = loss_fn(cyc_log_probs.transpose(1, 2), input_tokens) * input_mask
        cyc_rec_loss = cyc_rec_loss.sum() / batch_size
        cyc_rec_loss *= cycle_factor

        return cyc_rec_loss, gen_soft_tokens, gen_lengths

    def style_consistency(gen_soft_tokens, gen_lengths, rev_styles, adv_factor):
        batch_size = gen_soft_tokens.size(0)

        adv_log_porbs = model_D(gen_soft_tokens, gen_lengths, rev_styles)
        if model_D.discriminator_method == "Multi":
            adv_labels = rev_styles + 1
        else:
            adv_labels = torch.ones_like(rev_styles)
        adv_loss = loss_fn(adv_log_porbs, adv_labels)
        adv_loss = adv_loss.sum() / batch_size
        adv_loss *= adv_factor

        return adv_loss

    if split:
        pos_tokens, neg_tokens, pos_lengths, neg_lengths, pos_raw_styles, neg_raw_styles = batch_preprocess(
            batch, pad_idx, eos_idx, split=True
        )
        pos_rev_styles = 1 - pos_raw_styles
        neg_rev_styles = 1 - neg_raw_styles

        assert pos_tokens.size(0) == neg_tokens.size(0)
        pos_token_mask = (pos_tokens != pad_idx).float()
        neg_token_mask = (neg_tokens != pad_idx).float()

        # pos reconstruction
        pos_slf_rec_loss = self_reconstruction(
            pos_tokens,
            pos_lengths,
            pos_raw_styles,
            pos_token_mask,
            factors["pos_self_factor"],
        )
        pos_slf_rec_loss.backward()

        neg_slf_rec_loss = self_reconstruction(
            neg_tokens,
            neg_lengths,
            neg_raw_styles,
            neg_token_mask,
            factors["neg_self_factor"],
        )
        neg_slf_rec_loss.backward()

        # cycle consistency loss
        if not cyc_rec_enable:
            optimizer_F.step()
            model_D.train()
            slf_rec_loss = (pos_slf_rec_loss + neg_slf_rec_loss) / 2
            return slf_rec_loss, 0, 0

        pos_cyc_rec_loss, pos_gen_soft_tokens, pos_gen_lengths = cycle_reconstruction(
            pos_tokens,
            pos_lengths,
            pos_raw_styles,
            pos_rev_styles,
            pos_token_mask,
            factors["pos_cycle_factor"],
        )
        neg_cyc_rec_loss, neg_gen_soft_tokens, neg_gen_lengths = cycle_reconstruction(
            neg_tokens,
            neg_lengths,
            neg_raw_styles,
            neg_rev_styles,
            neg_token_mask,
            factors["neg_cycle_factor"],
        )

        # style consistency loss
        pos_adv_loss = style_consistency(
            pos_gen_soft_tokens,
            pos_gen_lengths,
            pos_rev_styles,
            factors["pos_adv_factor"],
        )
        neg_adv_loss = style_consistency(
            neg_gen_soft_tokens,
            neg_gen_lengths,
            neg_rev_styles,
            factors["neg_adv_factor"],
        )

        (pos_cyc_rec_loss + pos_adv_loss).backward()
        (neg_cyc_rec_loss + neg_adv_loss).backward()

        # update parameters
        optimizer_F.step()

        slf_rec_loss = pos_slf_rec_loss + neg_slf_rec_loss
        cyc_rec_loss = pos_cyc_rec_loss + neg_cyc_rec_loss
        adv_loss = pos_adv_loss + neg_adv_loss
    else:
        inp_tokens, inp_lengths, raw_styles = batch_preprocess(
            batch, pad_idx, eos_idx, split=False
        )
        rev_styles = 1 - raw_styles
        token_mask = (inp_tokens != pad_idx).float()

        # self reconstruction
        slf_rec_loss = self_reconstruction(
            inp_tokens, inp_lengths, raw_styles, token_mask, factors["self_factor"]
        )
        slf_rec_loss.backward()

        # cycle consistency loss
        if not cyc_rec_enable:
            optimizer_F.step()
            model_D.train()
            return slf_rec_loss.item(), 0, 0

        cyc_rec_loss, gen_soft_tokens, gen_lengths = cycle_reconstruction(
            inp_tokens,
            inp_lengths,
            raw_styles,
            rev_styles,
            token_mask,
            factors["cycle_factor"],
        )

        # style consistency loss
        adv_loss = style_consistency(
            gen_soft_tokens, gen_lengths, rev_styles, factors["adv_factor"]
        )

        (cyc_rec_loss + adv_loss).backward()

        # update parameters
        optimizer_F.step()

    model_D.train()

    return slf_rec_loss.item(), cyc_rec_loss.item(), adv_loss.item()


def train(
    config, vocab, model_F, model_D, opt_F, opt_D, train_iters, test_iters, evaluator
):
    his_d_adv_loss = []
    his_f_slf_loss = []
    his_f_cyc_loss = []
    his_f_adv_loss = []

    global_step = 0
    model_F.train()
    model_D.train()

    trainer_cfg = config["trainer"]
    pretrain_generator_steps = trainer_cfg["pretrain_generator_steps"]
    generator_steps = trainer_cfg["generator_steps"]
    discriminator_steps = trainer_cfg["discriminator_steps"]
    log_step = trainer_cfg["log_step"]
    evaluation_step = trainer_cfg["evaluation_step"]
    drop_rate_config = trainer_cfg["drop_rate_config"]
    temperature_config = trainer_cfg["temperature_config"]
    word_drop_prob = trainer_cfg["word_drop_prob"]
    factors = trainer_cfg["factors"]

    (config.save_dir / "ckpts").mkdir(parents=True)
    print("Save Path:", config.save_dir)

    print("Model F pretraining......")
    for i, batch in enumerate(train_iters):
        if i >= pretrain_generator_steps:
            break
        slf_loss, cyc_loss, _ = f_step(
            vocab=vocab,
            model_F=model_F,
            model_D=model_D,
            optimizer_F=opt_F,
            batch=batch,
            temperature=1.0,
            drop_decay=1.0,
            inp_drop_prob=word_drop_prob,
            factors=factors,
            cyc_rec_enable=False,
            split=trainer_cfg["split_pretrain_data"],
        )
        his_f_slf_loss.append(slf_loss)
        his_f_cyc_loss.append(cyc_loss)

        if (i + 1) % 10 == 0:
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            his_f_slf_loss = []
            his_f_cyc_loss = []
            print(
                "[iter: {}] slf_loss:{:.4f}, rec_loss:{:.4f}".format(
                    i + 1, avrg_f_slf_loss, avrg_f_cyc_loss
                )
            )

        if trainer_cfg["split_pretrain_data"]:
            factors = update_factor_rate(
                factors, trainer_cfg["pos_update_rate"], trainer_cfg["neg_update_rate"]
            )

    print("Training start......")
    def calc_temperature(temperature_config, step):
        num = len(temperature_config)
        for i in range(num):
            t_a, s_a = temperature_config[i]
            if i == num - 1:
                return t_a
            t_b, s_b = temperature_config[i + 1]
            if s_a <= step < s_b:
                k = (step - s_a) / (s_b - s_a)
                temperature = (1 - k) * t_a + k * t_b
                return temperature

    batch_iters = iter(train_iters)
    while True:
        drop_decay = calc_temperature(drop_rate_config, global_step)
        temperature = calc_temperature(temperature_config, global_step)
        batch = next(batch_iters)

        for _ in range(discriminator_steps):
            batch = next(batch_iters)
            d_adv_loss = d_step(vocab, model_F, model_D, opt_D, batch, temperature)
            his_d_adv_loss.append(d_adv_loss)

        for _ in range(generator_steps):
            batch = next(batch_iters)
            f_slf_loss, f_cyc_loss, f_adv_loss = f_step(
                vocab=vocab,
                model_F=model_F,
                model_D=model_D,
                optimizer_F=opt_F,
                batch=batch,
                temperature=temperature,
                drop_decay=drop_decay,
                inp_drop_prob=word_drop_prob,
                factors=factors,
                cyc_rec_enable=True,
                split=trainer_cfg["split_train_data"],
            )
            his_f_slf_loss.append(f_slf_loss)
            his_f_cyc_loss.append(f_cyc_loss)
            his_f_adv_loss.append(f_adv_loss)

        if trainer_cfg["split_train_data"]:
            factors = update_factor_rate(
                factors, trainer_cfg["pos_update_rate"], trainer_cfg["neg_update_rate"]
            )

        global_step += 1

        if global_step % log_step == 0:
            avrg_d_adv_loss = np.mean(his_d_adv_loss)
            avrg_f_slf_loss = np.mean(his_f_slf_loss)
            avrg_f_cyc_loss = np.mean(his_f_cyc_loss)
            avrg_f_adv_loss = np.mean(his_f_adv_loss)
            log_str = (
                "[iter {}] d_adv_loss: {:.4f}  "
                + "f_slf_loss: {:.4f}  f_cyc_loss: {:.4f}  "
                + "f_adv_loss: {:.4f}  temp: {:.4f}  drop: {:.4f}"
            )
            print(
                log_str.format(
                    global_step,
                    avrg_d_adv_loss,
                    avrg_f_slf_loss,
                    avrg_f_cyc_loss,
                    avrg_f_adv_loss,
                    temperature,
                    word_drop_prob * drop_decay,
                )
            )

        if global_step % evaluation_step == 0:
            his_d_adv_loss = []
            his_f_slf_loss = []
            his_f_cyc_loss = []
            his_f_adv_loss = []

            # save model
            torch.save(
                model_F.state_dict(),
                config.save_dir / "ckpts" / f"{str(global_step)}_F.pth",
            )
            torch.save(
                model_D.state_dict(),
                config.save_dir / "ckpts" / f"{str(global_step)}_D.pth",
            )
            auto_eval(
                save_dir=config.test_dir,
                evaluator=evaluator,
                vocab=vocab,
                model_F=model_F,
                test_iters=test_iters,
                global_step=global_step,
                temperature=temperature,
            )


def auto_eval(
    save_dir, evaluator, vocab, model_F, test_iters, global_step, temperature
) -> Tuple[List[str], List[str], List[str]]:
    model_F.eval()
    eos_idx = vocab.stoi["<eos>"]

    def inference(data_iter, raw_style):
        gold_text = []
        raw_output = []
        rev_output = []
        for batch in data_iter:
            inp_tokens = batch.text
            inp_lengths = get_lengths(inp_tokens, eos_idx)
            raw_styles = torch.full_like(inp_tokens[:, 0], raw_style)
            rev_styles = 1 - raw_styles

            with torch.no_grad():
                raw_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    raw_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )

            with torch.no_grad():
                rev_log_probs = model_F(
                    inp_tokens,
                    None,
                    inp_lengths,
                    rev_styles,
                    generate=True,
                    differentiable_decode=False,
                    temperature=temperature,
                )

            gold_text += indice2text(inp_tokens.cpu(), vocab)
            raw_output += indice2text(raw_log_probs.argmax(-1).cpu(), vocab)
            rev_output += indice2text(rev_log_probs.argmax(-1).cpu(), vocab)

        return gold_text, raw_output, rev_output

    pos_iter = test_iters.pos_iter
    neg_iter = test_iters.neg_iter

    gold_text, raw_output, rev_output = zip(
        inference(neg_iter, 0), inference(pos_iter, 1)
    )

    style_labels = evaluator.style_labels
    acc_neg = evaluator.get_style_accuracy(rev_output[0], style_labels[1])
    acc_pos = evaluator.get_style_accuracy(rev_output[1], style_labels[0])
    bleu_neg = evaluator.get_self_bleu_score(gold_text[0], rev_output[0])
    bleu_pos = evaluator.get_self_bleu_score(gold_text[1], rev_output[1])
    ppl_neg = evaluator.get_perplexity(rev_output[0])
    ppl_pos = evaluator.get_perplexity(rev_output[1])

    for k in range(5):
        idx = np.random.randint(len(rev_output[0]))
        print("*" * 20, "neg sample", "*" * 20)
        print("[gold]", gold_text[0][idx])
        print("[raw ]", raw_output[0][idx])
        print("[rev ]", rev_output[0][idx])
        if evaluator.reference:
            print("[ref ]", evaluator.reference[style_labels[1]][idx])

    print("*" * 20, "********", "*" * 20)

    for k in range(5):
        idx = np.random.randint(len(rev_output[1]))
        print("*" * 20, "pos sample", "*" * 20)
        print("[gold]", gold_text[1][idx])
        print("[raw ]", raw_output[1][idx])
        print("[rev ]", rev_output[1][idx])
        if evaluator.reference:
            print("[ref ]", evaluator.reference[style_labels[0]][idx])

    print("*" * 20, "********", "*" * 20)

    print(
        (
            "[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} "
            + "bleu_pos: {:.4f} bleu_neg: {:.4f} "
            + "ppl_pos: {:.4f} ppl_neg: {:.4f}\n"
        ).format(acc_pos, acc_neg, bleu_pos, bleu_neg, ppl_pos, ppl_neg)
    )

    # save output
    save_file = save_dir / f"{str(global_step)}.txt"
    eval_log_file = save_dir / "eval_log.txt"
    with open(eval_log_file, "a") as fl:
        print(
            (
                "iter{:5d}:  acc_pos: {:.4f} acc_neg: {:.4f} "
                + "bleu_pos: {:.4f} bleu_neg: {:.4f} "
                + "ppl_pos: {:.4f} ppl_neg: {:.4f}\n"
            ).format(
                global_step, acc_pos, acc_neg, bleu_pos, bleu_neg, ppl_pos, ppl_neg
            ),
            file=fl,
        )
    with open(save_file, "w") as fw:
        print(
            (
                "[auto_eval] acc_pos: {:.4f} acc_neg: {:.4f} "
                + "bleu_pos: {:.4f} bleu_neg: {:.4f} "
                + "ppl_pos: {:.4f} ppl_neg: {:.4f}\n"
            ).format(acc_pos, acc_neg, bleu_pos, bleu_neg, ppl_pos, ppl_neg),
            file=fw,
        )

        for idx in range(len(rev_output[0])):
            print("*" * 20, "neg sample", "*" * 20, file=fw)
            print("[gold]", gold_text[0][idx], file=fw)
            print("[raw ]", raw_output[0][idx], file=fw)
            print("[rev ]", rev_output[0][idx], file=fw)
            if evaluator.reference:
                print("[ref ]", evaluator.reference[style_labels[1]][idx], file=fw)

        print("*" * 20, "********", "*" * 20, file=fw)

        for idx in range(len(rev_output[1])):
            print("*" * 20, "pos sample", "*" * 20, file=fw)
            print("[gold]", gold_text[1][idx], file=fw)
            print("[raw ]", raw_output[1][idx], file=fw)
            print("[rev ]", rev_output[1][idx], file=fw)
            if evaluator.reference:
                print("[ref ]", evaluator.reference[style_labels[0]][idx], file=fw)

        print("*" * 20, "********", "*" * 20, file=fw)

    model_F.train()
