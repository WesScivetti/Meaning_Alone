from argparse import ArgumentParser
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from minicons import scorer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import os


def evaluate_blimp_model(model_name: str, causallm: bool = False, revision: str = "", batch_size: int = 32):
    """
    evaluates the target model / revision on the BLiMP dataset. For each model:
        Read in each jsonl file in the data/BLiMP directory
        Read good_bad sentences
        For each sentence pair, compute the log-probabilities of each sentence
        Create an output file BLiMP in the output directory
        File will contain columns: sentence good, sentence bad, phenomenon, log_prob_good, log_prob_bad
        If using a maskedLM, use psuedo log likelihood
    Args:
        model_name: name of Hugging Face model to load.
        causallm: if True, use causal LM mode with minicons.
        revision: optional revision of the model to load.
        batch_size: batch size to use for evaluation.
    """
    # uids = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive']

    uids = ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive',
              'animate_subject_trans', 'causative', 'complex_NP_island',
              'coordinate_structure_constraint_complex_left_branch',
              'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1',
              'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1',
              'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2',
              'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2',
              'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun',
              'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2',
              'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2',
              'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive',
              'irregular_past_participle_adjectives', 'irregular_past_participle_verbs',
              'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2',
              'left_branch_island_echo_question', 'left_branch_island_simple_question',
              'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present',
              'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1',
              'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3',
              'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1',
              'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present',
              'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1',
              'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island',
              'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance',
              'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap',
              'wh_vs_that_with_gap_long_distance']

    device = "cuda" if torch.cuda.is_available() else "cpu"



    if causallm:
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        dev = next(model.parameters()).device
        lm_scorer = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device=device)

    else:
        if revision:
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(device)
        dev = next(model.parameters()).device
        lm_scorer = scorer.MaskedLMScorer(model, tokenizer=tokenizer, device=device)


    print(f"Model is on device: {dev}")
    print(lm_scorer)


    output_rows = []

    for uid in uids:
        print("Evaluating phenomenon:", uid)
        ds = load_dataset("nyu-mll/blimp", uid)
        stimuli = []

        for row in ds["train"]:
            stimuli.append([row["sentence_good"], row["sentence_bad"], uid])


        stimuli_dl = torch.utils.data.DataLoader(stimuli, batch_size=batch_size)

        # wrap in tqdm for progress bar
        for batch in tqdm(stimuli_dl, desc=f"Processing {uid}"):
            good, bad, phenomenon = batch

            good_scores = lm_scorer.sequence_score(good, reduction=lambda x: x.sum(0))
            bad_scores = lm_scorer.sequence_score(bad, reduction=lambda x: x.sum(0))

            for g_sent, b_sent, phen, g_score, b_score in zip(good, bad, phenomenon, good_scores, bad_scores):
                output_rows.append({
                    "sentence_good": g_sent,
                    "sentence_bad": b_sent,
                    "phenomenon": phen,
                    "log_prob_good": g_score.item(),
                    "log_prob_bad": b_score.item()
                })



    output_df = pd.DataFrame(output_rows)

    return output_df

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--model_name", type=str, required=True, help="Hugging Face model name")
    argparse.add_argument("--output_dir",type=str, required=False, help="Path to output directory", default="outputs")
    argparse.add_argument("--causallm", action="store_true", help="Whether to use causal LM mode")
    argparse.add_argument("--revision", type=str, default="", help="Optional model revision to load")
    argparse.add_argument("--batch_size", type=int, default=64, help="Batch size to use")
    args = argparse.parse_args()

    output_df = evaluate_blimp_model(
        model_name=args.model_name,
        causallm=args.causallm,
        revision=args.revision,
        batch_size=args.batch_size
    )

    print(torch.cuda.memory_summary(device=None, abbreviated=False))

    print("Done")

    #create full output path with outputs/model_name/revision
    if args.revision:
        revision_name = args.revision
    else:
        revision_name = "NONE"
    output_path = f"{args.output_dir}/{args.model_name}/{revision_name}/"

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "blimp_all.tsv")
    output_df.to_csv(output_file, sep="\t", index=False)
    print(f"Saved results to {output_file}")

