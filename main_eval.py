import pandas as pd
import os
from eval import print_results_for_data_file_accuracy, print_results_for_data_file_syntax_accuracy
import re
from collections import defaultdict
from tqdm import tqdm
from eval_blimp import print_results_blimp_comps, print_results_ewok
from argparse import ArgumentParser
from utils import combine_two_dfs
import numpy as np

ewok_domains = ['social-relations', 'agent-properties', 'material-properties', 'spatial-relations', 'material-dynamics', 'quantitative-properties', 'social-interactions', 'physical-interactions', 'physical-dynamics', 'physical-relations', 'social-properties']

# models = [
#     "jhu-clsp/ettin-encoder-400m",
#     "jhu-clsp/ettin-decoder-400m",
#     "jhu-clsp/ettin-encoder-1b",
#     "jhu-clsp/ettin-decoder-1b"
# ]

models2 = [
    "jhu-clsp/ettin-encoder-400m"
]

# revisions = [
#     "step2999",
#     "step8994",
#     "step20986",
#     "step35974",
#     "step50960",
#     "step104916",
#     "step119903"
# ]

revisions = [
    "step2999",
    "step8994",
    "step20986",
    "step26982",
    "step29979",
    "step32976",
    "step35974",
    "step38972",
    "step41969",
    "step44967",
    "step47963",
    "step50960",
    "step53957",
    "step56955",
    "step59953",
    "step62950",
    "step65948",
    "step104916",
    "step119903"
]

all_revisions = [
    "step2999",
    "step5996",
    "step8994",
    "step11992",
    "step14991",
    "step17988",
    "step20986",
    "step23984",
    "step26982",
    "step29979",
    "step32976",
    "step35974",
    "step38972",
    "step41969",
    "step44967",
    "step47963",
    "step50960",
    "step53957",
    "step56955",
    "step59953",
    "step62950",
    "step65948",
    "step68944",
    "step71942",
    "step74940",
    "step77938",
    "step80935",
    "step83933",
    "step86931",
    "step89929",
    "step92926",
    "step95924",
    "step98921",
    "step101918",
    "step104916",
    "step107913",
    "step110911",
    "step113909",
    "step116906",
    "step119903",
    "step122900",
    "step125897",
    "step128895",
    "step131892",
    "step134889",
    "step137887",
    "step140884",
    "step143882",
    "step146880",
    "step149879",
]

def eval_syntax(models):
    for m in models:
        rows = []
        print("Evaluating model:", m)
        for r in tqdm(all_revisions):
            syn_file = os.path.join("outputs",m,r, "syntactic_tests_aligned.tsv")
            fixed_file = os.path.join("outputs",m,r, "syntactic_tests_fixed.tsv")
            syn_df = pd.read_csv(syn_file, sep="\t")
            fixed_df = pd.read_csv(fixed_file, sep="\t")
            syn_df = combine_two_dfs(syn_df, fixed_df)
            for cxn in ["let alone", "much less", "not to mention","never mind"]:
                print("Evaluating model:", m, "revision:", r, "construction:", cxn)
                scores, delta_surps, delta_surps_and = print_results_for_data_file_syntax_accuracy(syn_df, cxn)
                for test_type in ["NPI", "Pseudocleft", "CP Conjunction", "VP Conjunction", "VP Gap Conjunction", "Avg_Test", "Avg_Control"]:
                    row = [r, cxn, test_type, float(scores[test_type]), float(np.mean(delta_surps[test_type])), float(np.mean(delta_surps_and[test_type]))]
                    rows.append(row)


        output_dir = os.path.join("outputs_summary", m)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "accuracy_summary_syntax.tsv")
        res_df = pd.DataFrame(rows, columns=["revision", "construction", "test_type", "accuracy", "delta_surps", "delta_surps_and"])
        res_df.to_csv(output_file, sep="\t", index=False)



def eval_ettin():

    for m in models:
        res_model = defaultdict(dict)
        print("Evaluating model:", m)
        #wrap revisions in tqdm
        for r in tqdm(all_revisions, desc=f"Processing revisions for model {m}"):
            try:
                la_file = os.path.join("outputs",m,r, "combined_easy.tsv")
                #print(la_file)
                #check if file exists
                if not os.path.exists(la_file):
                    raise FileNotFoundError
            except:
                print("Missing file:", os.path.join("outputs",m,r, "combined_easy.tsv"))

            try:
                blimp_file = os.path.join("outputs",m,r, "blimp_all.tsv")
                if not os.path.exists(blimp_file):
                    raise FileNotFoundError
            except:
                print("Missing file:", os.path.join("outputs",m,r, "blimp_all.tsv"))

            try:
                comps_file = os.path.join("outputs",m,r, "comps_base.tsv")
                if not os.path.exists(comps_file):
                    raise FileNotFoundError
            except:
                print("Missing file:", os.path.join("outputs",m,r, "comps_base.tsv"))

            try:
                ewok_file = os.path.join("outputs",m,r, "ewok_core.tsv")
                if not os.path.exists(ewok_file):
                    raise FileNotFoundError
            except:
                print("Missing file:", os.path.join("outputs",m,r, "ewok_core.tsv"))



    for m in models:
        res_model = defaultdict(dict)
        print("Evaluating model:", m)
        #wrap revisions in tqdm
        for r in tqdm(all_revisions, desc=f"Processing revisions for model {m}"):
            la_file = os.path.join("outputs",m,r, "combined_easy.tsv")
            blimp_file = os.path.join("outputs",m,r, "blimp_all.tsv")
            comps_file = os.path.join("outputs",m,r, "comps_base.tsv")
            ewok_file = os.path.join("outputs",m,r, "ewok_core.tsv")
            if re.search("encoder", m):
                causal=False
                log_probs=False
            else:
                causal=True
                log_probs=True


            res = defaultdict(dict)

            plain_verbs = [
                 'hold',
                 'bake',
                 'cook',
                 'paint',
                 'draw',
                 'sing',
                 'write',
                 'make',
                 'pick up',
                 'lift',
                 'see',
                 'compose'
            ]


            for cxn in ["letalone", "muchless", "nottomention","nevermind"]:
                print("Evaluating model:", m, "revision:", r, "construction:", cxn)
                print("Full data")
                acc_full = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, cxn=cxn, print_output=True)
                #print("Aligned data")
                acc_al = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="aligned",
                                                                cxn=cxn, print_output=False)
                #print("Misaligned data")
                acc_mis = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="misaligned",
                                                                cxn=cxn, print_output=False)
                print("aligned1 and aligned2")
                acc_al1 = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="aligned1",
                                                                cxn=cxn, print_output=True)
                acc_al2 = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="aligned2",
                                                                cxn=cxn, print_output=True)
                print("Misaligned data 1 and 2")
                acc_mis1 = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="misaligned1",
                                                                cxn=cxn, print_output=True)
                #print("Misaligned data")
                acc_mis2 = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="misaligned2",
                                                                cxn=cxn, print_output=True)

                res[cxn]["all"] = acc_full
                res[cxn]["aligned"] = acc_al
                res[cxn]["misaligned"] = acc_mis
                res[cxn]["aligned1"] = acc_al1
                res[cxn]["misaligned1"] = acc_mis1
                res[cxn]["aligned2"] = acc_al2
                res[cxn]["misaligned2"] = acc_mis2

                for v in plain_verbs:
                    #print("Evaluating model:", m, "revision:", r, "construction:", cxn, "verb:", v)
                    acc_v_all = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, verb=v, cxn=cxn, print_output=False)
                    acc_v_al = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="aligned",
                                                                    verb=v, cxn=cxn, print_output=False)
                    acc_v_mis = print_results_for_data_file_accuracy(la_file, log_probs=log_probs, world_knowledge_filter="misaligned",
                                                                    verb=v, cxn=cxn, print_output=False)
                    res[cxn][f"verb_{v}_all"] = acc_v_all
                    res[cxn][f"verb_{v}_aligned"] = acc_v_al
                    res[cxn][f"verb_{v}_misaligned"] = acc_v_mis


            print("Evaluating BLIMP")
            acc_blimp = print_results_blimp_comps(blimp_file, print_output=True)
            res["blimp"]["all"] = acc_blimp
            print("Evaluating COMPS")
            acc_comps = print_results_blimp_comps(comps_file, print_output=True)
            res["comps"]["all"] = acc_comps
            print("Evaluating EWOK")
            acc_ewok = print_results_ewok(ewok_file, print_output=True)
            res["ewok"]["all"] = acc_ewok
            print("------------------------------")
            for domain in ewok_domains:
                print("domain")
                acc_domain = print_results_ewok(ewok_file, print_output=True, domain=domain)
                res["ewok"][f"domain_{domain}"] = acc_domain
            print("domains done")

            res_model[r] = res

        #create dataframe from res_model, and save it to tsv
        #columns are revision, construction, world_knowledge_filter, accuracy
        rows = []
        for rev in res_model:
            for cxn in res_model[rev]:
                for wkf in res_model[rev][cxn]:
                    acc = res_model[rev][cxn][wkf]
                    rows.append({
                        "revision": rev,
                        "construction": cxn,
                        "world_knowledge_filter": wkf,
                        "accuracy": acc
                    })

        res_df = pd.DataFrame(rows)
        output_dir = os.path.join("outputs_summary", m)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "accuracy_summary_la.tsv")
        res_df.to_csv(output_file, sep="\t", index=False)

def eval_olmo():
    olmo_revisions = [
        "main",
        "NONE",
      "stage1-step477000-tokens2001B",
      "stage1-step900-tokens4B",
      "stage1-step2000-tokens9B",
      "stage1-step4000-tokens17B",
      "stage1-step8000-tokens34B",
      "stage1-step17000-tokens72B",
      "stage1-step36000-tokens151B",
      "stage1-step72000-tokens302B",
      "stage1-step151000-tokens634B"
    ]
    olmo_revisions2 = ["main"]
    m = 'allenai/OLMo-2-1124-7B'
    res_model = defaultdict(dict)
    print("Evaluating model:", "olmo")
    # wrap revisions in tqdm
    for r in tqdm(olmo_revisions, desc=f"Processing revisions for model olmo!"):
        la_file = os.path.join("outputs", m, r, "combined_easy.tsv")
        # blimp_file = os.path.join("outputs", m, r, "blimp_all.tsv")
        # comps_file = os.path.join("outputs", m, r, "comps_base.tsv")
        # ewok_file = os.path.join("outputs", m, r, "ewok_core.tsv")
        causal = True
        res = defaultdict(dict)

        plain_verbs = [
            'hold',
            'bake',
            'cook',
            'paint',
            'draw',
            'sing',
            'write',
            'make',
            'pick up',
            'lift',
            'see',
            'compose'
        ]

        for cxn in ["letalone", "muchless", "nottomention", "nevermind"]:
            print("Evaluating model:", m, "revision:", r, "construction:", cxn)
            print("Full data")
            acc_full = print_results_for_data_file_accuracy(la_file, log_probs=True, cxn=cxn,
                                                            print_output=True)
            print("Aligned data")
            acc_al = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                          world_knowledge_filter="aligned",
                                                          cxn=cxn, print_output=True)

            print("Aligned data 1 and 2")
            acc_al1 = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                          world_knowledge_filter="aligned1",
                                                          cxn=cxn, print_output=True)
            acc_al2 = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                          world_knowledge_filter="aligned2",
                                                          cxn=cxn, print_output=True)
            print("Misaligned data")
            acc_mis = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                           world_knowledge_filter="misaligned",
                                                           cxn=cxn, print_output=True)
            acc_mis1 = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                           world_knowledge_filter="misaligned1",
                                                           cxn=cxn, print_output=True)
            acc_mis2 = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                           world_knowledge_filter="misaligned2",
                                                           cxn=cxn, print_output=True)

            res[cxn]["all"] = acc_full
            res[cxn]["aligned"] = acc_al
            res[cxn]["misaligned"] = acc_mis
            res[cxn]["aligned1"] = acc_al1
            res[cxn]["misaligned1"] = acc_mis1
            res[cxn]["aligned2"] = acc_al2
            res[cxn]["misaligned2"] = acc_mis2

            for v in plain_verbs:
                # print("Evaluating model:", m, "revision:", r, "construction:", cxn, "verb:", v)
                acc_v_all = print_results_for_data_file_accuracy(la_file, log_probs=True, verb=v, cxn=cxn,
                                                                 print_output=True)
                acc_v_al = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                                world_knowledge_filter="aligned",
                                                                verb=v, cxn=cxn, print_output=True)
                acc_v_mis = print_results_for_data_file_accuracy(la_file, log_probs=True,
                                                                 world_knowledge_filter="misaligned",
                                                                 verb=v, cxn=cxn, print_output=True)
                res[cxn][f"verb_{v}_all"] = acc_v_all
                res[cxn][f"verb_{v}_aligned"] = acc_v_al
                res[cxn][f"verb_{v}_misaligned"] = acc_v_mis

        # print("Evaluating BLIMP")
        # acc_blimp = print_results_blimp_comps(blimp_file, print_output=True)
        # res["blimp"]["all"] = acc_blimp
        # print("Evaluating COMPS")
        # acc_comps = print_results_blimp_comps(comps_file, print_output=True)
        # res["comps"]["all"] = acc_comps
        # print("Evaluating EWOK")
        # acc_ewok = print_results_ewok(ewok_file, print_output=True)
        # res["ewok"]["all"] = acc_ewok

        res_model[r] = res

        # create dataframe from res_model, and save it to tsv
        # columns are revision, construction, world_knowledge_filter, accuracy
        rows = []
        for rev in res_model:
            for cxn in res_model[rev]:
                for wkf in res_model[rev][cxn]:
                    acc = res_model[rev][cxn][wkf]
                    rows.append({
                        "revision": rev,
                        "construction": cxn,
                        "world_knowledge_filter": wkf,
                        "accuracy": acc
                    })

        res_df = pd.DataFrame(rows)
        output_dir = os.path.join("outputs_summary", m)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "accuracy_summary_la.tsv")
        res_df.to_csv(output_file, sep="\t", index=False)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--syntax", action="store_true")
    args = parser.parse_args()

    if args.syntax:
        models2 = [
            "jhu-clsp/ettin-decoder-1b"
        ]
        eval_syntax(models2)

    else:
        if args.model_name == "ettin":
            eval_ettin()

        if args.model_name == "olmo":
            eval_olmo()
