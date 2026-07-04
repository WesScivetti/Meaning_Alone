from argparse import ArgumentParser
import re
import pandas as pd
from tqdm import tqdm


import re
# cxn = "let alone"
# print(s1)
# s1_and = re.sub(cxn, "and", s1)
# print(s1_and)
# s2 = re.sub("couldn't", "could", s1)
# print(s2)
# s2_and = re.sub(cxn, "and", s2)
# print(s2_and)
#
# s1_split_part1, s1_split_part2 = s1.split(",")
# #print(s1_split_part1)
# s1_split_part1_split = s1_split_part1.split()
# s1_split_part1_split = [w.lower() for w in s1_split_part1_split if w != "I"]
# #print(s1_split_part1_split[-3:])
# noun_part = " ".join(s1_split_part1_split[-3:])
# verb_part = " ".join(s1_split_part1_split[:-3])
# #print(verb_part)
# #print(noun_part)
# new_sent = noun_part + "," + s1_split_part2.rstrip(".") + ", " + verb_part + "."
# s3 = new_sent.capitalize()
# print(s3)
# s3_and = re.sub(cxn, "and", s3)
# print(s3_and)
# pron = s1.split()[0]
# pron = pron if pron == "I" else pron.lower()
# #print(pron)
# s4_pattern = cxn + " " + verb_part
# s4 = re.sub(cxn, s4_pattern, s1)
# print(s4)
# s4_and = re.sub(cxn, "and", s4)
# print(s4_and)
# s5_pattern = pron + " couldn't "
# s5 = re.sub(s5_pattern, "", s4)
# print(s5)
# s5_and = re.sub(cxn, "and", s5)
# print(s5_and)
# if pron == "you":
#   opposite = "me"
# if pron == "we":
#   opposite = "them"
# if pron == "they":
#   opposite = "us"
# if pron == "I":
#   opposite = "you"
#
# s6_pattern = cxn + " " + opposite
# s6 = re.sub(cxn, s6_pattern, s1)
# print(s6)
# s6_and = re.sub(cxn, "and", s6)
# print(s6_and)



def manipulate(df):

    #We only need let-alone for the manipulations
    df = df[df["Cxn"]=="letalone"]
    cols = df.columns.tolist()
    cols.append("manipulation")

    # Create new dataframe to hold syntactic data
    new_df = pd.DataFrame(columns=cols)


    list_of_rows = []
    for r in tqdm(df.index):
        row = df.loc[r].copy()
        sent = row["Sentence"].split(".")[0] + "."
        #print(sent)

        for cxn in ["let alone", "much less", "not to mention", "never mind", "and"]:
            new_row = row.copy()
            s1 = re.sub("let alone", cxn, sent)
            s2 = re.sub("couldn't", "could", s1)
            s2 = re.sub("can't", "can", s2)
            s1_split_part1, s1_split_part2 = s1.split(",")
            # print(s1_split_part1)
            s1_split_part1_split = s1_split_part1.split()
            s1_split_part1_split = [w.lower() for w in s1_split_part1_split if w != "I"]
            # print(s1_split_part1_split[-3:])
            noun_part = " ".join(s1_split_part1_split[-3:])
            verb_part = " ".join(s1_split_part1_split[:-3])
            # print(verb_part)
            # print(noun_part)
            new_sent = noun_part + "," + s1_split_part2.rstrip(".") + ", " + verb_part + "."
            s3 = new_sent.capitalize()
            pron = s1.split()[0]
            pron = pron if pron == "I" else pron.lower()
            # print(pron)
            s4_pattern = cxn + " " + verb_part
            s4 = re.sub(cxn, s4_pattern, s1)
            #print(s4)
            s5_pattern = pron + " couldn't "
            s5 = re.sub(s5_pattern, "", s4)
            #print(s5)
            if pron == "you":
                opposite = "me"
            elif pron == "we":
                opposite = "them"
            elif pron == "they":
                opposite = "us"
            else:
                opposite = "you"

            s6_pattern = cxn + " " + opposite
            s6 = re.sub(cxn, s6_pattern, s1)
            #print(s6)
            manipulations = ["base", "No_NPI", "Psuedocleft", "CP Conjunction", "VP Conjunction", "VP Gap Conjunction"]
            sents = [s1, s2, s3, s4, s5, s6]
            for i in range(len(sents)):
                temp_row = new_row.copy()
                temp_row["Sentence"] = sents[i]
                temp_row["Cxn"] = cxn
                temp_row["manipulation"] = manipulations[i]
                list_of_rows.append(temp_row)

    new_df = pd.DataFrame(list_of_rows)

    return new_df

def fix_wrong_rows(df):
    """fixes the NPI rows and the VP conjunction rows"""
    df["old_index"] = df.index
    df = df[df["manipulation"].isin(["No_NPI", "VP Conjunction"])]
    df = df[~df["Sentence"].str.contains("couldn't", na=False)]
    df = df[~df["Sentence"].str.contains("could", na=False)]
    for r in df.index:
        row = df.loc[r]
        if "couldn't" in row["Sentence"]:
            continue
        else:
            if row["manipulation"] == "No_NPI":
                new_sent = re.sub("can't", "can", row["Sentence"])
                new_sent = re.sub("weren't able to", "were able to", new_sent)
                df.at[r, "Sentence"] = new_sent

            elif row["manipulation"] == "VP Conjunction":
                sent = row["Sentence"]
                sent_split = sent.split(",")
                back_part_verb, back_part_noun = sent_split[1].split()[:-3], sent_split[1].split()[-3:]

                back_part_verb_str = " ".join(back_part_verb)
                back_part_noun_str = " ".join(back_part_noun)
                back_part_verb_str = re.sub(r" ([A-Za-z]+) weren't able to", "", back_part_verb_str)
                back_part_verb_str = re.sub(r" ([A-Za-z]+) can't", "", back_part_verb_str)

                #print(back_part_verb_str)
                new_sent = sent_split[0] + ", " + back_part_verb_str + " " + back_part_noun_str
                df.at[r, "Sentence"] = new_sent

    return df

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input TSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output TSV file")
    parser.add_argument("--fix", action="store_true", help="fixes NPI rows and VP conjunctions")
    args = parser.parse_args()

    if args.fix:
        df = pd.read_csv(args.input_csv, sep="\t")
        new_df = fix_wrong_rows(df)
        new_df.to_csv(args.output_csv, sep="\t", index=False)

    else:
        df = pd.read_csv(args.input_csv, sep="\t")
        new_df = manipulate(df)
        new_df.to_csv(args.output_csv, sep="\t", index=False)


